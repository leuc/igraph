/*
   igraph library.
   Copyright (C) 2026  The igraph development team <igraph@igraph.org>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "igraph_layout.h"
#include "igraph_interface.h"
#include "igraph_progress.h"
#include "igraph_statusbar.h"
#include "igraph_step.h"
#include "igraph_random.h"
#include "igraph_memory.h"
#include "core/math.h"
#include "core/interruption.h"
#include "igraph_barnes_hut.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Structure of Arrays (SoA) layout for AVX2 vectorization */
typedef struct {
    igraph_real_t *x, *y, *z;
    igraph_real_t *dx, *dy, *dz;
    igraph_real_t *old_dx, *old_dy, *old_dz;
    igraph_real_t *mass;
} fa2_soa_t;

typedef struct {
    igraph_real_t scaling_ratio;
    igraph_bool_t is_3d;
} fa2_bh_data_t;

static void fa2_repulsion_kernel(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t dx,
    igraph_real_t dy,
    igraph_real_t dz,
    igraph_real_t dist_sq,
    igraph_real_t force[3],
    void *user_data
) {
    fa2_bh_data_t *data = (fa2_bh_data_t *)user_data;

    /* LAYOUT EXCLUSIVE LOGIC: Handle coordinate collisions deterministically.
       (p2->id == -1 indicates p2 is a macroscopic pseudo-node) */
    if (dist_sq < 1e-12) {
        igraph_real_t epsilon = (p2->id == -1 || p1->id > p2->id) ? 1e-5 : -1e-5;
        dx += epsilon;
        dy += epsilon;
        if (data->is_3d) dz += epsilon;
        dist_sq = dx*dx + dy*dy + dz*dz;
    }

    /* ForceAtlas2 base repulsion: Kr * (m1*m2) / d */
    /* Since we divide by dist_sq below to scale the vector, it calculates: (Kr * m1 * m2 * distance_vector) / d^2 */
    igraph_real_t factor = data->scaling_ratio * p1->mass * p2->mass / dist_sq;

    force[0] = factor * dx;
    force[1] = factor * dy;
    force[2] = factor * dz; /* Safe in 2D because dz is 0.0 */
}

/**
 * Internal backend function to handle both 2D and 3D implementations
 */
static igraph_error_t igraph_i_layout_forceatlas2(
        const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t iterations,
        igraph_bool_t linlog_mode,
        igraph_bool_t outbound_attraction_distribution,
        igraph_real_t edge_weight_influence,
        igraph_real_t jitter_tolerance,
        igraph_bool_t barnes_hut_optimize,
        igraph_real_t barnes_hut_theta,
        igraph_real_t scaling_ratio,
        igraph_bool_t strong_gravity_mode,
        igraph_real_t gravity,
        const igraph_vector_t *weights,
        igraph_bool_t is_3d) {

    igraph_integer_t no_of_nodes = igraph_vcount(graph);
    igraph_integer_t no_of_edges = igraph_ecount(graph);
    igraph_integer_t i, j, iter;

    fa2_soa_t nodes;
    igraph_vector_int_t degrees;

    igraph_real_t speed = 1.0, speed_efficiency = 1.0;
    igraph_real_t outbound_att_comp = 1.0, total_mass = 0.0;

    /* Proper resize for dimensions */
    IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, is_3d ? 3 : 2));

    nodes.x = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);
    nodes.y = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);
    nodes.z = is_3d ? IGRAPH_CALLOC(no_of_nodes, igraph_real_t) : NULL;
    nodes.dx = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);
    nodes.dy = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);
    nodes.dz = is_3d ? IGRAPH_CALLOC(no_of_nodes, igraph_real_t) : NULL;
    nodes.old_dx = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);
    nodes.old_dy = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);
    nodes.old_dz = is_3d ? IGRAPH_CALLOC(no_of_nodes, igraph_real_t) : NULL;
    nodes.mass = IGRAPH_CALLOC(no_of_nodes, igraph_real_t);

    IGRAPH_VECTOR_INT_INIT_FINALLY(&degrees, no_of_nodes);
    IGRAPH_CHECK(igraph_degree(graph, &degrees, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS));

    for (i = 0; i < no_of_nodes; i++) {
        nodes.mass[i] = 1.0 + VECTOR(degrees)[i];
        total_mass += nodes.mass[i];
        nodes.x[i] = igraph_rng_get_unif(igraph_rng_default(), 0.0, 1.0);
        nodes.y[i] = igraph_rng_get_unif(igraph_rng_default(), 0.0, 1.0);
        if (is_3d) nodes.z[i] = igraph_rng_get_unif(igraph_rng_default(), 0.0, 1.0);
    }

    if (outbound_attraction_distribution && no_of_nodes > 0) outbound_att_comp = total_mass / no_of_nodes;

    igraph_bh_tree_t tree = {0};
    igraph_matrix_t coords = {0};
    igraph_vector_t masses = {0};
    igraph_matrix_t forces = {0};

    if (barnes_hut_optimize) {
        IGRAPH_CHECK(igraph_matrix_init(&coords, no_of_nodes, is_3d ? 3 : 2));
        IGRAPH_FINALLY(igraph_matrix_destroy, &coords);
        IGRAPH_CHECK(igraph_vector_init(&masses, no_of_nodes));
        IGRAPH_FINALLY(igraph_vector_destroy, &masses);
        IGRAPH_CHECK(igraph_matrix_init(&forces, no_of_nodes, is_3d ? 3 : 2));
        IGRAPH_FINALLY(igraph_matrix_destroy, &forces);
        {
            igraph_integer_t bh_max_level, bh_leaf_capacity;
            igraph_bh_tree_get_scaling_params(no_of_nodes, is_3d ? 3 : 2, &bh_max_level, &bh_leaf_capacity);
            IGRAPH_CHECK(igraph_bh_tree_init(&tree, is_3d ? 3 : 2, barnes_hut_theta, bh_max_level, bh_leaf_capacity));
        }
        IGRAPH_FINALLY(igraph_bh_tree_destroy, &tree);
    } else {
        IGRAPH_FINALLY(igraph_bh_tree_destroy, &tree);
        IGRAPH_FINALLY(igraph_matrix_destroy, &forces);
        IGRAPH_FINALLY(igraph_vector_destroy, &masses);
        IGRAPH_FINALLY(igraph_matrix_destroy, &coords);
    }

    fa2_bh_data_t bh_data = {
        .scaling_ratio = scaling_ratio,
        .is_3d = is_3d
    };

    for (iter = 0; iter < iterations; iter++) {
        IGRAPH_PROGRESS("ForceAtlas2: ",  (100.0 * iter) / iterations, 0);
        IGRAPH_ALLOW_INTERRUPTION();

        #pragma omp parallel for simd schedule(static)
        for (i = 0; i < no_of_nodes; i++) {
            nodes.old_dx[i] = nodes.dx[i];
            nodes.old_dy[i] = nodes.dy[i];
            nodes.dx[i] = 0; nodes.dy[i] = 0;
            if (is_3d) {
                nodes.old_dz[i] = nodes.dz[i];
                nodes.dz[i] = 0;
            }
        }

        /* Repulsion Forces */
        if (barnes_hut_optimize) {
            for (i = 0; i < no_of_nodes; i++) {
                MATRIX(coords, i, 0) = nodes.x[i];
                MATRIX(coords, i, 1) = nodes.y[i];
                if (is_3d) MATRIX(coords, i, 2) = nodes.z[i];
                VECTOR(masses)[i] = nodes.mass[i];
            }

            IGRAPH_CHECK(igraph_bh_tree_build(&tree, &coords, &masses));

            igraph_matrix_null(&forces);
            IGRAPH_CHECK(igraph_bh_apply_repulsion_from_tree(&tree, &forces, fa2_repulsion_kernel, &bh_data));

            for (i = 0; i < no_of_nodes; i++) {
                nodes.dx[i] = MATRIX(forces, i, 0);
                nodes.dy[i] = MATRIX(forces, i, 1);
                if (is_3d) nodes.dz[i] = MATRIX(forces, i, 2);
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 64)
            for (i = 0; i < no_of_nodes; i++) {
                igraph_real_t local_dx = 0.0, local_dy = 0.0, local_dz = 0.0;
                #pragma omp simd reduction(+:local_dx, local_dy, local_dz)
                for (j = 0; j < no_of_nodes; j++) {
                    if (i == j) continue;
                    igraph_real_t xDist = nodes.x[i] - nodes.x[j];
                    igraph_real_t yDist = nodes.y[i] - nodes.y[j];
                    igraph_real_t zDist = is_3d ? (nodes.z[i] - nodes.z[j]) : 0.0;
                    igraph_real_t dist_sq = xDist*xDist + yDist*yDist + zDist*zDist;

                    if (dist_sq > 0) {
                        igraph_real_t factor = scaling_ratio * nodes.mass[i] * nodes.mass[j] / dist_sq;
                        local_dx += xDist * factor;
                        local_dy += yDist * factor;
                        if (is_3d) local_dz += zDist * factor;
                    }
                }
                nodes.dx[i] += local_dx;
                nodes.dy[i] += local_dy;
                if (is_3d) nodes.dz[i] += local_dz;
            }
        }

        /* Gravity Forces */
        #pragma omp parallel for simd schedule(static)
        for (i = 0; i < no_of_nodes; i++) {
            if (strong_gravity_mode) {
                if (nodes.x[i] != 0 && nodes.y[i] != 0) {
                    igraph_real_t factor = scaling_ratio * nodes.mass[i] * gravity;
                    nodes.dx[i] -= nodes.x[i] * factor;
                    nodes.dy[i] -= nodes.y[i] * factor;
                    if (is_3d) nodes.dz[i] -= nodes.z[i] * factor;
                }
            } else {
                igraph_real_t z_sq = is_3d ? (nodes.z[i]*nodes.z[i]) : 0.0;
                igraph_real_t dist = sqrt(nodes.x[i]*nodes.x[i] + nodes.y[i]*nodes.y[i] + z_sq);
                if (dist > 0) {
                    igraph_real_t factor = nodes.mass[i] * gravity / dist;
                    nodes.dx[i] -= nodes.x[i] * factor;
                    nodes.dy[i] -= nodes.y[i] * factor;
                    if (is_3d) nodes.dz[i] -= nodes.z[i] * factor;
                }
            }
        }

        /* Attraction Forces */
        #pragma omp parallel for schedule(static)
        for (i = 0; i < no_of_edges; i++) {
            igraph_integer_t u = IGRAPH_FROM(graph, i);
            igraph_integer_t v = IGRAPH_TO(graph, i);
            if (u == v) continue;

            igraph_real_t weight = weights ? VECTOR(*weights)[i] : 1.0;
            igraph_real_t xDist = nodes.x[u] - nodes.x[v];
            igraph_real_t yDist = nodes.y[u] - nodes.y[v];
            igraph_real_t zDist = is_3d ? (nodes.z[u] - nodes.z[v]) : 0.0;

            // Calculate euclidean distance for LinLog adjustment
            igraph_real_t dist = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);

            if (dist > 0) {
                igraph_real_t w_eff = (edge_weight_influence == 1.0) ? weight : pow(weight, edge_weight_influence);
                igraph_real_t factor_u = outbound_attraction_distribution ? (-outbound_att_comp * w_eff / nodes.mass[u]) : (-outbound_att_comp * w_eff);

                // Apply the LinLog energy model F_a = log(1 + d)
                // We divide by `dist` because `xDist`, `yDist`, `zDist` already implicitly multiply the distance.
                if (linlog_mode) {
                    factor_u *= log(1.0 + dist) / dist;
                }

                #pragma omp atomic update
                nodes.dx[u] += xDist * factor_u;
                #pragma omp atomic update
                nodes.dy[u] += yDist * factor_u;
                #pragma omp atomic update
                nodes.dx[v] -= xDist * factor_u;
                #pragma omp atomic update
                nodes.dy[v] -= yDist * factor_u;

                if (is_3d) {
                    #pragma omp atomic update
                    nodes.dz[u] += zDist * factor_u;
                    #pragma omp atomic update
                    nodes.dz[v] -= zDist * factor_u;
                }
            }
        }

        /* Speed Adjustments */
        igraph_real_t total_swinging = 0.0, total_effective_traction = 0.0;

        #pragma omp parallel for simd reduction(+:total_swinging, total_effective_traction) schedule(static)
        for (i = 0; i < no_of_nodes; i++) {
            igraph_real_t dxd = nodes.old_dx[i] - nodes.dx[i];
            igraph_real_t dyd = nodes.old_dy[i] - nodes.dy[i];
            igraph_real_t dzd = is_3d ? (nodes.old_dz[i] - nodes.dz[i]) : 0.0;

            igraph_real_t dxs = nodes.old_dx[i] + nodes.dx[i];
            igraph_real_t dys = nodes.old_dy[i] + nodes.dy[i];
            igraph_real_t dzs = is_3d ? (nodes.old_dz[i] + nodes.dz[i]) : 0.0;

            total_swinging += nodes.mass[i] * sqrt(dxd*dxd + dyd*dyd + dzd*dzd);
            total_effective_traction += 0.5 * nodes.mass[i] * sqrt(dxs*dxs + dys*dys + dzs*dzs);
        }

        igraph_real_t est_opt_jt = 0.05 * sqrt(no_of_nodes);
        igraph_real_t jt_calc = est_opt_jt * total_effective_traction / ((double)no_of_nodes * no_of_nodes);
        if (jt_calc < sqrt(est_opt_jt)) jt_calc = sqrt(est_opt_jt);
        if (jt_calc > 10.0) jt_calc = 10.0;
        igraph_real_t jt = jitter_tolerance * jt_calc;

        if (total_effective_traction > 0 && (total_swinging / total_effective_traction) > 2.0) {
            if (speed_efficiency > 0.05) speed_efficiency *= 0.5;
            if (jitter_tolerance > jt) jt = jitter_tolerance;
        }

        igraph_real_t target_speed = (total_swinging == 0) ? 1e10 : (jt * speed_efficiency * total_effective_traction / total_swinging);

        if (total_swinging > jt * total_effective_traction) {
            if (speed_efficiency > 0.05) speed_efficiency *= 0.7;
        } else if (speed < 1000) {
            speed_efficiency *= 1.3;
        }

        igraph_real_t speed_diff = target_speed - speed;
        igraph_real_t max_rise_speed = 0.5 * speed;
        speed += (speed_diff < max_rise_speed) ? speed_diff : max_rise_speed;

        #pragma omp parallel for simd schedule(static)
        for (i = 0; i < no_of_nodes; i++) {
            igraph_real_t dxd = nodes.old_dx[i] - nodes.dx[i];
            igraph_real_t dyd = nodes.old_dy[i] - nodes.dy[i];
            igraph_real_t dzd = is_3d ? (nodes.old_dz[i] - nodes.dz[i]) : 0.0;
            igraph_real_t swinging = nodes.mass[i] * sqrt(dxd*dxd + dyd*dyd + dzd*dzd);

            igraph_real_t factor = speed / (1.0 + sqrt(speed * swinging));
            nodes.x[i] += nodes.dx[i] * factor;
            nodes.y[i] += nodes.dy[i] * factor;
            if (is_3d) nodes.z[i] += nodes.dz[i] * factor;
        }

        IGRAPH_STATUSF(("ForceAtlas2: iter=%d, speed=%g, efficiency=%g, swinging=%g, traction=%g\n",
                NULL, (int)iter, speed, speed_efficiency, total_swinging, total_effective_traction));

        /* Final Matrix Transfer */
        #pragma omp parallel for simd schedule(static)
        for (i = 0; i < no_of_nodes; i++) {
            MATRIX(*res, i, 0) = nodes.x[i];
            MATRIX(*res, i, 1) = nodes.y[i];
            if (is_3d) MATRIX(*res, i, 2) = nodes.z[i];
        }
        IGRAPH_STEP(res, NULL);

    }
    /* Cleanup handled by IGRAPH_FINALLY - do not call manually */

    IGRAPH_FREE(nodes.x); IGRAPH_FREE(nodes.y); IGRAPH_FREE(nodes.dx); IGRAPH_FREE(nodes.dy);
    IGRAPH_FREE(nodes.old_dx); IGRAPH_FREE(nodes.old_dy); IGRAPH_FREE(nodes.mass);
    if (is_3d) { IGRAPH_FREE(nodes.z); IGRAPH_FREE(nodes.dz); IGRAPH_FREE(nodes.old_dz); }
    igraph_vector_int_destroy(&degrees);
    IGRAPH_FINALLY_CLEAN(4);

    return IGRAPH_SUCCESS;
}

/* Public Wrapper: 2D ForceAtlas2 */
igraph_error_t igraph_layout_forceatlas2(
        const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t iterations,
        igraph_bool_t linlog_mode,
        igraph_bool_t outbound_attraction_distribution,
        igraph_real_t edge_weight_influence,
        igraph_real_t jitter_tolerance,
        igraph_bool_t barnes_hut_optimize,
        igraph_real_t barnes_hut_theta,
        igraph_real_t scaling_ratio,
        igraph_bool_t strong_gravity_mode,
        igraph_real_t gravity,
        const igraph_vector_t *weights) {

    return igraph_i_layout_forceatlas2(graph, res, iterations, linlog_mode, outbound_attraction_distribution, edge_weight_influence, jitter_tolerance, barnes_hut_optimize, barnes_hut_theta, scaling_ratio, strong_gravity_mode, gravity, weights, 0);
}

/* Public Wrapper: 3D ForceAtlas2 */
igraph_error_t igraph_layout_forceatlas2_3d(
        const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t iterations,
        igraph_bool_t linlog_mode,
        igraph_bool_t outbound_attraction_distribution,
        igraph_real_t edge_weight_influence,
        igraph_real_t jitter_tolerance,
        igraph_bool_t barnes_hut_optimize,
        igraph_real_t barnes_hut_theta,
        igraph_real_t scaling_ratio,
        igraph_bool_t strong_gravity_mode,
        igraph_real_t gravity,
        const igraph_vector_t *weights) {

    return igraph_i_layout_forceatlas2(graph, res, iterations, linlog_mode, outbound_attraction_distribution, edge_weight_influence, jitter_tolerance, barnes_hut_optimize, barnes_hut_theta, scaling_ratio, strong_gravity_mode, gravity, weights, 1);
}
