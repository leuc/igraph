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
#include "igraph_random.h"
#include "igraph_memory.h"
#include "core/math.h"
#include "core/interruption.h"
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

/* Barnes-Hut OctTree Node */
typedef struct {
    igraph_real_t mass;
    igraph_real_t mass_cx, mass_cy, mass_cz;
    igraph_real_t size;
    igraph_integer_t sub[8];   /* Up to 8 children for 3D OctTree */
    igraph_integer_t node_id;
} bh_node_t;

typedef struct {
    bh_node_t *nodes;
    igraph_integer_t capacity;
    igraph_integer_t count;
} bh_tree_t;

/* Recursive in-place OctTree builder */
static igraph_integer_t build_bh_tree(bh_tree_t *tree, igraph_integer_t *indices, igraph_integer_t *temp, igraph_integer_t len, fa2_soa_t *nodes, igraph_bool_t is_3d) {
    if (len == 0) return -1;

    igraph_integer_t node_ptr = tree->count++;
    if (tree->count > tree->capacity) {
        tree->capacity *= 2;
        tree->nodes = (bh_node_t*) realloc(tree->nodes, tree->capacity * sizeof(bh_node_t));
    }
    bh_node_t *n = &tree->nodes[node_ptr];

    if (len == 1) {
        igraph_integer_t idx = indices[0];
        n->node_id = idx;
        n->mass = nodes->mass[idx];
        n->mass_cx = nodes->x[idx];
        n->mass_cy = nodes->y[idx];
        n->mass_cz = is_3d ? nodes->z[idx] : 0.0;
        n->size = 0.0;
        for(int i=0; i<8; ++i) n->sub[i] = -1;
        return node_ptr;
    }

    n->node_id = -1;
    igraph_real_t mass = 0, cx = 0, cy = 0, cz = 0;

    for (igraph_integer_t i = 0; i < len; i++) {
        igraph_integer_t idx = indices[i];
        igraph_real_t m = nodes->mass[idx];
        mass += m;
        cx += nodes->x[idx] * m;
        cy += nodes->y[idx] * m;
        if (is_3d) cz += nodes->z[idx] * m;
    }
    cx /= mass; cy /= mass; if (is_3d) cz /= mass;

    igraph_real_t max_dist = 0;
    for (igraph_integer_t i = 0; i < len; i++) {
        igraph_integer_t idx = indices[i];
        igraph_real_t dx = nodes->x[idx] - cx;
        igraph_real_t dy = nodes->y[idx] - cy;
        igraph_real_t dz = is_3d ? (nodes->z[idx] - cz) : 0.0;
        igraph_real_t dist = sqrt(dx*dx + dy*dy + dz*dz);
        if (dist > max_dist) max_dist = dist;
    }

    n->mass = mass; n->mass_cx = cx; n->mass_cy = cy; n->mass_cz = cz;
    n->size = 2.0 * max_dist;

    igraph_integer_t q_counts[8] = {0};
    for(igraph_integer_t i = 0; i < len; i++) {
        igraph_integer_t idx = indices[i];
        igraph_integer_t octant = 0;
        if (nodes->x[idx] >= cx) octant |= 1;
        if (nodes->y[idx] >= cy) octant |= 2;
        if (is_3d && nodes->z[idx] >= cz) octant |= 4;
        q_counts[octant]++;
    }

    igraph_integer_t offsets[8] = {0};
    igraph_integer_t cursors[8] = {0};
    for(int i=1; i<8; i++) offsets[i] = offsets[i-1] + q_counts[i-1];
    for(int i=0; i<8; i++) cursors[i] = offsets[i];

    for(igraph_integer_t i = 0; i < len; i++) {
        igraph_integer_t idx = indices[i];
        igraph_integer_t octant = 0;
        if (nodes->x[idx] >= cx) octant |= 1;
        if (nodes->y[idx] >= cy) octant |= 2;
        if (is_3d && nodes->z[idx] >= cz) octant |= 4;
        temp[cursors[octant]++] = idx;
    }

    for(igraph_integer_t i = 0; i < len; i++) indices[i] = temp[i];

    igraph_integer_t subs[8];
    for(int i=0; i<8; i++) {
        subs[i] = build_bh_tree(tree, indices + offsets[i], temp + offsets[i], q_counts[i], nodes, is_3d);
    }

    n = &tree->nodes[node_ptr];
    for(int i=0; i<8; i++) n->sub[i] = subs[i];

    return node_ptr;
}

/* Recursive Force Application */
static void apply_bh_force(bh_tree_t *tree, igraph_integer_t node_id, igraph_integer_t target_idx, fa2_soa_t *nodes, igraph_real_t theta, igraph_real_t scaling_ratio, igraph_bool_t is_3d, igraph_real_t *local_dx, igraph_real_t *local_dy, igraph_real_t *local_dz) {
    if (node_id < 0) return;
    bh_node_t *r = &tree->nodes[node_id];

    if (r->node_id >= 0) {
        if (r->node_id != target_idx) {
            igraph_real_t xDist = nodes->x[target_idx] - r->mass_cx;
            igraph_real_t yDist = nodes->y[target_idx] - r->mass_cy;
            igraph_real_t zDist = is_3d ? (nodes->z[target_idx] - r->mass_cz) : 0.0;
            igraph_real_t dist2 = xDist*xDist + yDist*yDist + zDist*zDist;
            if (dist2 > 0) {
                igraph_real_t factor = scaling_ratio * nodes->mass[target_idx] * r->mass / dist2;
                *local_dx += xDist * factor;
                *local_dy += yDist * factor;
                if (is_3d) *local_dz += zDist * factor;
            }
        }
    } else {
        igraph_real_t xDist = nodes->x[target_idx] - r->mass_cx;
        igraph_real_t yDist = nodes->y[target_idx] - r->mass_cy;
        igraph_real_t zDist = is_3d ? (nodes->z[target_idx] - r->mass_cz) : 0.0;
        igraph_real_t dist = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);

        if (dist * theta > r->size) {
            if (dist > 0) {
                igraph_real_t factor = scaling_ratio * nodes->mass[target_idx] * r->mass / (dist*dist);
                *local_dx += xDist * factor;
                *local_dy += yDist * factor;
                if (is_3d) *local_dz += zDist * factor;
            }
        } else {
            for (int i=0; i<8; i++) apply_bh_force(tree, r->sub[i], target_idx, nodes, theta, scaling_ratio, is_3d, local_dx, local_dy, local_dz);
        }
    }
}

/**
 * Internal backend function to handle both 2D and 3D implementations
 */
static igraph_error_t igraph_i_layout_forceatlas2(
        const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t iterations,
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

    bh_tree_t tree = {0};
    igraph_integer_t *bh_indices = NULL, *bh_temp = NULL;
    if (barnes_hut_optimize) {
        tree.capacity = no_of_nodes * (is_3d ? 8 : 4);
        tree.nodes = IGRAPH_MALLOC((size_t)tree.capacity * sizeof(bh_node_t));
        bh_indices = IGRAPH_MALLOC((size_t)no_of_nodes * sizeof(igraph_integer_t));
        bh_temp = IGRAPH_MALLOC((size_t)no_of_nodes * sizeof(igraph_integer_t));
    }

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
            tree.count = 0;
            for(i = 0; i < no_of_nodes; i++) bh_indices[i] = i;
            igraph_integer_t root = build_bh_tree(&tree, bh_indices, bh_temp, no_of_nodes, &nodes, is_3d);

            #pragma omp parallel for schedule(dynamic, 64)
            for (i = 0; i < no_of_nodes; i++) {
                igraph_real_t local_dx = 0.0, local_dy = 0.0, local_dz = 0.0;
                apply_bh_force(&tree, root, i, &nodes, barnes_hut_theta, scaling_ratio, is_3d, &local_dx, &local_dy, &local_dz);
                nodes.dx[i] += local_dx;
                nodes.dy[i] += local_dy;
                if (is_3d) nodes.dz[i] += local_dz;
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

            igraph_real_t w_eff = (edge_weight_influence == 1.0) ? weight : pow(weight, edge_weight_influence);
            igraph_real_t factor_u = outbound_attraction_distribution ? (-outbound_att_comp * w_eff / nodes.mass[u]) : (-outbound_att_comp * w_eff);

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
    }

    /* Final Matrix Transfer */
    #pragma omp parallel for simd schedule(static)
    for (i = 0; i < no_of_nodes; i++) {
        MATRIX(*res, i, 0) = nodes.x[i];
        MATRIX(*res, i, 1) = nodes.y[i];
        if (is_3d) MATRIX(*res, i, 2) = nodes.z[i];
    }

    if (barnes_hut_optimize) {
        IGRAPH_FREE(tree.nodes);
        IGRAPH_FREE(bh_indices);
        IGRAPH_FREE(bh_temp);
    }

    IGRAPH_FREE(nodes.x); IGRAPH_FREE(nodes.y); IGRAPH_FREE(nodes.dx); IGRAPH_FREE(nodes.dy);
    IGRAPH_FREE(nodes.old_dx); IGRAPH_FREE(nodes.old_dy); IGRAPH_FREE(nodes.mass);
    if (is_3d) { IGRAPH_FREE(nodes.z); IGRAPH_FREE(nodes.dz); IGRAPH_FREE(nodes.old_dz); }
    igraph_vector_int_destroy(&degrees);
    IGRAPH_FINALLY_CLEAN(1);

    return IGRAPH_SUCCESS;
}

/* Public Wrapper: 2D ForceAtlas2 */
igraph_error_t igraph_layout_forceatlas2(
        const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t iterations,
        igraph_bool_t outbound_attraction_distribution,
        igraph_real_t edge_weight_influence,
        igraph_real_t jitter_tolerance,
        igraph_bool_t barnes_hut_optimize,
        igraph_real_t barnes_hut_theta,
        igraph_real_t scaling_ratio,
        igraph_bool_t strong_gravity_mode,
        igraph_real_t gravity,
        const igraph_vector_t *weights) {

    return igraph_i_layout_forceatlas2(graph, res, iterations, outbound_attraction_distribution, edge_weight_influence, jitter_tolerance, barnes_hut_optimize, barnes_hut_theta, scaling_ratio, strong_gravity_mode, gravity, weights, 0);
}

/* Public Wrapper: 3D ForceAtlas2 */
igraph_error_t igraph_layout_forceatlas2_3d(
        const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t iterations,
        igraph_bool_t outbound_attraction_distribution,
        igraph_real_t edge_weight_influence,
        igraph_real_t jitter_tolerance,
        igraph_bool_t barnes_hut_optimize,
        igraph_real_t barnes_hut_theta,
        igraph_real_t scaling_ratio,
        igraph_bool_t strong_gravity_mode,
        igraph_real_t gravity,
        const igraph_vector_t *weights) {

    return igraph_i_layout_forceatlas2(graph, res, iterations, outbound_attraction_distribution, edge_weight_influence, jitter_tolerance, barnes_hut_optimize, barnes_hut_theta, scaling_ratio, strong_gravity_mode, gravity, weights, 1);
}
