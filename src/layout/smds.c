/*
 * Spherical MDS layout for igraph.
 * Based on stochastic gradient descent optimization.
 *
 * Internal coordinates: colatitude theta in [0, pi], azimuth phi in [0, 2*pi].
 * Output: Cartesian (x, y, z) on the unit sphere.
 *
 * Paper reference: Spherical Multi-Dimensional Scaling (SMDS)
 * using the spherical law of cosines for colatitude:
 *   cos(delta) = cos(theta_1)*cos(theta_2) + sin(theta_1)*sin(theta_2)*cos(phi_1 - phi_2)
 */

#include "igraph_layout.h"

#include "igraph_components.h"
#include "igraph_interface.h"
#include "igraph_paths.h"
#include "igraph_progress.h"
#include "igraph_random.h"
#include "igraph_step.h"

#include "core/interruption.h"
#include "core/math.h" /* M_PI */

#include <math.h>

/* Calculates a convergent schedule array of learning rates
 * mirroring the Python `schedule_convergent` implementation.
 */
static igraph_error_t igraph_i_smds_schedule(const igraph_matrix_t *d,
                                             igraph_vector_t *etas,
                                             igraph_int_t t_max,
                                             igraph_real_t eps,
                                             igraph_int_t t_maxmax) {
    igraph_int_t n = igraph_matrix_nrow(d);
    igraph_real_t w_min = 10000.0, w_max = 0.0;
    igraph_int_t i, j, t;
    igraph_real_t eta_max, eta_min, lamb, eta_switch;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && MATRIX(*d, i, j) > 0) {
                igraph_real_t w_ij = 1.0 / (MATRIX(*d, i, j) * MATRIX(*d, i, j));
                if (w_ij < w_min) w_min = w_ij;
                if (w_ij > w_max) w_max = w_ij;
            }
        }
    }

    /* Fallback if distance matrix is malformed or graph is trivial */
    if (w_max == 0.0) w_max = 1.0;
    if (w_min == 10000.0) w_min = 0.1;

    eta_max = 1.0 / w_min;
    eta_min = eps / w_max;
    lamb = log(eta_max / eta_min) / (t_max - 1.0);
    eta_switch = 1.0 / w_max;

    IGRAPH_CHECK(igraph_vector_resize(etas, t_maxmax));

    for (t = 0; t < t_maxmax; t++) {
        igraph_real_t eta = eta_max * exp(-lamb * t);
        if (eta < eta_switch) break;
        VECTOR(*etas)[t] = eta;
    }

    igraph_int_t tau = t;
    for (; t < t_maxmax; t++) {
        igraph_real_t eta = eta_switch / (1.0 + lamb * (t - tau));
        VECTOR(*etas)[t] = eta;
    }

    return IGRAPH_SUCCESS;
}

/**
 * \function igraph_layout_mds_spherical
 * \brief Place the vertices on a sphere using multidimensional scaling.
 *
 * Uses stochastic gradient descent to optimize vertex positions on a sphere
 * so that geodesic (great-circle) distances approximate the given shortest-path
 * distances. The result is a 3-column matrix containing Cartesian (x, y, z)
 * coordinates on the unit sphere.
 *
 * \param graph A graph object.
 * \param res Pointer to an initialized matrix object. This will contain the
 * result as an n-by-3 matrix of Cartesian coordinates on the unit sphere.
 * It will be resized if needed.
 * \param dist The distance matrix. If null, shortest paths will be used.
 * \param num_iter Number of iterations for the stochastic gradient descent.
 * \param lr_cap The cap for the learning rate step.
 * \return Error code.
 */
igraph_error_t igraph_layout_mds_spherical(const igraph_t *graph, igraph_matrix_t *res,
                                           const igraph_matrix_t *dist,
                                           igraph_int_t num_iter, igraph_real_t lr_cap) {

    const igraph_int_t no_of_nodes = igraph_vcount(graph);
    igraph_matrix_t d;
    igraph_matrix_t angles;
    igraph_vector_t etas;
    igraph_vector_int_t indices;
    igraph_real_t max_dist = 0.0;
    igraph_int_t i, j;

    if (no_of_nodes <= 1) {
        IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 3));
        igraph_matrix_null(res);
        return IGRAPH_SUCCESS;
    }

    /* 1. Distance Matrix Setup */
    if (dist == 0) {
        IGRAPH_MATRIX_INIT_FINALLY(&d, no_of_nodes, no_of_nodes);
        IGRAPH_CHECK(igraph_distances(graph, NULL, &d, igraph_vss_all(), igraph_vss_all(), IGRAPH_ALL));
    } else {
        IGRAPH_CHECK(igraph_matrix_init_copy(&d, dist));
        IGRAPH_FINALLY(igraph_matrix_destroy, &d);
    }

    for (i = 0; i < no_of_nodes; i++) {
        MATRIX(d, i, i) = 0.0;
        for (j = 0; j < no_of_nodes; j++) {
            if (MATRIX(d, i, j) > max_dist) max_dist = MATRIX(d, i, j);
        }
    }

    /* Scale heuristic: d *= (pi / d_max) */
    if (max_dist > 0.0) {
        igraph_real_t scale = M_PI / max_dist;
        igraph_matrix_scale(&d, scale);
    }

    /* 2. Initialize angular positions: theta in [0, pi], phi in [0, 2*pi] */
    IGRAPH_MATRIX_INIT_FINALLY(&angles, no_of_nodes, 2);
    for (i = 0; i < no_of_nodes; i++) {
        MATRIX(angles, i, 0) = RNG_UNIF(0.0, M_PI);
        MATRIX(angles, i, 1) = RNG_UNIF(0.0, 2.0 * M_PI);
    }

    /* Initialize res as n-by-3 Cartesian output */
    IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 3));
    for (i = 0; i < no_of_nodes; i++) {
        igraph_real_t theta = MATRIX(angles, i, 0);
        igraph_real_t phi   = MATRIX(angles, i, 1);
        MATRIX(*res, i, 0) = sin(theta) * cos(phi);
        MATRIX(*res, i, 1) = sin(theta) * sin(phi);
        MATRIX(*res, i, 2) = cos(theta);
    }

    /* 3. Prepare pairs for SGD */
    igraph_int_t num_pairs = (no_of_nodes * (no_of_nodes - 1)) / 2;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&indices, num_pairs * 2);
    igraph_int_t k = 0;
    for (i = 0; i < no_of_nodes; i++) {
        for (j = 0; j < i; j++) {
            VECTOR(indices)[k++] = i;
            VECTOR(indices)[k++] = j;
        }
    }

    /* 4. Prepare schedule */
    IGRAPH_VECTOR_INIT_FINALLY(&etas, 0);
    IGRAPH_CHECK(igraph_i_smds_schedule(&d, &etas, 30, 0.01, num_iter));

    /* Precomputed trig lookup tables — avoids 10 trig calls per pair in
     * the inner loop. sin/cos are O(n) per iteration vs O(n²) pairs. */
    igraph_vector_t sin_theta, cos_theta, sin_phi, cos_phi;
    IGRAPH_VECTOR_INIT_FINALLY(&sin_theta, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&cos_theta, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&sin_phi, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&cos_phi, no_of_nodes);

    /* 5. SGD Optimization Loop */
    IGRAPH_PROGRESS("Spherical MDS layout", 0, NULL);
    for (igraph_int_t step_idx = 0; step_idx < num_iter; step_idx++) {
        igraph_real_t step = VECTOR(etas)[step_idx];

        IGRAPH_PROGRESS("Spherical MDS layout",
                        100.0 * step_idx / num_iter, NULL);

        /* Precompute sin/cos for all vertices — O(n), amortized over O(n²) pairs */
        for (i = 0; i < no_of_nodes; i++) {
            VECTOR(sin_theta)[i] = sin(MATRIX(angles, i, 0));
            VECTOR(cos_theta)[i] = cos(MATRIX(angles, i, 0));
            VECTOR(sin_phi)[i]   = sin(MATRIX(angles, i, 1));
            VECTOR(cos_phi)[i]   = cos(MATRIX(angles, i, 1));
        }

        /* Shuffle indices for stochasticity */
        for (i = num_pairs - 1; i > 0; i--) {
            igraph_int_t swap_idx = RNG_INTEGER(0, i);
            /* Swap i and swap_idx pairs */
            igraph_int_t tmp1 = VECTOR(indices)[i * 2];
            igraph_int_t tmp2 = VECTOR(indices)[i * 2 + 1];
            VECTOR(indices)[i * 2] = VECTOR(indices)[swap_idx * 2];
            VECTOR(indices)[i * 2 + 1] = VECTOR(indices)[swap_idx * 2 + 1];
            VECTOR(indices)[swap_idx * 2] = tmp1;
            VECTOR(indices)[swap_idx * 2 + 1] = tmp2;
        }

        for (k = 0; k < num_pairs; k++) {
            IGRAPH_ALLOW_INTERRUPTION();

            igraph_int_t u = VECTOR(indices)[k * 2];
            igraph_int_t v = VECTOR(indices)[k * 2 + 1];

            igraph_real_t wc = step;
            if (wc > lr_cap) wc = lr_cap;

            igraph_real_t target_d = MATRIX(d, u, v);

            /* Lookup precomputed trig values */
            igraph_real_t st_u = VECTOR(sin_theta)[u];
            igraph_real_t ct_u = VECTOR(cos_theta)[u];
            igraph_real_t sp_u = VECTOR(sin_phi)[u];
            igraph_real_t cp_u = VECTOR(cos_phi)[u];
            igraph_real_t st_v = VECTOR(sin_theta)[v];
            igraph_real_t ct_v = VECTOR(cos_theta)[v];
            igraph_real_t sp_v = VECTOR(sin_phi)[v];
            igraph_real_t cp_v = VECTOR(cos_phi)[v];

            /* cos(delta) = cos(t_i)*cos(t_j) + sin(t_i)*sin(t_j)*cos(phi_i - phi_j)
             *            = ct_u*ct_v + st_u*st_v*(cp_u*cp_v + sp_u*sp_v) */
            igraph_real_t cos_dp = cp_u * cp_v + sp_u * sp_v;
            igraph_real_t val = ct_u * ct_v + st_u * st_v * cos_dp;
            if (val > 1.0) val = 1.0;
            if (val < -1.0) val = -1.0;

            igraph_real_t delta = acos(val);
            igraph_real_t factor = 2.0 * (delta - target_d);

            /* Gradient via precomputed values — no trig calls */
            igraph_real_t denom = sqrt(1.0 - val * val);
            if (denom < 1e-8) {
                continue;
            }

            igraph_real_t sin_dp = sp_u * cp_v - cp_u * sp_v;
            igraph_real_t inv_denom = 1.0 / denom;

            /* d(delta)/d(theta_i) */
            igraph_real_t dd_ti = (st_u * ct_v - ct_u * st_v * cos_dp) * inv_denom;
            /* d(delta)/d(phi_i) */
            igraph_real_t dd_pi = st_u * st_v * sin_dp * inv_denom;
            /* d(delta)/d(theta_j) */
            igraph_real_t dd_tj = (ct_u * st_v - st_u * ct_v * cos_dp) * inv_denom;
            /* d(delta)/d(phi_j) = -d(delta)/d(phi_i) */

            igraph_real_t g = wc * factor;

            MATRIX(angles, u, 0) -= g * dd_ti;
            MATRIX(angles, u, 1) -= g * dd_pi;
            MATRIX(angles, v, 0) -= g * dd_tj;
            MATRIX(angles, v, 1) += g * dd_pi;
        }

        /* Convert angular positions to Cartesian for step readback.
         * Reuse precomputed sin/cos tables from the inner loop. */
        for (i = 0; i < no_of_nodes; i++) {
            MATRIX(*res, i, 0) = VECTOR(sin_theta)[i] * VECTOR(cos_phi)[i];
            MATRIX(*res, i, 1) = VECTOR(sin_theta)[i] * VECTOR(sin_phi)[i];
            MATRIX(*res, i, 2) = VECTOR(cos_theta)[i];
        }
        IGRAPH_STEP(res, NULL);
    }
    IGRAPH_PROGRESS("Spherical MDS layout", 100, NULL);

    igraph_vector_destroy(&cos_phi);
    igraph_vector_destroy(&sin_phi);
    igraph_vector_destroy(&cos_theta);
    igraph_vector_destroy(&sin_theta);
    igraph_vector_destroy(&etas);
    igraph_vector_int_destroy(&indices);
    igraph_matrix_destroy(&angles);
    igraph_matrix_destroy(&d);
    IGRAPH_FINALLY_CLEAN(8);

    return IGRAPH_SUCCESS;
}
