/*
 * Spherical MDS layout for igraph.
 * Based on stochastic gradient descent optimization.
 * Enhanced with Gower's Interpolation MDS for large graph scalability.
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
#include <omp.h>

/* =========================================================================
 * SUB-ROUTINES FOR UNIT TESTING & PARALLELIZATION
 * ========================================================================= */

/**
 * Precomputes lookup tables for trigonometric identities across nodes.
 */
static void igraph_i_smds_compute_trig(const igraph_matrix_t *angles,
                                       igraph_vector_t *sin_theta, igraph_vector_t *cos_theta,
                                       igraph_vector_t *sin_phi, igraph_vector_t *cos_phi,
                                       igraph_int_t n) {
    #pragma omp parallel for default(none) shared(angles, sin_theta, cos_theta, sin_phi, cos_phi, n)
    for (igraph_int_t i = 0; i < n; i++) {
        VECTOR(*sin_theta)[i] = sin(MATRIX(*angles, i, 0));
        VECTOR(*cos_theta)[i] = cos(MATRIX(*angles, i, 0));
        VECTOR(*sin_phi)[i]   = sin(MATRIX(*angles, i, 1));
        VECTOR(*cos_phi)[i]   = cos(MATRIX(*angles, i, 1));
    }
}

/**
 * Maps updated polar angles onto Cartesian space mapping a unit sphere.
 */
static void igraph_i_smds_update_cartesian(igraph_matrix_t *res,
                                           const igraph_vector_t *sin_theta, const igraph_vector_t *cos_theta,
                                           const igraph_vector_t *sin_phi, const igraph_vector_t *cos_phi,
                                           igraph_int_t n) {
    #pragma omp parallel for default(none) shared(res, sin_theta, cos_theta, sin_phi, cos_phi, n)
    for (igraph_int_t i = 0; i < n; i++) {
        MATRIX(*res, i, 0) = VECTOR(*sin_theta)[i] * VECTOR(*cos_phi)[i];
        MATRIX(*res, i, 1) = VECTOR(*sin_theta)[i] * VECTOR(*sin_phi)[i];
        MATRIX(*res, i, 2) = VECTOR(*cos_theta)[i];
    }
}

/**
 * Handles classical double-centering properties tracking Gower MDS structures.
 */
static void igraph_i_smds_gower_center(const igraph_matrix_t *d_sub,
                                       igraph_vector_t *row_means,
                                       igraph_matrix_t *Q,
                                       igraph_vector_t *q_vector,
                                       igraph_int_t l) {
    igraph_int_t i, j;
    igraph_real_t grand_sum = 0.0;

    #pragma omp parallel for reduction(+:grand_sum) private(j) default(none) shared(d_sub, row_means, l)
    for (i = 0; i < l; i++) {
        igraph_real_t row_sum = 0.0;
        for (j = 0; j < l; j++) {
            igraph_real_t val = MATRIX(*d_sub, i, j);
            row_sum += val * val;
        }
        VECTOR(*row_means)[i] = row_sum / l;
        grand_sum += row_sum;
    }
    
    igraph_real_t grand_mean = grand_sum / (l * l);

    #pragma omp parallel for private(j) default(none) shared(d_sub, row_means, Q, q_vector, grand_mean, l)
    for (i = 0; i < l; i++) {
        for (j = 0; j < l; j++) {
            igraph_real_t val = MATRIX(*d_sub, i, j);
            igraph_real_t q_val = -0.5 * (val * val - VECTOR(*row_means)[i] - VECTOR(*row_means)[j] + grand_mean);
            MATRIX(*Q, i, j) = q_val;
            if (i == j) {
                VECTOR(*q_vector)[i] = q_val;
            }
        }
    }
}

/**
 * Reconstructs out-of-sample coordinates using the landmark embedding transformation.
 */
static void igraph_i_smds_gower_interpolate(igraph_matrix_t *res,
                                            const igraph_matrix_t *res_sub,
                                            const igraph_matrix_t *d_landmark,
                                            const igraph_matrix_t *x_1_s_inv,
                                            const igraph_vector_t *q_vector,
                                            const igraph_vector_int_t *landmark_map,
                                            igraph_int_t no_of_nodes,
                                            igraph_int_t l) {
    #pragma omp parallel for default(none) shared(res, res_sub, d_landmark, x_1_s_inv, q_vector, landmark_map, no_of_nodes, l)
    for (igraph_int_t v = 0; v < no_of_nodes; v++) {
        igraph_int_t l_idx = VECTOR(*landmark_map)[v];
        if (l_idx != -1) {
            MATRIX(*res, v, 0) = MATRIX(*res_sub, l_idx, 0);
            MATRIX(*res, v, 1) = MATRIX(*res_sub, l_idx, 1);
            MATRIX(*res, v, 2) = MATRIX(*res_sub, l_idx, 2);
        } else {
            igraph_real_t x_v[3] = {0.0, 0.0, 0.0};
            for (igraph_int_t i = 0; i < l; i++) {
                igraph_real_t d_vi = MATRIX(*d_landmark, i, v);
                igraph_real_t a_vi = d_vi * d_vi;
                igraph_real_t diff = VECTOR(*q_vector)[i] - a_vi;
                
                x_v[0] += diff * MATRIX(*x_1_s_inv, i, 0);
                x_v[1] += diff * MATRIX(*x_1_s_inv, i, 1);
                x_v[2] += diff * MATRIX(*x_1_s_inv, i, 2);
            }
            x_v[0] /= (2.0 * l);
            x_v[1] /= (2.0 * l);
            x_v[2] /= (2.0 * l);

            igraph_real_t norm = sqrt(x_v[0]*x_v[0] + x_v[1]*x_v[1] + x_v[2]*x_v[2]);
            if (norm > 1e-8) {
                MATRIX(*res, v, 0) = x_v[0] / norm;
                MATRIX(*res, v, 1) = x_v[1] / norm;
                MATRIX(*res, v, 2) = x_v[2] / norm;
            } else {
                /* Thread-safe pseudo-random fallback avoiding global state race conditions */
                unsigned int seed = (unsigned int)(v + 1);
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                igraph_real_t theta = ((igraph_real_t)(seed % 1000) / 1000.0) * M_PI;
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                igraph_real_t phi   = ((igraph_real_t)(seed % 1000) / 1000.0) * 2.0 * M_PI;

                MATRIX(*res, v, 0) = sin(theta) * cos(phi);
                MATRIX(*res, v, 1) = sin(theta) * sin(phi);
                MATRIX(*res, v, 2) = cos(theta);
            }
        }
    }
}

static igraph_error_t igraph_i_smds_schedule(const igraph_matrix_t *d,
                                             igraph_vector_t *etas,
                                             igraph_int_t t_max,
                                             igraph_real_t eps,
                                             igraph_int_t t_maxmax) {
    igraph_int_t n = igraph_matrix_nrow(d);
    igraph_real_t w_min = 10000.0, w_max = 0.0;
    igraph_int_t i, j, t;
    igraph_real_t eta_max, eta_min, lamb, eta_switch;

    #pragma omp parallel for reduction(min:w_min) reduction(max:w_max) private(j) default(none) shared(d, n)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && MATRIX(*d, i, j) > 0) {
                igraph_real_t w_ij = 1.0 / (MATRIX(*d, i, j) * MATRIX(*d, i, j));
                if (w_ij < w_min) w_min = w_ij;
                if (w_ij > w_max) w_max = w_ij;
            }
        }
    }

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

/* =========================================================================
 * MAIN ORCHESTRATION ROUTINE
 * ========================================================================= */

igraph_error_t igraph_layout_mds_spherical(const igraph_t *graph, igraph_matrix_t *res,
                                           const igraph_matrix_t *dist,
                                           igraph_int_t num_iter, igraph_real_t lr_cap) {

    const igraph_int_t no_of_nodes = igraph_vcount(graph);

    if (no_of_nodes <= 1) {
        IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 3));
        igraph_matrix_null(res);
        return IGRAPH_SUCCESS;
    }

    const igraph_int_t l = 1000;

    if (no_of_nodes <= l) {
        igraph_matrix_t d;
        igraph_matrix_t angles;
        igraph_vector_t etas;
        igraph_vector_int_t indices;
        igraph_real_t max_dist = 0.0;
        igraph_int_t i, j, k;

        if (dist == NULL) {
            IGRAPH_MATRIX_INIT_FINALLY(&d, no_of_nodes, no_of_nodes);
            IGRAPH_CHECK(igraph_distances(graph, NULL, &d, igraph_vss_all(), igraph_vss_all(), IGRAPH_ALL));
        } else {
            IGRAPH_CHECK(igraph_matrix_init_copy(&d, dist));
            IGRAPH_FINALLY(igraph_matrix_destroy, &d);
        }

        #pragma omp parallel for reduction(max:max_dist) private(j) default(none) shared(d, no_of_nodes)
        for (i = 0; i < no_of_nodes; i++) {
            MATRIX(d, i, i) = 0.0;
            for (j = 0; j < no_of_nodes; j++) {
                if (MATRIX(d, i, j) > max_dist) max_dist = MATRIX(d, i, j);
            }
        }

        if (max_dist > 0.0) {
            igraph_real_t scale = M_PI / max_dist;
            igraph_matrix_scale(&d, scale);
        }

        IGRAPH_MATRIX_INIT_FINALLY(&angles, no_of_nodes, 2);
        for (i = 0; i < no_of_nodes; i++) {
            MATRIX(angles, i, 0) = RNG_UNIF(0.0, M_PI);
            MATRIX(angles, i, 1) = RNG_UNIF(0.0, 2.0 * M_PI);
        }

        IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 3));

        igraph_int_t num_pairs = (no_of_nodes * (no_of_nodes - 1)) / 2;
        IGRAPH_VECTOR_INT_INIT_FINALLY(&indices, num_pairs * 2);
        k = 0;
        for (i = 0; i < no_of_nodes; i++) {
            for (j = 0; j < i; j++) {
                VECTOR(indices)[k++] = i;
                VECTOR(indices)[k++] = j;
            }
        }

        IGRAPH_VECTOR_INIT_FINALLY(&etas, 0);
        IGRAPH_CHECK(igraph_i_smds_schedule(&d, &etas, 30, 0.01, num_iter));

        igraph_vector_t sin_theta, cos_theta, sin_phi, cos_phi;
        IGRAPH_VECTOR_INIT_FINALLY(&sin_theta, no_of_nodes);
        IGRAPH_VECTOR_INIT_FINALLY(&cos_theta, no_of_nodes);
        IGRAPH_VECTOR_INIT_FINALLY(&sin_phi, no_of_nodes);
        IGRAPH_VECTOR_INIT_FINALLY(&cos_phi, no_of_nodes);

        IGRAPH_PROGRESS("Spherical MDS layout", 0, NULL);
        for (igraph_int_t step_idx = 0; step_idx < num_iter; step_idx++) {
            igraph_real_t step = VECTOR(etas)[step_idx];

            IGRAPH_PROGRESS("Spherical MDS layout", 100.0 * step_idx / num_iter, NULL);

            igraph_i_smds_compute_trig(&angles, &sin_theta, &cos_theta, &sin_phi, &cos_phi, no_of_nodes);

            for (i = num_pairs - 1; i > 0; i--) {
                igraph_int_t swap_idx = RNG_INTEGER(0, i);
                igraph_int_t tmp1 = VECTOR(indices)[i * 2];
                igraph_int_t tmp2 = VECTOR(indices)[i * 2 + 1];
                VECTOR(indices)[i * 2] = VECTOR(indices)[swap_idx * 2];
                VECTOR(indices)[i * 2 + 1] = VECTOR(indices)[swap_idx * 2 + 1];
                VECTOR(indices)[swap_idx * 2] = tmp1;
                VECTOR(indices)[swap_idx * 2 + 1] = tmp2;
            }

            /* SGD updates are stateful sequential transitions; kept un-parallelized to preserve stochasticity pathing */
            for (k = 0; k < num_pairs; k++) {
                IGRAPH_ALLOW_INTERRUPTION();

                igraph_int_t u = VECTOR(indices)[k * 2];
                igraph_int_t v = VECTOR(indices)[k * 2 + 1];

                igraph_real_t wc = (step > lr_cap) ? lr_cap : step;
                igraph_real_t target_d = MATRIX(d, u, v);

                igraph_real_t st_u = VECTOR(sin_theta)[u], ct_u = VECTOR(cos_theta)[u];
                igraph_real_t sp_u = VECTOR(sin_phi)[u],   cp_u = VECTOR(cos_phi)[u];
                igraph_real_t st_v = VECTOR(sin_theta)[v], ct_v = VECTOR(cos_theta)[v];
                igraph_real_t sp_v = VECTOR(sin_phi)[v],   cp_v = VECTOR(cos_phi)[v];

                igraph_real_t cos_dp = cp_u * cp_v + sp_u * sp_v;
                igraph_real_t val = ct_u * ct_v + st_u * st_v * cos_dp;
                if (val > 1.0) val = 1.0;
                if (val < -1.0) val = -1.0;

                igraph_real_t delta = acos(val);
                igraph_real_t factor = 2.0 * (delta - target_d);

                igraph_real_t denom = sqrt(1.0 - val * val);
                if (denom < 1e-8) continue;

                igraph_real_t sin_dp = sp_u * cp_v - cp_u * sp_v;
                igraph_real_t inv_denom = 1.0 / denom;

                igraph_real_t dd_ti = (st_u * ct_v - ct_u * st_v * cos_dp) * inv_denom;
                igraph_real_t dd_pi = st_u * st_v * sin_dp * inv_denom;
                igraph_real_t dd_tj = (ct_u * st_v - st_u * ct_v * cos_dp) * inv_denom;

                igraph_real_t g = wc * factor;

                MATRIX(angles, u, 0) -= g * dd_ti;
                MATRIX(angles, u, 1) -= g * dd_pi;
                MATRIX(angles, v, 0) -= g * dd_tj;
                MATRIX(angles, v, 1) += g * dd_pi;
            }

            igraph_i_smds_update_cartesian(res, &sin_theta, &cos_theta, &sin_phi, &cos_phi, no_of_nodes);
            IGRAPH_STEP(res, NULL);
        }

        /* Recompute trig from final angles after last SGD iteration */
        igraph_i_smds_compute_trig(&angles, &sin_theta, &cos_theta, &sin_phi, &cos_phi, no_of_nodes);
        igraph_i_smds_update_cartesian(res, &sin_theta, &cos_theta, &sin_phi, &cos_phi, no_of_nodes);

        IGRAPH_PROGRESS("Spherical MDS layout", 100, NULL);

        igraph_vector_destroy(&cos_phi); igraph_vector_destroy(&sin_phi);
        igraph_vector_destroy(&cos_theta); igraph_vector_destroy(&sin_theta);
        igraph_vector_destroy(&etas); igraph_vector_int_destroy(&indices);
        igraph_matrix_destroy(&angles); igraph_matrix_destroy(&d);
        IGRAPH_FINALLY_CLEAN(8);

        return IGRAPH_SUCCESS;

    } else {
        /* Big Data Extension Path */
        igraph_vector_int_t perm;
        igraph_matrix_t d_landmark;
        igraph_t landmark_graph;
        igraph_matrix_t d_sub;
        igraph_matrix_t res_sub;
        igraph_vector_t row_means;
        igraph_matrix_t Q;
        igraph_vector_t q_vector;
        igraph_matrix_t S;
        igraph_matrix_t S_inv;
        igraph_matrix_t x_1_s_inv;
        igraph_vector_int_t landmark_map;
        igraph_int_t i, j, k, v;

        IGRAPH_VECTOR_INT_INIT_FINALLY(&perm, no_of_nodes);
        for (i = 0; i < no_of_nodes; i++) VECTOR(perm)[i] = i;
        for (i = no_of_nodes - 1; i > 0; i--) {
            igraph_int_t swap_idx = RNG_INTEGER(0, i);
            igraph_int_t tmp = VECTOR(perm)[i];
            VECTOR(perm)[i] = VECTOR(perm)[swap_idx];
            VECTOR(perm)[swap_idx] = tmp;
        }

        IGRAPH_MATRIX_INIT_FINALLY(&d_landmark, l, no_of_nodes);

        if (dist == NULL) {
            igraph_vs_t from_vs;
            igraph_vector_int_t landmark_vids;
            IGRAPH_VECTOR_INT_INIT_FINALLY(&landmark_vids, l);
            for (i = 0; i < l; i++) VECTOR(landmark_vids)[i] = VECTOR(perm)[i];
            IGRAPH_CHECK(igraph_vs_vector(&from_vs, &landmark_vids));
            IGRAPH_FINALLY(igraph_vs_destroy, &from_vs);
            IGRAPH_CHECK(igraph_distances(graph, NULL, &d_landmark, from_vs, igraph_vss_all(), IGRAPH_ALL));
            igraph_vs_destroy(&from_vs);
            igraph_vector_int_destroy(&landmark_vids);
            IGRAPH_FINALLY_CLEAN(2);
        } else {
            #pragma omp parallel for private(j) default(none) shared(d_landmark, dist, perm, l, no_of_nodes)
            for (i = 0; i < l; i++) {
                igraph_int_t u = VECTOR(perm)[i];
                for (j = 0; j < no_of_nodes; j++) {
                    MATRIX(d_landmark, i, j) = MATRIX(*dist, u, j);
                }
            }
        }

        igraph_real_t max_dist = 0.0;
        #pragma omp parallel for reduction(max:max_dist) private(j) default(none) shared(d_landmark, l, no_of_nodes)
        for (i = 0; i < l; i++) {
            for (j = 0; j < no_of_nodes; j++) {
                if (MATRIX(d_landmark, i, j) > max_dist) max_dist = MATRIX(d_landmark, i, j);
            }
        }
        if (max_dist > 0.0) {
            igraph_real_t scale = M_PI / max_dist;
            igraph_matrix_scale(&d_landmark, scale);
        }

        IGRAPH_CHECK(igraph_empty(&landmark_graph, l, IGRAPH_UNDIRECTED));
        IGRAPH_FINALLY(igraph_destroy, &landmark_graph);

        IGRAPH_MATRIX_INIT_FINALLY(&d_sub, l, l);
        #pragma omp parallel for private(j) default(none) shared(d_sub, d_landmark, perm, l)
        for (i = 0; i < l; i++) {
            for (j = 0; j < l; j++) {
                igraph_int_t v_idx = VECTOR(perm)[j];
                MATRIX(d_sub, i, j) = MATRIX(d_landmark, i, v_idx);
            }
            MATRIX(d_sub, i, i) = 0.0;
        }

        IGRAPH_MATRIX_INIT_FINALLY(&res_sub, l, 3);
        IGRAPH_CHECK(igraph_layout_mds_spherical(&landmark_graph, &res_sub, &d_sub, num_iter, lr_cap));

        IGRAPH_VECTOR_INIT_FINALLY(&row_means, l);
        IGRAPH_MATRIX_INIT_FINALLY(&Q, l, l);
        IGRAPH_VECTOR_INIT_FINALLY(&q_vector, l);
        igraph_i_smds_gower_center(&d_sub, &row_means, &Q, &q_vector, l);

        IGRAPH_MATRIX_INIT_FINALLY(&S, 3, 3);
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                igraph_real_t sum = 0.0;
                for (k = 0; k < l; k++) {
                    IGRAPH_ALLOW_INTERRUPTION();
                    sum += MATRIX(res_sub, k, i) * MATRIX(res_sub, k, j);
                }
                MATRIX(S, i, j) = sum / l;
            }
        }

        IGRAPH_MATRIX_INIT_FINALLY(&S_inv, 3, 3);
        igraph_real_t det = MATRIX(S, 0, 0) * (MATRIX(S, 1, 1) * MATRIX(S, 2, 2) - MATRIX(S, 1, 2) * MATRIX(S, 2, 1))
                          - MATRIX(S, 0, 1) * (MATRIX(S, 1, 0) * MATRIX(S, 2, 2) - MATRIX(S, 1, 2) * MATRIX(S, 2, 0))
                          + MATRIX(S, 0, 2) * (MATRIX(S, 1, 0) * MATRIX(S, 2, 1) - MATRIX(S, 1, 1) * MATRIX(S, 2, 0));

        if (fabs(det) < 1e-9) {
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) MATRIX(S_inv, i, j) = (i == j) ? 1.0 : 0.0;
            }
        } else {
            igraph_real_t inv_det = 1.0 / det;
            MATRIX(S_inv, 0, 0) = (MATRIX(S, 1, 1) * MATRIX(S, 2, 2) - MATRIX(S, 1, 2) * MATRIX(S, 2, 1)) * inv_det;
            MATRIX(S_inv, 0, 1) = (MATRIX(S, 0, 2) * MATRIX(S, 2, 1) - MATRIX(S, 0, 1) * MATRIX(S, 2, 2)) * inv_det;
            MATRIX(S_inv, 0, 2) = (MATRIX(S, 0, 1) * MATRIX(S, 1, 2) - MATRIX(S, 0, 2) * MATRIX(S, 1, 1)) * inv_det;
            MATRIX(S_inv, 1, 0) = (MATRIX(S, 1, 2) * MATRIX(S, 2, 0) - MATRIX(S, 1, 0) * MATRIX(S, 2, 2)) * inv_det;
            MATRIX(S_inv, 1, 1) = (MATRIX(S, 0, 0) * MATRIX(S, 2, 2) - MATRIX(S, 0, 2) * MATRIX(S, 2, 0)) * inv_det;
            MATRIX(S_inv, 1, 2) = (MATRIX(S, 0, 2) * MATRIX(S, 1, 0) - MATRIX(S, 0, 0) * MATRIX(S, 1, 2)) * inv_det;
            MATRIX(S_inv, 2, 0) = (MATRIX(S, 1, 0) * MATRIX(S, 2, 1) - MATRIX(S, 1, 1) * MATRIX(S, 2, 0)) * inv_det;
            MATRIX(S_inv, 2, 1) = (MATRIX(S, 0, 1) * MATRIX(S, 2, 0) - MATRIX(S, 0, 0) * MATRIX(S, 2, 1)) * inv_det;
            MATRIX(S_inv, 2, 2) = (MATRIX(S, 0, 0) * MATRIX(S, 1, 1) - MATRIX(S, 0, 1) * MATRIX(S, 1, 0)) * inv_det;
        }

        IGRAPH_MATRIX_INIT_FINALLY(&x_1_s_inv, l, 3);
        #pragma omp parallel for private(j, k) default(none) shared(x_1_s_inv, res_sub, S_inv, l)
        for (i = 0; i < l; i++) {
            for (j = 0; j < 3; j++) {
                igraph_real_t sum = 0.0;
                for (k = 0; k < 3; k++) sum += MATRIX(res_sub, i, k) * MATRIX(S_inv, k, j);
                MATRIX(x_1_s_inv, i, j) = sum;
            }
        }

        IGRAPH_VECTOR_INT_INIT_FINALLY(&landmark_map, no_of_nodes);
        igraph_vector_int_fill(&landmark_map, -1);
        for (i = 0; i < l; i++) VECTOR(landmark_map)[VECTOR(perm)[i]] = i;

        IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 3));

        /* Highly parallel global reconstruction step */
        igraph_i_smds_gower_interpolate(res, &res_sub, &d_landmark, &x_1_s_inv, &q_vector, &landmark_map, no_of_nodes, l);

        igraph_vector_int_destroy(&landmark_map); igraph_matrix_destroy(&x_1_s_inv);
        igraph_matrix_destroy(&S_inv); igraph_matrix_destroy(&S);
        igraph_vector_destroy(&q_vector); igraph_matrix_destroy(&Q);
        igraph_vector_destroy(&row_means); igraph_matrix_destroy(&res_sub);
        igraph_matrix_destroy(&d_sub); igraph_destroy(&landmark_graph);
        igraph_matrix_destroy(&d_landmark); igraph_vector_int_destroy(&perm);
        IGRAPH_FINALLY_CLEAN(12);

        return IGRAPH_SUCCESS;
    }
}