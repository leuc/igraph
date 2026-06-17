/*
   igraph library.
   Copyright (C) 2025  The igraph development team <igraph@igraph.org>

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

/*
 * Implementation of the BCGL (Binary Classification-Based Graph Layout)
 * algorithm from:
 *
 *   Yan, Kai, Tiejun Zhao, and Muyun Yang:
 *   BCGL: Binary Classification-Based Graph Layout.
 *   IEICE Transactions on Information and Systems E105.D, no. 9 (2022): 1610--1619.
 *   https://doi.org/10.1587/transinf.2021EDP7260
 *
 * The algorithm frames graph layout as a binary classification problem
 * (Section 3.1.2). A probability model p(u,v) assigns high probability
 * to vertex pairs that should be connected and low probability to those
 * that should not. The layout is obtained by minimizing a cross-entropy
 * loss (Eq. 3) augmented with aesthetic regularization terms (Eq. 5).
 *
 * The total objective (Eq. 5) is:
 *   C = lambda_BC * C_BC + C_A
 *
 * where C_BC (Eq. 6) is the binary classification loss:
 *   C_BC = sum_{(u,v) in E} -log p(u,v)
 *        + sum_{(u,v) not in E} -log(1 - p(u,v))
 *
 * and C_A (Eq. 7) is the aesthetic regularization:
 *   C_A = lambda_compact * C_compact + lambda_length * C_length
 *
 * C_compact (Eq. 8) pulls vertices toward the origin:
 *   C_compact = (1 / |V|^2) * sum_u ||L(u)||^2
 *
 * C_length (Eq. 9) encourages unit edge length:
 *   C_length = (1 / |E|) * sum_{(u,v) in E} ||L(u) - L(v) - 1||^2
 *
 * Optimization uses momentum-based SGD (Section 3.2.3). The gradient of
 * C_BC (Eq. 10) decomposes into attractive forces on edges and repulsive
 * forces on non-edges:
 *   d C_BC / d L(u) = -sum_edges [1/p * dp/dL(u)]
 *                    + sum_non-edges [1/(1-p) * dp/dL(u)]
 *
 * The code computes the gradient of C_BC directly using the exact
 * normalized probability p(u,v) = q(u,v) / Z where
 * q(u,v) = 1/(1+dist^2*b) and Z = sum_{i!=j} q(i,j). The gradient
 * dp/dL(u) uses the quotient rule (Z*dq - q*dZ) / Z^2, requiring a
 * two-pass strategy: first compute Z, then for each vertex u compute
 * dZ/dL(u) and evaluate all pair gradients.
 *
 * The paper also describes a multilevel strategy (Section 3.2.4, (2))
 * where the graph is coarsened into G_1, G_2, ..., G_K with decreasing
 * sizes, laid out from the coarsest, and refined via interpolation.
 * This is not implemented here; only the single-level SGD layout is
 * provided.
 *
 * Time complexity per this implementation: O(N * |V|^2) where N is the
 * number of iterations. The paper's full multilevel BCGL has complexity
 * O(K(|V| + |E|) + N|V|^2) where K is the coarsening depth (typically
 * less than 5).
 */

#include "igraph_layout.h"

#include "igraph_interface.h"
#include "igraph_random.h"
#include "igraph_structural.h"

#include "core/interruption.h"

#include "igraph_progress.h"
#include "igraph_step.h"
#include "igraph_barnes_hut.h"
#include "igraph_conversion.h"

#include <math.h>

/* Hyperparameters from the BCGL paper (Section 3.2.1, Eq. 5, 7).
 * lambda_bc weights the classification loss C_BC (Eq. 6).
 * lambda_compact weights the compact regularization C_compact (Eq. 8).
 * lambda_length weights the edge length regularization C_length (Eq. 9).
 * t_dist_b is the degrees of freedom b in the Student's t-distribution
 * (Section 3.2.2, Algorithm BCGL-T): p(u,v) = 1 / (1 + dist^2 / b).
 */
#define IGRAPH_I_BCGL_LAMBDA_BC 1.0
#define IGRAPH_I_BCGL_LAMBDA_COMPACT 0.01
#define IGRAPH_I_BCGL_LAMBDA_LENGTH 0.01
#define IGRAPH_I_BCGL_T_DIST_B 1.0

/* Enable per-iteration debug logging to stderr.
 * Set to 0 to disable (zero overhead). */
#define IGRAPH_DEBUG_BCGL 1

/**
 * \function igraph_i_layout_bcgl
 * \brief Core BCGL layout implementation (internal).
 *
 * Implements Section 3.2.3 (SGD optimization) of the paper. The gradient
 * is computed according to Eq. 10, decomposing into attractive forces
 * (connected pairs) and repulsive forces (non-connected pairs). The
 * probability p(u,v) is computed using the selected distribution
 * (Section 3.2.2).
 *
 * For the Student's t-distribution (Algorithm BCGL-T):
 *   p_T(u,v) = q(u,v) / Z,
 *   q(u,v) = 1 / (1 + dist^2 * b),
 *   Z = sum_{i != j} q(i,j)
 *
 * The gradient of C_BC w.r.t. L(u) for a pair (u,v) is (Eq. 10):
 *   For edges:      d/dL [-log p] = -(1/p) * dp/dL(u)
 *   For non-edges:  d/dL [-log(1-p)] = (1/(1-p)) * dp/dL(u)
 *
 * where dp/dL(u) uses the quotient rule:
 *   dp(u,v)/dL(u) = (Z * dq(u,v)/dL(u) - q(u,v) * dZ/dL(u)) / Z^2
 *
 * The global coupling term dZ/dL(u) is computed as:
 *   dZ/dL(u) = 2 * sum_{v != u} dq(u,v)/dL(u)
 *            = 2 * sum_{v != u} -2*b*q(u,v)^2 * (L(u)-L(v))
 *
 * The C_length gradient (Eq. 9) is applied separately to edges:
 *   d C_length / d L(u) = (2 / |E|) * (||L(u)-L(v)|| - 1) * (diff / dist)
 *
 * The C_compact gradient (Eq. 8) is:
 *   d C_compact / d L(u) = (2 / |V|^2) * L(u)
 */
static igraph_error_t igraph_i_layout_bcgl_exact(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t dim,
        igraph_int_t niter,
        igraph_real_t learning_rate,
        igraph_real_t momentum,
        igraph_layout_bcgl_distribution_t distribution) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (niter < 0) {
        IGRAPH_ERROR("Number of iterations must be non-negative in "
                     "BCGL layout.", IGRAPH_EINVAL);
    }

    if (dim != 2 && dim != 3) {
        IGRAPH_ERROR("Dimension must be 2 or 3 in BCGL layout.", IGRAPH_EINVAL);
    }

    if (learning_rate <= 0) {
        IGRAPH_ERROR("Learning rate must be positive in BCGL layout.",
                     IGRAPH_EINVAL);
    }

    if (momentum < 0 || momentum > 1) {
        IGRAPH_ERROR("Momentum must be between 0 and 1 in BCGL layout.",
                     IGRAPH_EINVAL);
    }

    if (distribution != IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T &&
        distribution != IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN) {
        IGRAPH_ERROR("Invalid BCGL distribution.", IGRAPH_EINVAL);
    }

    if (distribution == IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN) {
        IGRAPH_ERROR("Gaussian distribution is not yet implemented in "
                     "BCGL layout.", IGRAPH_UNIMPLEMENTED);
    }

    /* Initialize layout: random positions in [-1, 1] or use seed */
    if (!use_seed) {
        IGRAPH_CHECK(igraph_matrix_resize(res, vcount, dim));
        for (igraph_int_t i = 0; i < vcount; i++) {
            for (igraph_int_t d = 0; d < dim; d++) {
                MATRIX(*res, i, d) = RNG_UNIF(-1.0, 1.0);
            }
        }
    } else {
        if (igraph_matrix_nrow(res) != vcount ||
            igraph_matrix_ncol(res) != dim) {
            IGRAPH_ERROR("Invalid start position matrix size in "
                         "BCGL layout.", IGRAPH_EINVAL);
        }
    }

    {
        igraph_matrix_t velocity;
        igraph_matrix_t gradients;

        /* Momentum-based SGD (Section 3.2.3) */
        IGRAPH_MATRIX_INIT_FINALLY(&velocity, vcount, dim);
        igraph_matrix_null(&velocity);

        IGRAPH_MATRIX_INIT_FINALLY(&gradients, vcount, dim);

#if IGRAPH_DEBUG_BCGL
        fprintf(stderr,
                "%-6s %-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s\n",
                "iter", "C_BC", "C_compact", "C_length", "C_total",
                "||grad||", "||vel||", "mean_edge", "mean_nonedge");
#endif

        IGRAPH_PROGRESS("BCGL layout", 0, NULL);
        for (igraph_int_t iter = 0; iter < niter; iter++) {
            IGRAPH_ALLOW_INTERRUPTION();

            IGRAPH_PROGRESS("BCGL layout", 100.0 * iter / niter, NULL);
            IGRAPH_STEP(res, NULL);

#if IGRAPH_DEBUG_BCGL
            igraph_real_t _dbg_loss_bc = 0.0;
            igraph_real_t _dbg_loss_compact = 0.0;
            igraph_real_t _dbg_loss_length = 0.0;
            igraph_real_t _dbg_sum_edge_dist = 0.0;
            igraph_real_t _dbg_sum_nonedge_dist = 0.0;
            igraph_int_t _dbg_edge_count = 0;
            igraph_int_t _dbg_nonedge_count = 0;
#endif

            igraph_matrix_null(&gradients);

            /* ==========================================================
             * PASS 1: Compute global normalization constant Z
             *
             *   Z = sum_{i != j} q(i,j)
             *   q(i,j) = 1 / (1 + dist(i,j)^2 * b)
             *
             * Since q(i,j) = q(j,i), we sum over i < j and double.
             * ========================================================== */
            igraph_real_t Z = 0.0;
#ifdef _OPENMP
#  pragma omp parallel for schedule(guided) reduction(+: Z)
#endif
            for (igraph_int_t i = 0; i < vcount; i++) {
                for (igraph_int_t j = i + 1; j < vcount; j++) {
                    igraph_real_t diff[3];
                    igraph_real_t dist_sq = 0;

                    for (igraph_int_t d = 0; d < dim; d++) {
                        diff[d] = MATRIX(*res, i, d) - MATRIX(*res, j, d);
                        dist_sq += diff[d] * diff[d];
                    }

                    if (sqrt(dist_sq) < 1e-5) {
                        for (igraph_int_t d = 0; d < dim; d++) {
#ifdef _OPENMP
                            diff[d] = 1e-4 * (igraph_real_t) ((i + j + d) % 100) / 100.0;
#else
                            diff[d] = RNG_UNIF(0, 1e-4);
#endif
                        }
                        dist_sq = 0;
                        for (igraph_int_t d = 0; d < dim; d++) {
                            dist_sq += diff[d] * diff[d];
                        }
                    }

                    igraph_real_t q = 1.0 / (1.0 + dist_sq * IGRAPH_I_BCGL_T_DIST_B);
                    Z += 2.0 * q;  /* q_ij + q_ji */
                }
            }

            /* ==========================================================
             * PASS 2: Compute gradients with exact Z normalization
             *
             * Probability: p(u,v) = q(u,v) / Z
             *
             * Cross-entropy gradient via quotient rule:
             *   dp(u,v)/dL(u) = (Z * dq - q * dZ/dL(u)) / Z^2
             *   dZ/dL(u) = 2 * sum_{v != u} dq(u,v)/dL(u)
             *
             * Edge loss:   d/dL [-log p] = -(1/p) * dp/dL(u)
             * Non-edge:    d/dL [-log(1-p)] = (1/(1-p)) * dp/dL(u)
             * ========================================================== */
#ifdef _OPENMP
#  if IGRAPH_DEBUG_BCGL
#    pragma omp parallel for schedule(guided) \
        reduction(+: _dbg_loss_bc, _dbg_loss_compact, _dbg_loss_length, \
                  _dbg_sum_edge_dist, _dbg_sum_nonedge_dist, \
                  _dbg_edge_count, _dbg_nonedge_count)
#  else
#    pragma omp parallel for schedule(guided)
#  endif
#endif
            for (igraph_int_t u = 0; u < vcount; u++) {
                igraph_real_t grad_u[3] = {0, 0, 0};

                /* Step 2a: Precompute dZ/dL(u) */
                igraph_real_t dZ_u[3] = {0, 0, 0};
                for (igraph_int_t v = 0; v < vcount; v++) {
                    igraph_real_t diff[3];
                    igraph_real_t dist_sq = 0;

                    if (u == v) continue;

                    for (igraph_int_t d = 0; d < dim; d++) {
                        diff[d] = MATRIX(*res, u, d) - MATRIX(*res, v, d);
                        dist_sq += diff[d] * diff[d];
                    }

                    if (sqrt(dist_sq) < 1e-5) {
                        for (igraph_int_t d = 0; d < dim; d++) {
#ifdef _OPENMP
                            diff[d] = 1e-4 * (igraph_real_t) ((u + v + d) % 100) / 100.0;
#else
                            diff[d] = RNG_UNIF(0, 1e-4);
#endif
                        }
                        dist_sq = 0;
                        for (igraph_int_t d = 0; d < dim; d++) {
                            dist_sq += diff[d] * diff[d];
                        }
                    }

                    igraph_real_t q = 1.0 / (1.0 + dist_sq * IGRAPH_I_BCGL_T_DIST_B);
                    igraph_real_t dq_coeff = -2.0 * IGRAPH_I_BCGL_T_DIST_B * q * q;

                    /* dZ/dL(u) = sum_{v != u} (dq_uv/dL(u) + dq_vu/dL(u))
                     *           = 2 * sum_{v != u} dq_coeff * diff */
                    for (igraph_int_t d = 0; d < dim; d++) {
                        dZ_u[d] += 2.0 * dq_coeff * diff[d];
                    }
                }

                /* Step 2b: Compute pair forces using exact Z */
                for (igraph_int_t v = 0; v < vcount; v++) {
                    igraph_real_t diff[3];
                    igraph_real_t dist_sq = 0;
                    igraph_real_t dist;
                    igraph_bool_t connected;

                    if (u == v) continue;

                    for (igraph_int_t d = 0; d < dim; d++) {
                        diff[d] = MATRIX(*res, u, d) - MATRIX(*res, v, d);
                        dist_sq += diff[d] * diff[d];
                    }

                    dist = sqrt(dist_sq);

                    if (dist < 1e-5) {
                        for (igraph_int_t d = 0; d < dim; d++) {
#ifdef _OPENMP
                            diff[d] = 1e-4 * (igraph_real_t) ((u + v + d) % 100) / 100.0;
#else
                            diff[d] = RNG_UNIF(0, 1e-4);
#endif
                        }
                        dist_sq = 0;
                        for (igraph_int_t d = 0; d < dim; d++) {
                            dist_sq += diff[d] * diff[d];
                        }
                        dist = sqrt(dist_sq);
                    }

                    igraph_real_t q = 1.0 / (1.0 + dist_sq * IGRAPH_I_BCGL_T_DIST_B);
                    igraph_real_t p = q / Z;

                    if (p < 1e-12) p = 1e-12;
                    if (p > 1.0 - 1e-12) p = 1.0 - 1e-12;

                    igraph_real_t dq_coeff = -2.0 * IGRAPH_I_BCGL_T_DIST_B * q * q;

                    /* Quotient rule: grad_p = (Z*dq - q*dZ_u) / Z^2 */
                    igraph_real_t inv_Z = 1.0 / Z;
                    igraph_real_t q_over_Z2 = q * inv_Z * inv_Z;

                    igraph_real_t grad_p[3];
                    for (igraph_int_t d = 0; d < dim; d++) {
                        grad_p[d] = dq_coeff * diff[d] * inv_Z - q_over_Z2 * dZ_u[d];
                    }

                    igraph_are_adjacent(graph, u, v, &connected);

                    if (connected) {
                        igraph_real_t loss_coeff = -1.0 / p;
                        for (igraph_int_t d = 0; d < dim; d++) {
                            grad_u[d] += IGRAPH_I_BCGL_LAMBDA_BC * loss_coeff * grad_p[d];
                        }

                        igraph_real_t length_grad_scalar = 2.0 * IGRAPH_I_BCGL_LAMBDA_LENGTH *
                                                           (dist - 1.0) /
                                                           (ecount > 0 ? ecount : 1);
                        for (igraph_int_t d = 0; d < dim; d++) {
                            grad_u[d] += length_grad_scalar * diff[d] / dist;
                        }

#if IGRAPH_DEBUG_BCGL
                        _dbg_loss_bc += -log(p);
                        _dbg_loss_length += (dist - 1.0) * (dist - 1.0);
                        _dbg_sum_edge_dist += dist;
                        _dbg_edge_count++;
#endif
                    } else {
                        igraph_real_t loss_coeff = 1.0 / (1.0 - p);
                        for (igraph_int_t d = 0; d < dim; d++) {
                            grad_u[d] += IGRAPH_I_BCGL_LAMBDA_BC * loss_coeff * grad_p[d];
                        }

#if IGRAPH_DEBUG_BCGL
                        _dbg_loss_bc += -log(1.0 - p);
                        _dbg_sum_nonedge_dist += dist;
                        _dbg_nonedge_count++;
#endif
                    }
                }

                /* Compact penalty gradient (Eq. 8):
                 *   d C_compact / d L(u) = (2 / |V|^2) * L(u) */
                for (igraph_int_t d = 0; d < dim; d++) {
                    grad_u[d] += 2.0 * IGRAPH_I_BCGL_LAMBDA_COMPACT *
                                 MATRIX(*res, u, d) / (vcount * vcount);
                }

#if IGRAPH_DEBUG_BCGL
                for (igraph_int_t d = 0; d < dim; d++) {
                    _dbg_loss_compact += MATRIX(*res, u, d) * MATRIX(*res, u, d);
                }
#endif

                for (igraph_int_t d = 0; d < dim; d++) {
                    MATRIX(gradients, u, d) = grad_u[d];
                }
            }

#if IGRAPH_DEBUG_BCGL
            {
                igraph_real_t _dbg_grad_norm = 0.0;
                igraph_real_t _dbg_vel_norm = 0.0;
                igraph_real_t _dbg_total_c;
                igraph_real_t _dbg_mean_edge = 0.0;
                igraph_real_t _dbg_mean_nonedge = 0.0;
                igraph_int_t _i, _d;

                for (_i = 0; _i < vcount; _i++) {
                    for (_d = 0; _d < dim; _d++) {
                        igraph_real_t _g = MATRIX(gradients, _i, _d);
                        _dbg_grad_norm += _g * _g;
                        igraph_real_t _v = MATRIX(velocity, _i, _d);
                        _dbg_vel_norm += _v * _v;
                    }
                }
                _dbg_grad_norm = sqrt(_dbg_grad_norm);
                _dbg_vel_norm = sqrt(_dbg_vel_norm);

                _dbg_loss_length /= (ecount > 0 ? ecount : 1);
                if (vcount > 0)
                    _dbg_loss_compact /= (vcount * vcount);
                _dbg_total_c = IGRAPH_I_BCGL_LAMBDA_BC * _dbg_loss_bc +
                               IGRAPH_I_BCGL_LAMBDA_COMPACT * _dbg_loss_compact +
                               IGRAPH_I_BCGL_LAMBDA_LENGTH * _dbg_loss_length;

                if (_dbg_edge_count > 0)
                    _dbg_mean_edge = _dbg_sum_edge_dist / _dbg_edge_count;
                if (_dbg_nonedge_count > 0)
                    _dbg_mean_nonedge = _dbg_sum_nonedge_dist / _dbg_nonedge_count;

                fprintf(stderr,
                        "%-6" IGRAPH_PRId " %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e %-14.4f %-14.4f\n",
                        iter, _dbg_loss_bc, _dbg_loss_compact, _dbg_loss_length,
                        _dbg_total_c, _dbg_grad_norm, _dbg_vel_norm,
                        _dbg_mean_edge, _dbg_mean_nonedge);
            }
#endif

            /* Apply momentum-based SGD update (Section 3.2.3):
             *   velocity(t) = momentum * velocity(t-1) - lr * gradient(t)
             *   L(t) = L(t-1) + velocity(t)
             */
#ifdef _OPENMP
#  pragma omp parallel for schedule(static) if(vcount > 1000)
#endif
            for (igraph_int_t u = 0; u < vcount; u++) {
                for (igraph_int_t d = 0; d < dim; d++) {
                    MATRIX(velocity, u, d) = momentum * MATRIX(velocity, u, d) -
                                             learning_rate * MATRIX(gradients, u, d);
                    MATRIX(*res, u, d) += MATRIX(velocity, u, d);
                }
            }
        }

        IGRAPH_PROGRESS("BCGL layout", 100, NULL);

        igraph_matrix_destroy(&gradients);
        igraph_matrix_destroy(&velocity);
        IGRAPH_FINALLY_CLEAN(2);
    }

    return IGRAPH_SUCCESS;
}

/* =========================================================================
 * User data for BH non-edge force kernel
 * ========================================================================= */
typedef struct {
    igraph_real_t Z;
    igraph_real_t inv_Z;
    igraph_real_t b;
    igraph_real_t lambda_bc;
    const igraph_real_t *dZ_x;
    const igraph_real_t *dZ_y;
    const igraph_real_t *dZ_z;
} igraph_i_bcgl_bh_data_t;

/* BH kernel: compute Z contribution (accumulates into force[0]) */
static void bcgl_z_kernel(
    const igraph_bh_point_t *p1, const igraph_bh_point_t *p2,
    igraph_real_t dx, igraph_real_t dy, igraph_real_t dz,
    igraph_real_t dist_sq, igraph_real_t force[3], void *user_data)
{
    igraph_real_t b = *(const igraph_real_t*)user_data;
    igraph_real_t q = 1.0 / (1.0 + dist_sq * b);
    igraph_real_t scale = (p2->id < 0) ? p2->mass : 1.0;
    force[0] += q * scale;
}

/* BH kernel: compute dZ_u contribution */
static void bcgl_dZ_kernel(
    const igraph_bh_point_t *p1, const igraph_bh_point_t *p2,
    igraph_real_t dx, igraph_real_t dy, igraph_real_t dz,
    igraph_real_t dist_sq, igraph_real_t force[3], void *user_data)
{
    igraph_real_t b = *(const igraph_real_t*)user_data;
    igraph_real_t q = 1.0 / (1.0 + dist_sq * b);
    igraph_real_t dq_coeff = -2.0 * b * q * q;
    igraph_real_t scale = (p2->id < 0) ? p2->mass : 1.0;
    force[0] += 2.0 * dq_coeff * dx * scale;
    force[1] += 2.0 * dq_coeff * dy * scale;
    if (p1->id >= 0) force[2] += 2.0 * dq_coeff * dz * scale;
}

/* BH kernel: compute non-edge force contribution for all pairs */
static void bcgl_nonedge_kernel(
    const igraph_bh_point_t *p1, const igraph_bh_point_t *p2,
    igraph_real_t dx, igraph_real_t dy, igraph_real_t dz,
    igraph_real_t dist_sq, igraph_real_t force[3], void *user_data)
{
    igraph_i_bcgl_bh_data_t *ud = (igraph_i_bcgl_bh_data_t*)user_data;
    igraph_int_t u = p1->id;
    igraph_real_t q = 1.0 / (1.0 + dist_sq * ud->b);
    igraph_real_t p = q * ud->inv_Z;
    igraph_real_t inv_one_minus_p = 1.0 / (1.0 - p);
    igraph_real_t dq_coeff = -2.0 * ud->b * q * q;
    igraph_real_t dq_over_Z = dq_coeff * ud->inv_Z;
    igraph_real_t q_over_Z2 = q * ud->inv_Z * ud->inv_Z;
    igraph_real_t scale = (p2->id < 0) ? p2->mass : 1.0;

    force[0] = ud->lambda_bc * inv_one_minus_p *
               (dq_over_Z * dx - q_over_Z2 * ud->dZ_x[u]) * scale;
    force[1] = ud->lambda_bc * inv_one_minus_p *
               (dq_over_Z * dy - q_over_Z2 * ud->dZ_y[u]) * scale;
    if (p1->id >= 0) {
        force[2] = ud->lambda_bc * inv_one_minus_p *
                   (dq_over_Z * dz - q_over_Z2 * ud->dZ_z[u]) * scale;
    }
}

/* =========================================================================
 * BH-accelerated BCGL gradient computation
 * ========================================================================= */
static igraph_error_t igraph_i_layout_bcgl_bh(
    const igraph_t *graph,
    igraph_matrix_t *res,
    igraph_int_t dim,
    igraph_int_t niter,
    igraph_real_t learning_rate,
    igraph_real_t momentum)
{
    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    /* Tiny graphs: no pairs to process, nothing to do */
    if (vcount < 2) {
        return IGRAPH_SUCCESS;
    }

    /* Build edge list for correction pass (flat alternating from/to) */
    igraph_vector_int_t edgelist;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&edgelist, 0);
    if (ecount > 0) {
        IGRAPH_CHECK(igraph_get_edgelist(graph, &edgelist, /*bycol=*/ 0));
    }

    /* BH tree setup */
    igraph_bh_tree_t tree;
    igraph_integer_t max_level, leaf_capacity;
    igraph_bh_tree_get_scaling_params(vcount, dim, &max_level, &leaf_capacity);
    igraph_bh_tree_init(&tree, dim, 0.6, max_level, leaf_capacity);

    igraph_matrix_t velocity;
    igraph_matrix_t gradients;
    igraph_matrix_t z_forces, dZ_forces;

    IGRAPH_MATRIX_INIT_FINALLY(&velocity, vcount, dim);
    igraph_matrix_null(&velocity);

    IGRAPH_MATRIX_INIT_FINALLY(&gradients, vcount, dim);

    IGRAPH_MATRIX_INIT_FINALLY(&z_forces, vcount, dim);
    IGRAPH_MATRIX_INIT_FINALLY(&dZ_forces, vcount, dim);

    /* Per-vertex dZ storage (always allocate dZ_z for kernel safety) */
    igraph_vector_t dZ_x, dZ_y, dZ_z;
    IGRAPH_VECTOR_INIT_FINALLY(&dZ_x, vcount);
    IGRAPH_VECTOR_INIT_FINALLY(&dZ_y, vcount);
    IGRAPH_VECTOR_INIT_FINALLY(&dZ_z, vcount);

    igraph_i_bcgl_bh_data_t ud;
    ud.b = IGRAPH_I_BCGL_T_DIST_B;
    ud.lambda_bc = IGRAPH_I_BCGL_LAMBDA_BC;

    IGRAPH_PROGRESS("BCGL layout", 0, NULL);
    for (igraph_int_t iter = 0; iter < niter; iter++) {
        IGRAPH_ALLOW_INTERRUPTION();
        IGRAPH_PROGRESS("BCGL layout", 100.0 * iter / niter, NULL);
        IGRAPH_STEP(res, NULL);

        igraph_matrix_null(&gradients);
        igraph_matrix_null(&z_forces);
        igraph_matrix_null(&dZ_forces);

        /* Rebuild BH tree from current positions */
        igraph_bh_tree_build(&tree, res, NULL);

        /* 1. Compute normalization constant Z via BH */
        igraph_bh_apply_repulsion_from_tree(&tree, &z_forces, bcgl_z_kernel, &ud.b);
        {
            igraph_real_t Z_sum = 0.0;
            for (igraph_int_t i = 0; i < vcount; i++) {
                Z_sum += MATRIX(z_forces, i, 0);
            }
            ud.Z = Z_sum / 2.0;
            ud.inv_Z = 1.0 / ud.Z;
        }

        /* 2. Compute dZ_u for all u via BH */
        igraph_bh_apply_repulsion_from_tree(&tree, &dZ_forces, bcgl_dZ_kernel, &ud.b);
        ud.dZ_x = VECTOR(dZ_x);
        ud.dZ_y = VECTOR(dZ_y);
        ud.dZ_z = VECTOR(dZ_z);
        for (igraph_int_t i = 0; i < vcount; i++) {
            VECTOR(dZ_x)[i] = MATRIX(dZ_forces, i, 0);
            VECTOR(dZ_y)[i] = MATRIX(dZ_forces, i, 1);
            VECTOR(dZ_z)[i] = (dim == 3) ? MATRIX(dZ_forces, i, 2) : 0.0;
        }

        /* 3. Compute non-edge (all-pairs repulsive) forces via BH */
        igraph_bh_apply_repulsion_from_tree(&tree, &gradients, bcgl_nonedge_kernel, &ud);

        /* 4. Edge correction: subtract BH non-edge, add correct edge force + length penalty */
        for (igraph_int_t e = 0; e < ecount; e++) {
            igraph_int_t u = VECTOR(edgelist)[2 * e];
            igraph_int_t v = VECTOR(edgelist)[2 * e + 1];
            igraph_real_t diff[3];
            igraph_real_t dist_sq = 0;

            for (igraph_int_t d = 0; d < dim; d++) {
                diff[d] = MATRIX(*res, u, d) - MATRIX(*res, v, d);
                dist_sq += diff[d] * diff[d];
            }
            igraph_real_t dist = sqrt(dist_sq);
            if (dist < 1e-12) dist = 1e-12;

            igraph_real_t q = 1.0 / (1.0 + dist_sq * ud.b);
            igraph_real_t p = q * ud.inv_Z;
            igraph_real_t dq_coeff = -2.0 * ud.b * q * q;
            igraph_real_t dq_over_Z = dq_coeff * ud.inv_Z;
            igraph_real_t q_over_Z2 = q * ud.inv_Z * ud.inv_Z;

            /* correction coefficient: (-1/p) - (1/(1-p)) = (2p-1) / (p*(1-p)) */
            igraph_real_t corr_coeff = (2.0 * p - 1.0) / (p * (1.0 - p));

            /* length penalty scalar */
            igraph_real_t len_scalar = 2.0 * IGRAPH_I_BCGL_LAMBDA_LENGTH *
                                       (dist - 1.0) / (ecount > 0 ? ecount : 1);

            const igraph_real_t dZ_u[] = { VECTOR(dZ_x)[u], VECTOR(dZ_y)[u], VECTOR(dZ_z)[u] };
            const igraph_real_t dZ_v[] = { VECTOR(dZ_x)[v], VECTOR(dZ_y)[v], VECTOR(dZ_z)[v] };

            for (igraph_int_t d = 0; d < dim; d++) {
                igraph_real_t grad_p_u = dq_over_Z * diff[d] - q_over_Z2 * dZ_u[d];
                igraph_real_t grad_p_v = -dq_over_Z * diff[d] - q_over_Z2 * dZ_v[d];
                igraph_real_t dir = diff[d] / dist;

                MATRIX(gradients, u, d) += ud.lambda_bc * corr_coeff * grad_p_u + len_scalar * dir;
                MATRIX(gradients, v, d) += ud.lambda_bc * corr_coeff * grad_p_v - len_scalar * dir;
            }
        }

        /* 5. Compact penalty (Eq. 8) */
        for (igraph_int_t u = 0; u < vcount; u++) {
            for (igraph_int_t d = 0; d < dim; d++) {
                MATRIX(gradients, u, d) += 2.0 * IGRAPH_I_BCGL_LAMBDA_COMPACT *
                                            MATRIX(*res, u, d) / (vcount * vcount);
            }
        }

        /* 6. Momentum-based SGD update */
        for (igraph_int_t u = 0; u < vcount; u++) {
            for (igraph_int_t d = 0; d < dim; d++) {
                MATRIX(velocity, u, d) = momentum * MATRIX(velocity, u, d) -
                                         learning_rate * MATRIX(gradients, u, d);
                MATRIX(*res, u, d) += MATRIX(velocity, u, d);
            }
        }
    }
    IGRAPH_PROGRESS("BCGL layout", 100, NULL);

    /* Cleanup in reverse order of FINALLY pushes */
    igraph_vector_destroy(&dZ_z);
    igraph_vector_destroy(&dZ_y);
    igraph_vector_destroy(&dZ_x);
    igraph_matrix_destroy(&dZ_forces);
    igraph_matrix_destroy(&z_forces);
    igraph_matrix_destroy(&gradients);
    igraph_matrix_destroy(&velocity);
    igraph_bh_tree_destroy(&tree);
    igraph_vector_int_destroy(&edgelist);
    IGRAPH_FINALLY_CLEAN(8);

    return IGRAPH_SUCCESS;
}

/* =========================================================================
 * Dispatcher: validate, seed, then dispatch to exact or BH
 * ========================================================================= */
static igraph_error_t igraph_i_layout_bcgl(
    const igraph_t *graph,
    igraph_matrix_t *res,
    igraph_bool_t use_seed,
    igraph_int_t dim,
    igraph_int_t niter,
    igraph_real_t learning_rate,
    igraph_real_t momentum,
    igraph_layout_bcgl_distribution_t distribution,
    igraph_bool_t use_bh)
{
    const igraph_int_t vcount = igraph_vcount(graph);

    if (niter < 0) {
        IGRAPH_ERROR("Number of iterations must be non-negative in "
                     "BCGL layout.", IGRAPH_EINVAL);
    }

    if (dim != 2 && dim != 3) {
        IGRAPH_ERROR("Dimension must be 2 or 3 in BCGL layout.", IGRAPH_EINVAL);
    }

    if (learning_rate <= 0) {
        IGRAPH_ERROR("Learning rate must be positive in BCGL layout.",
                     IGRAPH_EINVAL);
    }

    if (momentum < 0 || momentum > 1) {
        IGRAPH_ERROR("Momentum must be between 0 and 1 in BCGL layout.",
                     IGRAPH_EINVAL);
    }

    if (distribution != IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T &&
        distribution != IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN) {
        IGRAPH_ERROR("Invalid BCGL distribution.", IGRAPH_EINVAL);
    }

    if (distribution == IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN) {
        IGRAPH_ERROR("Gaussian distribution is not yet implemented in "
                     "BCGL layout.", IGRAPH_UNIMPLEMENTED);
    }

    /* Initialize layout: random positions in [-1, 1] or use seed */
    if (!use_seed) {
        IGRAPH_CHECK(igraph_matrix_resize(res, vcount, dim));
        for (igraph_int_t i = 0; i < vcount; i++) {
            for (igraph_int_t d = 0; d < dim; d++) {
                MATRIX(*res, i, d) = RNG_UNIF(-1.0, 1.0);
            }
        }
    } else {
        if (igraph_matrix_nrow(res) != vcount ||
            igraph_matrix_ncol(res) != dim) {
            IGRAPH_ERROR("Invalid start position matrix size in "
                         "BCGL layout.", IGRAPH_EINVAL);
        }
    }

    if (use_bh) {
        IGRAPH_CHECK(igraph_i_layout_bcgl_bh(graph, res, dim, niter,
                                              learning_rate, momentum));
    } else {
        IGRAPH_CHECK(igraph_i_layout_bcgl_exact(graph, res, use_seed, dim, niter,
                                                 learning_rate, momentum,
                                                 distribution));
    }

    return IGRAPH_SUCCESS;
}

/**
 * \ingroup layout
 * \function igraph_layout_bcgl
 * \brief Places the vertices on a plane according to the BCGL algorithm.
 *
 * </para><para>
 * This is a force-directed layout based on the Binary Classification-Based
 * Graph Layout (BCGL) method (Section 3.1.2 of the paper). The algorithm
 * frames graph layout as a binary classification problem where a probability
 * model p(u,v) assigns high probability to edge pairs and low probability
 * to non-edge pairs.
 *
 * </para><para>
 * The layout minimizes the total objective (Eq. 5):
 * <code>C = C_BC + C_A</code>,
 * where C_BC is the binary classification loss (Eq. 6) and C_A is the
 * aesthetic regularization (Eq. 7). The classification loss is:
 * <code>C_BC = sum_{edges} -log p(u,v) + sum_{non-edges} -log(1 - p(u,v))</code>.
 * The aesthetic terms include a compact penalty (Eq. 8) pulling vertices
 * toward the origin, and a length penalty (Eq. 9) encouraging unit edge
 * lengths.
 *
 * </para><para>
 * Optimization uses momentum-based stochastic gradient descent (Section 3.2.3),
 * with the gradient decomposing into attractive forces on edges and repulsive
 * forces on non-edges (Eq. 10). Each iteration has complexity O(|V|^2).
 *
 * </para><para>
 * The \p distribution parameter selects the probability model (Section 3.2.2):
 *
 * </para><para>
 * \c IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T uses a Student's
 * t-distribution with degrees of freedom b (Algorithm BCGL-T):
 * <code>p_T(u,v) = 1 / (Z * (1 + dist^2 * b))</code>,
 * where Z is the normalization constant. The fat-tailed nature of the
 * t-distribution helps disperse vertices that are at small distances,
 * producing more flexible layouts.
 *
 * </para><para>
 * \c IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN uses a Gaussian
 * distribution (Algorithm BCGL-G):
 * <code>p_G(u,v) = exp(-dist^2 / 2) / Z</code>.
 * This is not yet implemented. The paper notes that Gaussian distributions
 * can cause a "crowding problem", but the length term (Eq. 9) mitigates this.
 *
 * </para><para>
 * Reference:
 *
 * </para><para>
 * Yan, Kai, Tiejun Zhao, and Muyun Yang:
 * BCGL: Binary Classification-Based Graph Layout.
 * IEICE Transactions on Information and Systems E105.D, no. 9 (2022): 1610--1619.
 * https://doi.org/10.1587/transinf.2021EDP7260
 *
 * \param graph Pointer to an initialized graph object.
 * \param res Pointer to an initialized matrix object. This will
 *        contain the result and will be resized as needed.
 * \param use_seed If true the supplied values in the
 *        \p res argument are used as an initial layout, if
 *        false a random initial layout is used.
 * \param niter The number of SGD iterations to perform. A reasonable
 *        default value is 500.
 * \param learning_rate Step size for the gradient descent. Must be
 *        positive. A reasonable default value is 0.01.
 * \param momentum Momentum factor for SGD, between 0 and 1.
 *        A reasonable default value is 0.9.
 * \param distribution The probability distribution to use for the
 *        edge model (Section 3.2.2). Possible values:
 *        \c IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T and
 *        \c IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN (not yet
 *        implemented).
 * \return Error code.
 *
 * Time complexity: O(N * |V|^2) where N is the number of iterations
 * and |V| is the number of vertices in the graph. The paper's full
 * multilevel BCGL (Section 3.2.4, (2)) has complexity
 * O(K(|V| + |E|) + N|V|^2) with coarsening depth K (typically < 5),
 * but the multilevel strategy is not implemented here.
 */

igraph_error_t igraph_layout_bcgl(const igraph_t *graph,
                                  igraph_matrix_t *res,
                                  igraph_bool_t use_seed,
                                  igraph_int_t niter,
                                  igraph_real_t learning_rate,
                                  igraph_real_t momentum,
                                  igraph_layout_bcgl_distribution_t distribution,
                                  igraph_bool_t use_bh) {
    return igraph_i_layout_bcgl(graph, res, use_seed, 2, niter,
                                learning_rate, momentum, distribution, use_bh);
}

/**
 * \function igraph_layout_bcgl_3d
 * \brief 3D BCGL layout algorithm.
 *
 * This is the 3D version of the BCGL force-directed layout.
 * See \ref igraph_layout_bcgl() for the 2D version and for a
 * detailed description of the algorithm.
 *
 * \param graph Pointer to an initialized graph object.
 * \param res Pointer to an initialized matrix object. This will
 *        contain the result and will be resized as needed.
 * \param use_seed If true the supplied values in the
 *        \p res argument are used as an initial layout, if
 *        false a random initial layout is used.
 * \param niter The number of SGD iterations to perform. A reasonable
 *        default value is 500.
 * \param learning_rate Step size for the gradient descent. Must be
 *        positive. A reasonable default value is 0.01.
 * \param momentum Momentum factor for SGD, between 0 and 1.
 *        A reasonable default value is 0.9.
 * \param distribution The probability distribution to use for the
 *        edge model. See \ref igraph_layout_bcgl() for possible values.
 * \return Error code.
 *
 * Time complexity: O(N * |V|^2) where N is the number of iterations
 * and |V| is the number of vertices in the graph. The multilevel
 * strategy from the paper is not implemented; see
 * \ref igraph_layout_bcgl() for details.
 */

igraph_error_t igraph_layout_bcgl_3d(const igraph_t *graph,
                                     igraph_matrix_t *res,
                                     igraph_bool_t use_seed,
                                     igraph_int_t niter,
                                     igraph_real_t learning_rate,
                                     igraph_real_t momentum,
                                     igraph_layout_bcgl_distribution_t distribution,
                                     igraph_bool_t use_bh) {
    return igraph_i_layout_bcgl(graph, res, use_seed, 3, niter,
                                learning_rate, momentum, distribution, use_bh);
}
