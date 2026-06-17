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
static igraph_error_t igraph_i_layout_bcgl(
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

        for (igraph_int_t iter = 0; iter < niter; iter++) {
            IGRAPH_ALLOW_INTERRUPTION();

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
                            diff[d] = RNG_UNIF(0, 1e-4);
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
                            diff[d] = RNG_UNIF(0, 1e-4);
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
                            diff[d] = RNG_UNIF(0, 1e-4);
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
            for (igraph_int_t u = 0; u < vcount; u++) {
                for (igraph_int_t d = 0; d < dim; d++) {
                    MATRIX(velocity, u, d) = momentum * MATRIX(velocity, u, d) -
                                             learning_rate * MATRIX(gradients, u, d);
                    MATRIX(*res, u, d) += MATRIX(velocity, u, d);
                }
            }
        }

        igraph_matrix_destroy(&gradients);
        igraph_matrix_destroy(&velocity);
        IGRAPH_FINALLY_CLEAN(2);
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
                                  igraph_layout_bcgl_distribution_t distribution) {
    return igraph_i_layout_bcgl(graph, res, use_seed, 2, niter,
                                learning_rate, momentum, distribution);
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
                                     igraph_layout_bcgl_distribution_t distribution) {
    return igraph_i_layout_bcgl(graph, res, use_seed, 3, niter,
                                learning_rate, momentum, distribution);
}
