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
 *   C = C_BC + C_A
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
 *   d C_BC / d L(u) = sum_v [ 1/p * dp/dL(u)  -  1/(1-p) * dp/dL(u) ]
 *
 * The first term attracts connected vertices; the second repels
 * non-connected vertices.
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
 *   p_T(u,v) = 1 / (1 + dist^2 / b)
 *
 * The gradient of C_BC w.r.t. L(u) for a pair (u,v) is (Eq. 10):
 *   For edges:       (1/p) * dp/dL(u)  =  p * dist * (diff / dist)
 *   For non-edges:  -1/(1-p) * dp/dL(u) = -(1-p)/dist * (diff / dist)
 *
 * The compact term gradient (Eq. 8) is:
 *   d C_compact / d L(u) = (2 / |V|^2) * L(u)
 *
 * The length term gradient (Eq. 9) is applied only to edges:
 *   d C_length / d L(u) = (2 / |E|) * (||L(u)-L(v)|| - 1) * (diff / dist)
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

        for (igraph_int_t iter = 0; iter < niter; iter++) {
            IGRAPH_ALLOW_INTERRUPTION();

            igraph_matrix_null(&gradients);

            for (igraph_int_t u = 0; u < vcount; u++) {
                igraph_real_t grad_u[3] = {0, 0, 0};

                /* Compute gradient for vertex u over all pairs (u, v).
                 * This is the O(|V|^2) loop from Section 3.2.3. */
                for (igraph_int_t v = 0; v < vcount; v++) {
                    igraph_real_t diff[3];
                    igraph_real_t dist_sq = 0;
                    igraph_real_t dist;
                    igraph_bool_t connected;
                    igraph_real_t p_val;
                    igraph_real_t force;

                    if (u == v) continue;

                    /* Compute displacement vector and squared distance */
                    for (igraph_int_t d = 0; d < dim; d++) {
                        diff[d] = MATRIX(*res, u, d) - MATRIX(*res, v, d);
                        dist_sq += diff[d] * diff[d];
                    }

                    dist = sqrt(dist_sq);

                    /* Prevent division by zero when vertices overlap */
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

                    /* Compute p(u,v) using the selected distribution
                     * (Section 3.2.2).
                     *
                     * Student's t-distribution (Algorithm BCGL-T):
                     *   p_T(u,v) = 1 / (1 + dist^2 / b)
                     * where b = IGRAPH_I_BCGL_T_DIST_B is the degrees
                     * of freedom parameter. The normalization Z is
                     * omitted because it cancels in the gradient.
                     */
                    p_val = 1.0 / (1.0 + dist_sq / IGRAPH_I_BCGL_T_DIST_B);

                    igraph_are_adjacent(graph, u, v, &connected);

                    /* Compute force from Eq. 10:
                     *
                     * For connected vertices (edges, Eq. 6 first term):
                     *   The attractive force is derived from
                     *   -(1/p) * dp/dL(u). For the t-distribution this
                     *   simplifies to p * dist * (diff/dist).
                     *   We also add the length regularization gradient
                     *   (Eq. 9): lambda_length * (dist - 1) / |E|.
                     *
                     * For non-connected vertices (non-edges, Eq. 6 second term):
                     *   The repulsive force is derived from
                     *   1/(1-p) * dp/dL(u). This simplifies to
                     *   -(1-p)/dist * (diff/dist).
                     */
                    if (connected) {
                        /* Attractive force from C_BC (Eq. 6) + C_length (Eq. 9) */
                        force = p_val * dist;
                        force += IGRAPH_I_BCGL_LAMBDA_LENGTH * (dist - 1.0) /
                                 (ecount > 0 ? ecount : 1);
                    } else {
                        /* Repulsive force from C_BC (Eq. 6) */
                        force = -(1.0 - p_val) / dist;
                    }

                    /* Accumulate gradient: lambda_bc * force * unit_direction
                     * (Eq. 10 applied to each coordinate) */
                    for (igraph_int_t d = 0; d < dim; d++) {
                        grad_u[d] += IGRAPH_I_BCGL_LAMBDA_BC * force *
                                     (diff[d] / dist);
                    }
                }

                /* Compact penalty gradient (Eq. 8):
                 *   d C_compact / d L(u) = (2 / |V|^2) * L(u)
                 * We absorb the factor of 2 into lambda_compact. */
                for (igraph_int_t d = 0; d < dim; d++) {
                    grad_u[d] += IGRAPH_I_BCGL_LAMBDA_COMPACT *
                                 MATRIX(*res, u, d) / vcount;
                }

                for (igraph_int_t d = 0; d < dim; d++) {
                    MATRIX(gradients, u, d) = grad_u[d];
                }
            }

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
 * t-distribution with one degree of freedom (Algorithm BCGL-T):
 * <code>p_T(u,v) = 1 / (1 + dist^2 / b)</code>,
 * where b is the degrees of freedom parameter. The fat-tailed nature of the
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
