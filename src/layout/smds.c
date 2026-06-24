/*
   Spherical MDS layout for igraph.
   Based on stochastic gradient descent optimization.
*/

#include "igraph_layout.h"

#include "igraph_components.h"
#include "igraph_interface.h"
#include "igraph_paths.h"
#include "igraph_random.h"

#include "core/math.h" /* M_PI */

#include <math.h>

/* Helper function to compute the spherical geodesic distance between two points */
static igraph_real_t igraph_i_smds_geodesic(igraph_real_t y, igraph_real_t x,
                                            igraph_real_t b, igraph_real_t a) {
    igraph_real_t val = sin(y)*sin(b) + cos(y)*cos(b)*cos(a-x);
    /* Clamp to [-1, 1] to avoid NaNs in acos due to floating point inaccuracies */
    if (val > 1.0) val = 1.0;
    if (val < -1.0) val = -1.0;
    return acos(val);
}

/* Helper function to compute the gradient of the spherical distance */
static void igraph_i_smds_gradient(igraph_real_t y, igraph_real_t x,
                                   igraph_real_t b, igraph_real_t a,
                                   igraph_matrix_t *grad) {
    igraph_real_t val = sin(b)*sin(y) + cos(b)*cos(y)*cos(a-x);
    igraph_real_t denom = sqrt(1.0 - val*val);

    /* Guard against division by zero if points are identical or antipodal */
    if (denom < 1e-8) {
        igraph_matrix_null(grad);
        return;
    }

    MATRIX(*grad, 0, 1) = -(sin(a-x)*cos(b)*cos(y)) / denom;
    MATRIX(*grad, 0, 0) = (-sin(b)*cos(y) + sin(y)*cos(b)*cos(a-x)) / denom;

    MATRIX(*grad, 1, 1) = -MATRIX(*grad, 0, 1);
    MATRIX(*grad, 1, 0) = (sin(b)*cos(y)*cos(a-x) - sin(y)*cos(b)) / denom;
}

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
 * \param graph A graph object.
 * \param res Pointer to an initialized matrix object. This will contain the
 * result (angles theta and phi). It will be resized if needed.
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
    igraph_vector_t etas;
    igraph_vector_int_t indices;
    igraph_matrix_t grad;
    igraph_real_t max_dist = 0.0;
    igraph_int_t i, j;

    if (no_of_nodes <= 1) {
        IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 2));
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

    /* 2. Initialize positions: theta in [0, pi], phi in [0, 2*pi] */
    IGRAPH_CHECK(igraph_matrix_resize(res, no_of_nodes, 2));
    for (i = 0; i < no_of_nodes; i++) {
        MATRIX(*res, i, 0) = RNG_UNIF(0.0, M_PI);
        MATRIX(*res, i, 1) = RNG_UNIF(0.0, 2.0 * M_PI);
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

    IGRAPH_MATRIX_INIT_FINALLY(&grad, 2, 2);

    /* 5. SGD Optimization Loop */
    for (igraph_int_t step_idx = 0; step_idx < num_iter; step_idx++) {
        igraph_real_t step = VECTOR(etas)[step_idx];

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
            igraph_int_t u = VECTOR(indices)[k * 2];
            igraph_int_t v = VECTOR(indices)[k * 2 + 1];

            igraph_real_t wc = step;
            if (wc > lr_cap) wc = lr_cap;

            igraph_real_t target_d = MATRIX(d, u, v);
            igraph_real_t y = MATRIX(*res, u, 0);
            igraph_real_t x = MATRIX(*res, u, 1);
            igraph_real_t b = MATRIX(*res, v, 0);
            igraph_real_t a = MATRIX(*res, v, 1);

            igraph_real_t delta = igraph_i_smds_geodesic(y, x, b, a);

            igraph_i_smds_gradient(y, x, b, a, &grad);

            /* g = gradient * 2 * (delta - target_d) */
            igraph_real_t factor = 2.0 * (delta - target_d);

            MATRIX(*res, u, 0) -= wc * MATRIX(grad, 0, 0) * factor;
            MATRIX(*res, u, 1) -= wc * MATRIX(grad, 0, 1) * factor;
            MATRIX(*res, v, 0) -= wc * MATRIX(grad, 1, 0) * factor;
            MATRIX(*res, v, 1) -= wc * MATRIX(grad, 1, 1) * factor;
        }
    }

    igraph_matrix_destroy(&grad);
    igraph_vector_destroy(&etas);
    igraph_vector_int_destroy(&indices);
    igraph_matrix_destroy(&d);
    IGRAPH_FINALLY_CLEAN(4);

    return IGRAPH_SUCCESS;
}
