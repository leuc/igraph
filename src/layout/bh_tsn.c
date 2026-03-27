/*
   igraph library.
   Copyright (C) 2026 The igraph development team <igraph@igraph.org>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation.
*/

#include "igraph_layout.h"
#include "igraph_interface.h"
#include "igraph_matrix.h"
#include "igraph_random.h"
#include "core/interruption.h"
#include "core/barnes_hut.h"
#include "igraph_progress.h"
#include "igraph_step.h"
#include <math.h>

/* Structure to pass configuration and accumulated Z (sum_Q) to the force functions */
typedef struct {
    igraph_real_t sum_Q;
    igraph_real_t p_multiplier;
    igraph_integer_t dim;
} tsne_force_data_t;

/* Helper sign function for the adaptive learning rate */
static int sign_real(igraph_real_t x) {
    return (x > 0.0) ? 1 : ((x < 0.0) ? -1 : 0);
}

/* Repulsive force approximation via Barnes-Hut */
static void tsne_repulsive_force(const igraph_bh_point_t *p1, const igraph_bh_point_t *p2, igraph_real_t force[3], void *user_data) {
    tsne_force_data_t *data = (tsne_force_data_t*)user_data;
    igraph_real_t dist_sq = 0.0;

    for (igraph_integer_t d = 0; d < data->dim; d++) {
        igraph_real_t diff = p1->coord[d] - p2->coord[d];
        dist_sq += diff * diff;
    }

    /* Student-t distribution q_ij (unnormalized) */
    igraph_real_t q_ij = 1.0 / (1.0 + dist_sq);

    /* Accumulate normalization sum_Q (Z) */
    data->sum_Q += p2->mass * q_ij;

    /* Repulsive multiplier */
    igraph_real_t mult = p2->mass * q_ij * q_ij;
    for (igraph_integer_t d = 0; d < data->dim; d++) {
        force[d] = mult * (p1->coord[d] - p2->coord[d]);
    }
}

/* Exact attractive forces strictly applied on non-zero P edges */
static void tsne_attractive_force(const igraph_bh_point_t *p1, const igraph_bh_point_t *p2, igraph_real_t force[3], void *user_data) {
    tsne_force_data_t *data = (tsne_force_data_t*)user_data;
    igraph_real_t dist_sq = 0.0;

    for (igraph_integer_t d = 0; d < data->dim; d++) {
        igraph_real_t diff = p1->coord[d] - p2->coord[d];
        dist_sq += diff * diff;
    }

    /* Student-t distribution q_ij */
    igraph_real_t q_ij = 1.0 / (1.0 + dist_sq);

    /* Apply early exaggeration multiplier directly to the attractive force */
    igraph_real_t mult = data->p_multiplier * q_ij;

    for (igraph_integer_t d = 0; d < data->dim; d++) {
        force[d] = mult * (p1->coord[d] - p2->coord[d]);
    }
}

/* Re-center layout around origin to prevent drift */
static void igraph_i_tsne_center_layout(igraph_matrix_t *layout) {
    igraph_integer_t n = igraph_matrix_nrow(layout);
    igraph_integer_t dim = igraph_matrix_ncol(layout);

    for (igraph_integer_t d = 0; d < dim; d++) {
        igraph_real_t mean = 0.0;
        for (igraph_integer_t i = 0; i < n; i++) {
            mean += MATRIX(*layout, i, d);
        }
        mean /= n;
        for (igraph_integer_t i = 0; i < n; i++) {
            MATRIX(*layout, i, d) -= mean;
        }
    }
}

/* Core t-SNE Gradient Descent loop */
static igraph_error_t igraph_i_layout_tsne_barnes_hut(
    const igraph_t *graph,
    igraph_matrix_t *res,
    igraph_bool_t use_seed,
    const igraph_vector_t *weights,
    igraph_integer_t epochs,
    igraph_real_t theta,
    igraph_integer_t ndim
) {
    igraph_integer_t no_nodes = igraph_vcount(graph);
    igraph_integer_t no_edges = igraph_ecount(graph);

    if (no_nodes <= 1) {
        IGRAPH_CHECK(igraph_matrix_resize(res, no_nodes, ndim));
        igraph_matrix_null(res);
        return IGRAPH_SUCCESS;
    }

    /* Initialize coordinates */
    if (use_seed) {
        if (igraph_matrix_nrow(res) != no_nodes || igraph_matrix_ncol(res) != ndim) {
            IGRAPH_ERRORF("Seed layout should have %" IGRAPH_PRId " points in %" IGRAPH_PRId " dimensions.",
                IGRAPH_EINVAL, no_nodes, ndim);
        }
    } else {
        IGRAPH_CHECK(igraph_matrix_resize(res, no_nodes, ndim));

        igraph_rng_t *rng = igraph_rng_default();
        for (igraph_integer_t i = 0; i < no_nodes; i++) {
            for (igraph_integer_t d = 0; d < ndim; d++) {
                /* Start with small variance normal distribution.
                    * Variance = 0.0001 means Standard Deviation = 0.01
                    */
                MATRIX(*res, i, d) = igraph_rng_get_normal(rng, 0.0, 0.01);
            }
        }
    }
    igraph_i_tsne_center_layout(res);

    /* Prep vectors mapping to the graph's edge list */
    igraph_vector_int_t from, to;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&from, no_edges);
    IGRAPH_VECTOR_INT_INIT_FINALLY(&to, no_edges);
    for (igraph_integer_t e = 0; e < no_edges; e++) {
        VECTOR(from)[e] = IGRAPH_FROM(graph, e);
        VECTOR(to)[e] = IGRAPH_TO(graph, e);
    }

    /* Gradient descent allocations */
    igraph_matrix_t uY, gains, pos_f, neg_f;
    IGRAPH_MATRIX_INIT_FINALLY(&uY, no_nodes, ndim);
    IGRAPH_MATRIX_INIT_FINALLY(&gains, no_nodes, ndim);
    IGRAPH_MATRIX_INIT_FINALLY(&pos_f, no_nodes, ndim);
    IGRAPH_MATRIX_INIT_FINALLY(&neg_f, no_nodes, ndim);

    for (igraph_integer_t i = 0; i < no_nodes; i++) {
        for (igraph_integer_t d = 0; d < ndim; d++) {
            MATRIX(gains, i, d) = 1.0;
        }
    }

    /* Init BH tree (quadtree/octree config) */
    igraph_bh_tree_t tree;
    IGRAPH_CHECK(igraph_bh_tree_init(&tree, ndim, theta, 0, 0));

    tsne_force_data_t force_data;
    force_data.dim = ndim;

    /* t-SNE specific learning constants */
    igraph_real_t eta = 200.0;
    igraph_integer_t stop_lying_iter = 250;
    igraph_integer_t mom_switch_iter = 250;

    for (igraph_integer_t iter = 0; iter < epochs; iter++) {
        IGRAPH_PROGRESS(NULL, (100.0 * iter) / epochs, NULL);
        IGRAPH_ALLOW_INTERRUPTION();

        /* Switch hyperparameters mid-run */
        force_data.p_multiplier = (iter < stop_lying_iter) ? 12.0 : 1.0;
        igraph_real_t momentum = (iter < mom_switch_iter) ? 0.5 : 0.8;

        /* Dynamically assemble spatial tree this iteration */
        IGRAPH_CHECK(igraph_bh_tree_build(&tree, res, NULL));

        /* 1. Calculate Repulsive Force (and Z) */
        force_data.sum_Q = 0.0;
        IGRAPH_CHECK(igraph_bh_calculate_repulsive_forces(&tree, &neg_f, tsne_repulsive_force, &force_data));

        if (force_data.sum_Q <= 0.0) force_data.sum_Q = 1e-12;

        /* 2. Calculate Attractive Force */
        igraph_matrix_null(&pos_f);
        IGRAPH_CHECK(igraph_bh_calculate_attractive_forces(&tree, &from, &to, weights, &pos_f, tsne_attractive_force, &force_data));

        /* 3. Gradient update w/ adaptive learning rates */
        for (igraph_integer_t i = 0; i < no_nodes; i++) {
            for (igraph_integer_t d = 0; d < ndim; d++) {

                igraph_real_t dC = MATRIX(pos_f, i, d) - (MATRIX(neg_f, i, d) / force_data.sum_Q);
                igraph_real_t current_uY = MATRIX(uY, i, d);
                igraph_real_t current_gain = MATRIX(gains, i, d);

                if (sign_real(dC) != sign_real(current_uY)) {
                    current_gain += 0.2;
                } else {
                    current_gain *= 0.8;
                }
                if (current_gain < 0.01) {
                    current_gain = 0.01;
                }

                MATRIX(gains, i, d) = current_gain;
                current_uY = momentum * current_uY - eta * current_gain * dC;

                MATRIX(uY, i, d) = current_uY;
                MATRIX(*res, i, d) += current_uY;
            }
        }
        igraph_i_tsne_center_layout(res);
        IGRAPH_STEP(res, NULL);
    }

    igraph_bh_tree_destroy(&tree);

    /* Free memory */
    igraph_matrix_destroy(&neg_f);
    igraph_matrix_destroy(&pos_f);
    igraph_matrix_destroy(&gains);
    igraph_matrix_destroy(&uY);
    igraph_vector_int_destroy(&to);
    igraph_vector_int_destroy(&from);
    IGRAPH_FINALLY_CLEAN(6);

    return IGRAPH_SUCCESS;
}

/**
 * \function igraph_layout_tsne
 * \brief 2D Layout using Barnes-Hut t-Distributed Stochastic Neighbor Embedding (t-SNE).
 */
igraph_error_t igraph_layout_tsne(const igraph_t *graph,
                                  igraph_matrix_t *res,
                                  igraph_bool_t use_seed,
                                  const igraph_vector_t *weights,
                                  igraph_integer_t epochs,
                                  igraph_real_t theta) {
    return igraph_i_layout_tsne_barnes_hut(graph, res, use_seed, weights, epochs, theta, 2);
}

/**
 * \function igraph_layout_tsne_3d
 * \brief 3D Layout using Barnes-Hut t-Distributed Stochastic Neighbor Embedding (t-SNE).
 */
igraph_error_t igraph_layout_tsne_3d(const igraph_t *graph,
                                     igraph_matrix_t *res,
                                     igraph_bool_t use_seed,
                                     const igraph_vector_t *weights,
                                     igraph_integer_t epochs,
                                     igraph_real_t theta) {
    return igraph_i_layout_tsne_barnes_hut(graph, res, use_seed, weights, epochs, theta, 3);
}
