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
#include "igraph_barnes_hut.h"
#include "igraph_progress.h"
#include "igraph_step.h"
#include <math.h>
#include <omp.h>

/* Structure to pass configuration and accumulated Z (sum_Q) to the force functions */
typedef struct {
    igraph_real_t sum_Q;
    igraph_real_t p_multiplier;
    igraph_real_t total_weight;
    igraph_integer_t dim;
} tsne_force_data_t;

/* Helper sign function for the adaptive learning rate */
static int sign_real(igraph_real_t x) {
    return (x > 0.0) ? 1 : ((x < 0.0) ? -1 : 0);
}

/* Repulsive force approximation via Barnes-Hut */
static void tsne_repulsion_kernel(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t dx,
    igraph_real_t dy,
    igraph_real_t dz,
    igraph_real_t dist_sq,
    igraph_real_t force[3],
    void *user_data
) {
    tsne_force_data_t *data = (tsne_force_data_t*)user_data;

    /* LAYOUT EXCLUSIVE LOGIC: Handle coordinate collisions deterministically.
       (p2->id == -1 indicates p2 is a macroscopic pseudo-node) */
    if (dist_sq < 1e-12) {
        igraph_real_t epsilon = (p2->id == -1 || p1->id > p2->id) ? 1e-5 : -1e-5;
        dx += epsilon;
        dy += epsilon;
        if (data->dim == 3) dz += epsilon;
        dist_sq = dx*dx + dy*dy + dz*dz;
    }

    /* Student-t distribution q_ij (unnormalized) */
    igraph_real_t q_ij = 1.0 / (1.0 + dist_sq);

    /* Accumulate normalization sum_Q (Z) */
    data->sum_Q += p2->mass * q_ij;

    /* Repulsive multiplier */
    igraph_real_t mult = p2->mass * q_ij * q_ij;

    /* Apply to vector. Branchless execution (dz is naturally 0.0 in 2D mode) */
    force[0] = mult * dx;
    force[1] = mult * dy;

    if (data->dim == 3) force[2] = mult * dz;
        else force[2] = 0.0; /* Play it completely safe with the array memory */
}

/* Exact attractive forces strictly applied on non-zero P edges */
static void tsne_attraction_kernel(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t dx,
    igraph_real_t dy,
    igraph_real_t dz,
    igraph_real_t dist_sq,
    igraph_real_t weight,
    igraph_real_t force_p1[3],
    igraph_real_t force_p2[3],
    void *user_data
) {
    tsne_force_data_t *data = (tsne_force_data_t*)user_data;

    if (dist_sq < 1e-12)  {
        force_p1[0] = force_p1[1] = force_p1[2] = 0.0;
        force_p2[0] = force_p2[1] = force_p2[2] = 0.0;
        return; /* Safely ignore zero-distance exact overlaps */
    }
    igraph_real_t p_ij = weight / data->total_weight;
    igraph_real_t q_ij = 1.0 / (1.0 + dist_sq);
    igraph_real_t mult = data->p_multiplier * p_ij * q_ij;

    /* Matches bhtsne: pos_f[n] += p_ij * q_ij * (y_n - y_m) */
    force_p1[0] = mult * dx;
    force_p1[1] = mult * dy;
    force_p1[2] = mult * dz;

    /* Apply symmetric Newton's Third Law forces back to p2 */
    force_p2[0] = -force_p1[0];
    force_p2[1] = -force_p1[1];
    force_p2[2] = -force_p1[2];
}

/* Re-center layout around origin to prevent drift */
static void igraph_i_tsne_center_layout(igraph_matrix_t *layout) {
    igraph_integer_t n = igraph_matrix_nrow(layout);
    igraph_integer_t dim = igraph_matrix_ncol(layout);

    for (igraph_integer_t d = 0; d < dim; d++) {
        igraph_real_t mean = 0.0;
        #pragma omp parallel for reduction(+:mean) schedule(static)
        for (igraph_integer_t i = 0; i < n; i++) {
            mean += MATRIX(*layout, i, d);
        }
        mean /= n;
        #pragma omp parallel for schedule(static)
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

    /* Init BH tree with dynamically-scaled parameters */
    igraph_bh_tree_t tree;
    igraph_integer_t bh_max_level, bh_leaf_capacity;
    igraph_bh_tree_get_scaling_params(no_nodes, ndim, &bh_max_level, &bh_leaf_capacity);
    IGRAPH_CHECK(igraph_bh_tree_init(&tree, ndim, theta, bh_max_level, bh_leaf_capacity));

    tsne_force_data_t force_data;
    force_data.dim = ndim;

    /* Compute total edge weight for P matrix normalization */
    force_data.total_weight = 0.0;
    for (igraph_integer_t e = 0; e < no_edges; e++) {
        force_data.total_weight += weights ? VECTOR(*weights)[e] : 1.0;
    }
    if (force_data.total_weight <= 0.0) force_data.total_weight = 1.0;

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

        /* 1. Calculate Repulsive Force (and Z) via O(N log N) Tree */
        force_data.sum_Q = 0.0;
        igraph_matrix_null(&neg_f);
        IGRAPH_CHECK(igraph_bh_apply_repulsion_from_tree(&tree, &neg_f, tsne_repulsion_kernel, &force_data));

        if (force_data.sum_Q <= 0.0) force_data.sum_Q = 1e-12;

        /* 2. Calculate Attractive Force via O(E) Edge List */
        igraph_matrix_null(&pos_f);
        IGRAPH_CHECK(igraph_bh_apply_attraction_from_edges(&tree, &from, &to, weights, &pos_f, tsne_attraction_kernel, &force_data));

        /* 3. Gradient update w/ adaptive learning rates */
        #pragma omp parallel for schedule(static)
        for (igraph_integer_t idx = 0; idx < no_nodes * ndim; idx++) {
            igraph_integer_t i = idx / ndim;
            igraph_integer_t d = idx % ndim;

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
            if (current_gain > 10.0) current_gain = 10.0;

            MATRIX(gains, i, d) = current_gain;
            current_uY = momentum * current_uY - eta * current_gain * dC;

            /* Clamp velocity magnitude to prevent gradient explosion */
            if (current_uY > 5.0) current_uY = 5.0;
            if (current_uY < -5.0) current_uY = -5.0;

            MATRIX(uY, i, d) = current_uY;
            MATRIX(*res, i, d) += current_uY;
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
igraph_error_t igraph_layout_bhtsne(const igraph_t *graph,
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
igraph_error_t igraph_layout_bhtsne_3d(const igraph_t *graph,
                                     igraph_matrix_t *res,
                                     igraph_bool_t use_seed,
                                     const igraph_vector_t *weights,
                                     igraph_integer_t epochs,
                                     igraph_real_t theta) {
    return igraph_i_layout_tsne_barnes_hut(graph, res, use_seed, weights, epochs, theta, 3);
}
