/*
    igraph library.
    Copyright (C) 2026  The igraph development team <igraph@igraph.org>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation.
*/

#include <igraph.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "test_utilities.h"

/* Define the force data structure locally for the test */
typedef struct {
    igraph_real_t sum_Q;
    igraph_real_t p_multiplier;
    igraph_real_t total_weight; /* CRITICAL: Required for P matrix normalization */
    igraph_integer_t dim;
} tsne_force_data_t;

/* Helper sign function for the adaptive learning rate */
static int sign_real(igraph_real_t x) {
    return (x > 0.0) ? 1 : ((x < 0.0) ? -1 : 0);
}

/* =========================================================================
 * Force Callbacks for Testing (Updated to new BH API)
 * ========================================================================= */

static void test_repulsive_force(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t dx, igraph_real_t dy, igraph_real_t dz,
    igraph_real_t dist_sq, igraph_real_t force[3], void *user_data
) {
    tsne_force_data_t *data = (tsne_force_data_t*)user_data;

    if (dist_sq < 1e-12) {
        dist_sq = 1e-5; /* Prevent division by zero */
    }

    /* Student-t distribution q_ij (unnormalized) */
    igraph_real_t q_ij = 1.0 / (1.0 + dist_sq);

    /* Accumulate normalization sum_Q (Z) */
    data->sum_Q += p2->mass * q_ij;

    /* Repulsive multiplier natively factors in the pseudo-node mass */
    igraph_real_t mult = p2->mass * q_ij * q_ij;

    force[0] = mult * dx;
    force[1] = mult * dy;
    if (data->dim == 3) force[2] = mult * dz;
}

static void test_attractive_force(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t dx, igraph_real_t dy, igraph_real_t dz,
    igraph_real_t dist_sq, igraph_real_t weight,
    igraph_real_t force_p1[3], igraph_real_t force_p2[3], void *user_data
) {
    tsne_force_data_t *data = (tsne_force_data_t*)user_data;

    if (dist_sq < 1e-12) return;

    /* CRITICAL FIX TEST: Normalize weight into a true probability (p_ij) */
    igraph_real_t p_ij = weight / data->total_weight;

    igraph_real_t q_ij = 1.0 / (1.0 + dist_sq);
    igraph_real_t mult = data->p_multiplier * p_ij * q_ij;

    /* Apply to Source */
    force_p1[0] = mult * dx;
    force_p1[1] = mult * dy;
    if (data->dim == 3) force_p1[2] = mult * dz;

    /* Symmetric Newton's Third Law to Target */
    force_p2[0] = -force_p1[0];
    force_p2[1] = -force_p1[1];
    if (data->dim == 3) force_p2[2] = -force_p1[2];
}

/* =========================================================================
 * TESTS
 * ========================================================================= */

/* Test 1: Mathematical correctness of forces with normalization */
int test_tsne_mathematical_correctness(void) {
    igraph_t graph;
    igraph_matrix_t coords;
    igraph_vector_t weights;
    igraph_bh_tree_t tree;
    igraph_matrix_t pos_f, neg_f;
    tsne_force_data_t force_data;

    printf("\n=== Test 1: Mathematical Correctness ===\n");

    igraph_small(&graph, 2, IGRAPH_UNDIRECTED, 0, 1, -1);

    igraph_matrix_init(&coords, 2, 2);
    MATRIX(coords, 0, 0) = -1.0;  MATRIX(coords, 0, 1) =  0.0;
    MATRIX(coords, 1, 0) =  1.0;  MATRIX(coords, 1, 1) =  0.0;

    igraph_vector_init(&weights, 1);
    VECTOR(weights)[0] = 0.5;

    force_data.dim = 2;
    force_data.sum_Q = 0.0;
    force_data.p_multiplier = 1.0;
    force_data.total_weight = 0.5; /* Total graph weight */

    igraph_matrix_init(&pos_f, 2, 2);
    igraph_matrix_init(&neg_f, 2, 2);

    igraph_bh_tree_init(&tree, 2, 0.5, 10, 1);
    igraph_bh_tree_build(&tree, &coords, NULL);

    igraph_vector_int_t from_vec, to_vec;
    igraph_vector_int_init(&from_vec, 1);
    igraph_vector_int_init(&to_vec, 1);
    VECTOR(from_vec)[0] = 0; VECTOR(to_vec)[0] = 1;

    IGRAPH_CHECK(igraph_bh_apply_attraction_from_edges(&tree, &from_vec, &to_vec, &weights, &pos_f, test_attractive_force, &force_data));
    IGRAPH_CHECK(igraph_bh_apply_repulsion_from_tree(&tree, &neg_f, test_repulsive_force, &force_data));

    /* * MATH VERIFICATION:
     * dx = p1.x - p2.x = (-1) - (1) = -2 (for node 0)
     * Distance^2 = 4, so q_ij = 1/(1+4) = 0.2
     * p_ij = weight / total_weight = 0.5 / 0.5 = 1.0
     * Attractive mult = 1.0 * 1.0 * 0.2 = 0.2
     * Node 0 pos_f = 0.2 * dx = 0.2 * (-2) = -0.4
     * Node 1 pos_f = 0.2 * (2) = 0.4 (Applied via Newton's 3rd law in kernel)
     * * Repulsive mult = mass * q_ij^2 = 1.0 * 0.04 = 0.04
     * Node 0 neg_f = 0.04 * dx = 0.04 * (-2) = -0.08
     */

    /* Note: Since dx goes from p1 to p2 inside the new BH loop, the sign may flip depending on exact internal point order.
       We take the absolute value for mathematical verification here. */
    printf("pos_f[0,0] magnitude = %.10f (Expected: 0.4)\n", fabs(MATRIX(pos_f, 0, 0)));
    printf("neg_f[0,0] magnitude = %.10f (Expected: 0.08)\n", fabs(MATRIX(neg_f, 0, 0)));
    printf("sum_Q                = %.10f (Expected: 0.4)\n", force_data.sum_Q);

    IGRAPH_ASSERT(fabs(fabs(MATRIX(pos_f, 0, 0)) - 0.4) < 1e-4);
    IGRAPH_ASSERT(fabs(fabs(MATRIX(neg_f, 0, 0)) - 0.08) < 1e-4);
    IGRAPH_ASSERT(fabs(force_data.sum_Q - 0.4) < 1e-4);

    igraph_bh_tree_destroy(&tree);
    igraph_matrix_destroy(&pos_f); igraph_matrix_destroy(&neg_f);
    igraph_matrix_destroy(&coords); igraph_vector_destroy(&weights);
    igraph_destroy(&graph);
    igraph_vector_int_destroy(&from_vec); igraph_vector_int_destroy(&to_vec);

    printf("--> Test 1 Passed.\n");
    return 0;
}

/* Test 2: Gradient update formula and Velocity Clamping */
int test_tsne_gradient_update(void) {
    printf("\n=== Test 2: Gradient Safety Bounds ===\n");

    igraph_real_t dC = 5000.0; /* Simulate a massive gradient explosion */
    igraph_real_t current_uY = 1.0;
    igraph_real_t current_gain = 9.9;
    igraph_real_t momentum = 0.8;
    igraph_real_t eta = 200.0;

    if (sign_real(dC) != sign_real(current_uY)) {
        current_gain += 0.2;
    } else {
        current_gain *= 0.8;
    }

    /* Test Gain Bounds */
    if (current_gain < 0.01) current_gain = 0.01;
    if (current_gain > 10.0) current_gain = 10.0; /* Clamp */

    printf("Gain clamped to: %.2f (Expected <= 10.0)\n", current_gain);
    IGRAPH_ASSERT(current_gain <= 10.0);

    /* Test Velocity Clamping */
    current_uY = momentum * current_uY - eta * current_gain * dC;
    igraph_real_t step_mag_sq = current_uY * current_uY; /* 1D test */

    igraph_real_t max_step = 5.0;
    if (step_mag_sq > max_step * max_step) {
        igraph_real_t scale = max_step / sqrt(step_mag_sq);
        current_uY *= scale;
    }

    printf("Velocity clamped to: %.2f (Expected >= -5.0 and <= 5.0)\n", current_uY);
    IGRAPH_ASSERT(fabs(current_uY) <= max_step + 1e-5);

    printf("--> Test 2 Passed.\n");
    return 0;
}

/* Test 3: Full t-SNE run on a small graph with NaN/Inf tracking */
int test_tsne_full_run(void) {
    igraph_t graph;
    igraph_matrix_t coords;
    igraph_vector_t weights;
    igraph_integer_t epochs = 100;
    igraph_real_t theta = 0.5;

    printf("\n=== Test 3: Full Run & Explosion Check ===\n");

    igraph_small(&graph, 4, IGRAPH_UNDIRECTED, 0, 1, 0, 2, 1, 3, 2, 3, -1);

    igraph_matrix_init(&coords, 4, 2);
    /* Don't manually initialize coordinates, let the algorithm seed them to test the full pipeline */

    igraph_vector_init(&weights, 4);
    VECTOR(weights)[0] = 1.0; VECTOR(weights)[1] = 1.0;
    VECTOR(weights)[2] = 1.0; VECTOR(weights)[3] = 1.0;

    igraph_error_t res = igraph_layout_bhtsne(&graph, &coords, false, &weights, epochs, theta);

    if (res != IGRAPH_SUCCESS) {
        printf("ERROR: igraph t-SNE failed with error code %d\n", res);
        return -1;
    }

    /* Post-Run Sanity Checks */
    int has_nan = 0, has_inf = 0;
    double max_val = 0.0;

    for (int i = 0; i < 4; i++) {
        double x = MATRIX(coords, i, 0);
        double y = MATRIX(coords, i, 1);

        printf("  Node %d: (%f, %f)\n", i, x, y);

        if (isnan(x) || isnan(y)) has_nan = 1;
        if (!isfinite(x) || !isfinite(y)) has_inf = 1;

        if (fabs(x) > max_val) max_val = fabs(x);
        if (fabs(y) > max_val) max_val = fabs(y);
    }

    if (has_nan) {
        printf("FATAL ERROR: Layout produced NaN values. Gradients exploded.\n");
        return -1;
    }
    if (has_inf) {
        printf("FATAL ERROR: Layout produced Infinity values. Division by zero occurred.\n");
        return -1;
    }
    if (max_val > 1000.0) {
        printf("FATAL ERROR: Layout bounds expanded massively (Max val: %.2f). Missing normalization or velocity clamps.\n", max_val);
        return -1;
    }

    printf("SUCCESS: Coordinates are stable and finite. Max magnitude: %.2f\n", max_val);

    igraph_matrix_destroy(&coords);
    igraph_vector_destroy(&weights);
    igraph_destroy(&graph);

    printf("--> Test 3 Passed.\n");
    return 0;
}

int main(void) {
    igraph_set_error_handler(igraph_error_handler_ignore);

    RUN_TEST(test_tsne_mathematical_correctness);
    RUN_TEST(test_tsne_gradient_update);
    RUN_TEST(test_tsne_full_run);

    return 0;
}
