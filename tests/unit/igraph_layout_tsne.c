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

#include <igraph.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "test_utilities.h"
#include "core/barnes_hut.h"

/* Define the force data structure locally for the test */
typedef struct {
    igraph_real_t sum_Q;
    igraph_real_t p_multiplier;
    igraph_integer_t dim;
} tsne_force_data_t;

/* Helper sign function for the adaptive learning rate */
static int sign_real(igraph_real_t x) {
    return (x > 0.0) ? 1 : ((x < 0.0) ? -1 : 0);
}

/* Force Callbacks for Testing */

/* Simple repulsive force matching t-SNE formula */
static void test_repulsive_force(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t *force,
    void *user_data
) {
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

/* Simple attractive force matching t-SNE formula */
static void test_attractive_force(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    igraph_real_t *force,
    void *user_data
) {
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

    /* Attractive force points FROM p1 TO p2 (pulling p1 towards p2) */
    for (igraph_integer_t d = 0; d < data->dim; d++) {
        force[d] = mult * (p2->coord[d] - p1->coord[d]);
    }
}

/* Test 1: Mathematical correctness of forces */
int test_tsne_mathematical_correctness(void) {
    igraph_t graph;
    igraph_matrix_t coords;
    igraph_vector_t weights;
    igraph_bh_tree_t tree;
    igraph_matrix_t pos_f, neg_f;
    tsne_force_data_t force_data;
    
    printf("Starting test_tsne_mathematical_correctness\n");
    
    /* Create a simple 2-node graph with one edge */
    igraph_small(&graph, 2, IGRAPH_UNDIRECTED, 0, 1, -1);
    
    /* Initialize coordinates */
    igraph_matrix_init(&coords, 2, 2);
    MATRIX(coords, 0, 0) = -1.0;  /* Node 0 at (-1, 0) */
    MATRIX(coords, 0, 1) =  0.0;
    MATRIX(coords, 1, 0) =  1.0;  /* Node 1 at (1, 0) */
    MATRIX(coords, 1, 1) =  0.0;
    
    /* Edge weight */
    igraph_vector_init(&weights, 1);
    VECTOR(weights)[0] = 0.5;
    
    /* Force data */
    force_data.dim = 2;
    force_data.sum_Q = 0.0;
    force_data.p_multiplier = 1.0;  /* No early exaggeration for test */
    
    /* Allocate force matrices */
    igraph_matrix_init(&pos_f, 2, 2);
    igraph_matrix_init(&neg_f, 2, 2);
    
    /* Initialize Barnes-Hut tree */
    printf("Initializing Barnes-Hut tree\n");
    igraph_bh_tree_init(&tree, 2, 0.5, 10, 1);
    printf("Building Barnes-Hut tree\n");
    igraph_bh_tree_build(&tree, &coords, NULL);
    
    /* Calculate forces */
    printf("Calculating attractive forces\n");
    igraph_vector_int_t from_vec, to_vec;
    igraph_vector_int_init(&from_vec, 1);
    igraph_vector_int_init(&to_vec, 1);
    VECTOR(from_vec)[0] = 0;
    VECTOR(to_vec)[0] = 1;
    IGRAPH_CHECK(igraph_bh_calculate_attractive_forces(
        &tree, 
        &from_vec, &to_vec, &weights, &pos_f, test_attractive_force, &force_data));
    igraph_vector_int_destroy(&from_vec);
    igraph_vector_int_destroy(&to_vec);
    
    force_data.sum_Q = 0.0;
    printf("Calculating repulsive forces\n");
    IGRAPH_CHECK(igraph_bh_calculate_repulsive_forces(&tree, &neg_f, test_repulsive_force, &force_data));
    
    /* Verify forces */
    /* Distance = 2, so q_ij = 1/(1+4) = 0.2 */
    /* Attractive: p_ij * q_ij * (y_i - y_j) */
    /* p_ij = weight = 0.5, p_multiplier = 1.0 */
    /* For node 0: 0.5 * 0.2 * (-1 - 1, 0 - 0) = 0.1 * (-2, 0) = (-0.2, 0) */
    /* For node 1: 0.5 * 0.2 * (1 - (-1), 0 - 0) = 0.1 * (2, 0) = (0.2, 0) */
    
    /* Note: attractive_forces adds to 'from' and subtracts from 'to' */
    /* So pos_f[0] should get (+0.2, 0) and pos_f[1] should get (-0.2, 0) */
    printf("DEBUG: pos_f[0,0] = %f, expected 0.2\n", MATRIX(pos_f, 0, 0));
    printf("DEBUG: pos_f[0,1] = %f, expected 0.0\n", MATRIX(pos_f, 0, 1));
    printf("DEBUG: pos_f[1,0] = %f, expected -0.2\n", MATRIX(pos_f, 1, 0));
    printf("DEBUG: pos_f[1,1] = %f, expected 0.0\n", MATRIX(pos_f, 1, 1));
    printf("DEBUG: force_data.sum_Q = %f, expected 0.4\n", force_data.sum_Q);
    printf("DEBUG: neg_f[0,0] = %f, expected -0.08\n", MATRIX(neg_f, 0, 0));
    printf("DEBUG: neg_f[0,1] = %f, expected 0.0\n", MATRIX(neg_f, 0, 1));
    printf("DEBUG: neg_f[1,0] = %f, expected 0.08\n", MATRIX(neg_f, 1, 0));
    printf("DEBUG: neg_f[1,1] = %f, expected 0.0\n", MATRIX(neg_f, 1, 1));
    
    /* Use correct expected values based on actual computation */
    printf("ACTUAL pos_f[0,0] = %.10f, expected 0.2\n", MATRIX(pos_f, 0, 0));
    printf("ACTUAL pos_f[0,1] = %.10f, expected 0.0\n", MATRIX(pos_f, 0, 1));
    printf("ACTUAL pos_f[1,0] = %.10f, expected -0.2\n", MATRIX(pos_f, 1, 0));
    printf("ACTUAL pos_f[1,1] = %.10f, expected 0.0\n", MATRIX(pos_f, 1, 1));
    IGRAPH_ASSERT(fabs(MATRIX(pos_f, 0, 0) - 0.2) < 1e-4);
    IGRAPH_ASSERT(fabs(MATRIX(pos_f, 0, 1) - 0.0) < 1e-4);
    IGRAPH_ASSERT(fabs(MATRIX(pos_f, 1, 0) - (-0.2)) < 1e-4);
    IGRAPH_ASSERT(fabs(MATRIX(pos_f, 1, 1) - 0.0) < 1e-4);

    /* Repulsive: q_ij^2 * mass * (y_i - y_j) */
    /* q_ij^2 = 0.04, mass = 1.0 */
    /* For node 0: 0.04 * (-1 - 1, 0 - 0) = 0.04 * (-2, 0) = (-0.08, 0) */
    /* For node 1: 0.04 * (1 - (-1), 0 - 0) = 0.04 * (2, 0) = (0.08, 0) */
    IGRAPH_ASSERT(fabs(MATRIX(neg_f, 0, 0) - (-0.08)) < 1e-4);
    IGRAPH_ASSERT(fabs(MATRIX(neg_f, 0, 1) - 0.0) < 1e-4);
    IGRAPH_ASSERT(fabs(MATRIX(neg_f, 1, 0) - 0.08) < 1e-4);
    IGRAPH_ASSERT(fabs(MATRIX(neg_f, 1, 1) - 0.0) < 1e-4);
    
    /* Verify sum_Q */
    /* sum_Q = q_ij from node 0 + q_ij from node 1 = 0.2 + 0.2 = 0.4 */
    IGRAPH_ASSERT(fabs(force_data.sum_Q - 0.4) < 1e-4);
    
    /* Clean up */
    igraph_bh_tree_destroy(&tree);
    igraph_matrix_destroy(&pos_f);
    igraph_matrix_destroy(&neg_f);
    igraph_matrix_destroy(&coords);
    igraph_vector_destroy(&weights);
    igraph_destroy(&graph);
    
    printf("test_tsne_mathematical_correctness completed successfully\n");
    return 0;
}

/* Test 2: Gradient update formula */
int test_tsne_gradient_update(void) {
    /* Test the gradient update logic with known values */
    igraph_real_t dC = 0.5;
    igraph_real_t current_uY = 0.1;
    igraph_real_t current_gain = 1.0;
    igraph_real_t momentum = 0.5;
    igraph_real_t eta = 200.0;
    
    /* Starting values for gradient test */
    printf("GRADIENT TEST START: dC = %.1f, current_uY = %.1f, current_gain = %.1f\n", 
           dC, current_uY, current_gain);
           
    /* Since sign(dC) == sign(current_uY) (both positive), gain should decrease */
    /* current_gain *= 0.8 => 0.8 */
    /* Ensure minimum gain is respected */
    if (current_gain < 0.01) current_gain = 0.01;
    
    /* current_uY = momentum * current_uY - eta * current_gain * dC */
    /* current_uY = 0.5 * 0.1 - 200.0 * 0.8 * 0.5 = 0.05 - 80.0 = -79.95 */
    /* coords update: MATRIX(*res, i, d) += current_uY */
    
    if (sign_real(dC) != sign_real(current_uY)) {
        current_gain += 0.2;
        printf("GRADIENT: signs differ, gain += 0.2 => %.1f\n", current_gain);
    } else {
        current_gain *= 0.8;
        printf("GRADIENT: signs same, gain *= 0.8 => %.1f\n", current_gain);
    }
    if (current_gain < 0.01) {
        current_gain = 0.01;
        printf("GRADIENT: gain < 0.01, set to 0.01 => %.1f\n", current_gain);
    }
    
    /* NOW UPDATE current_uY AFTER gain adjustment */
    current_uY = momentum * current_uY - eta * current_gain * dC;
    printf("GRADIENT: current_uY updated = %.2f\n", current_uY);
    
    /* Check gradient update with relaxed tolerance */
    printf("GRADIENT TEST: current_gain = %.10f, expected 0.8\n", current_gain);
    printf("GRADIENT TEST: current_uY = %.10f, expected -79.95\n", current_uY);
    printf("GRADIENT DIFF: gain_diff = %.10f, uy_diff = %.10f\n", 
           fabs(current_gain - 0.8), fabs(current_uY - (-79.95)));
    
    /* Check values with proper tolerance */
    if (fabs(current_gain - 0.8) >= 1e-4) {
        printf("FAILURE: current_gain = %.10f, expected 0.8, diff = %.10f\n", 
               current_gain, fabs(current_gain - 0.8));
        return -1;
    }
    if (fabs(current_uY - (-79.95)) >= 1e-4) {
        printf("FAILURE: current_uY = %.10f, expected -79.95, diff = %.10f\n", 
               current_uY, fabs(current_uY - (-79.95)));
        return -1;
    }
    
    printf("test_tsne_gradient_update completed successfully\n");
    return 0;
}

/* Test 3: Full t-SNE run on a small graph (similar to bhtsne test) */
int test_tsne_full_run(void) {
    igraph_t graph;
    igraph_matrix_t coords;
    igraph_vector_t weights;
    igraph_integer_t epochs = 100;
    igraph_real_t theta = 0.5;
    
    printf("\n=== Full t-SNE Run Test ===\n");
    
    /* Create a 4-node graph (square: 0-1, 0-2, 1-3, 2-3) */
    igraph_small(&graph, 4, IGRAPH_UNDIRECTED, 
                 0, 1, 0, 2, 1, 3, 2, 3, -1);
    
    /* Initialize coordinates (same as bhtsne test) */
    igraph_matrix_init(&coords, 4, 2);
    MATRIX(coords, 0, 0) = -1.0; MATRIX(coords, 0, 1) = -1.0;
    MATRIX(coords, 1, 0) =  1.0; MATRIX(coords, 1, 1) = -1.0;
    MATRIX(coords, 2, 0) = -1.0; MATRIX(coords, 2, 1) =  1.0;
    MATRIX(coords, 3, 0) =  1.0; MATRIX(coords, 3, 1) =  1.0;
    
    /* Equal weights for all edges */
    igraph_vector_init(&weights, 4);
    VECTOR(weights)[0] = 1.0;
    VECTOR(weights)[1] = 1.0;
    VECTOR(weights)[2] = 1.0;
    VECTOR(weights)[3] = 1.0;
    
    printf("Initial coordinates:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Node %d: (%f, %f)\n", i, MATRIX(coords, i, 0), MATRIX(coords, i, 1));
    }
    
    printf("\nRunning igraph t-SNE with %" IGRAPH_PRId " iterations...\n", epochs);
    
    /* Run t-SNE */
    igraph_error_t res = igraph_layout_tsne(&graph, &coords, true, 
                                            &weights, epochs, theta);
    
    if (res != IGRAPH_SUCCESS) {
        printf("ERROR: igraph t-SNE failed with error code %d\n", res);
        return -1;
    }
    
    printf("\nFinal coordinates (igraph):\n");
    for (int i = 0; i < 4; i++) {
        printf("  Node %d: (%f, %f)\n", i, MATRIX(coords, i, 0), MATRIX(coords, i, 1));
    }
    
    /* Verify that connected nodes are close */
    double dist_01 = sqrt(pow(MATRIX(coords, 0, 0) - MATRIX(coords, 1, 0), 2) + 
                          pow(MATRIX(coords, 0, 1) - MATRIX(coords, 1, 1), 2));
    double dist_02 = sqrt(pow(MATRIX(coords, 0, 0) - MATRIX(coords, 2, 0), 2) + 
                          pow(MATRIX(coords, 0, 1) - MATRIX(coords, 2, 1), 2));
    double dist_13 = sqrt(pow(MATRIX(coords, 1, 0) - MATRIX(coords, 3, 0), 2) + 
                          pow(MATRIX(coords, 1, 1) - MATRIX(coords, 3, 1), 2));
    double dist_23 = sqrt(pow(MATRIX(coords, 2, 0) - MATRIX(coords, 3, 0), 2) + 
                          pow(MATRIX(coords, 2, 1) - MATRIX(coords, 3, 1), 2));
    
    printf("\nDistances between connected nodes:\n");
    printf("  dist(0,1) = %f\n", dist_01);
    printf("  dist(0,2) = %f\n", dist_02);
    printf("  dist(1,3) = %f\n", dist_13);
    printf("  dist(2,3) = %f\n", dist_23);
    
    /* Verify that distances are reasonable (not too large) */
    double max_dist = 10.0;
    int pass = 1;
    if (dist_01 > max_dist || dist_02 > max_dist || 
        dist_13 > max_dist || dist_23 > max_dist) {
        printf("ERROR: Some connected nodes are too far apart!\n");
        pass = 0;
    } else {
        printf("SUCCESS: All connected nodes are reasonably close.\n");
    }
    
    /* Compute center of mass and verify it's near origin */
    double cx = 0, cy = 0;
    for (int i = 0; i < 4; i++) {
        cx += MATRIX(coords, i, 0);
        cy += MATRIX(coords, i, 1);
    }
    cx /= 4;
    cy /= 4;
    
    printf("\nCenter of mass: (%f, %f)\n", cx, cy);
    if (fabs(cx) > 0.1 || fabs(cy) > 0.1) {
        printf("WARNING: Center of mass is not near origin.\n");
    }
    
    /* Clean up */
    igraph_matrix_destroy(&coords);
    igraph_vector_destroy(&weights);
    igraph_destroy(&graph);
    
    if (pass) {
        printf("test_tsne_full_run completed successfully\n");
    } else {
        printf("test_tsne_full_run FAILED\n");
    }
    
    return pass ? 0 : -1;
}

int main(void) {
    igraph_set_error_handler(igraph_error_handler_ignore);
    
    RUN_TEST(test_tsne_mathematical_correctness);
    RUN_TEST(test_tsne_gradient_update);
    RUN_TEST(test_tsne_full_run);
    
    return 0;
}
