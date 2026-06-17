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

#include <igraph.h>
#include "test_utilities.h"

int main(void) {
    igraph_t g;
    igraph_matrix_t result;

    igraph_rng_seed(igraph_rng_default(), 42);

    printf("Empty graph.\n");
    igraph_small(&g, 0, 0, -1);
    igraph_matrix_init(&result, 0, 0);
    IGRAPH_ASSERT(igraph_layout_bcgl(&g, &result, /*use_seed*/ 0,
                  /*niter*/ 10, /*learning_rate*/ 0.01, /*momentum*/ 0.9,
                  IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T) == IGRAPH_SUCCESS);
    print_matrix(&result);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Singleton graph.\n");
    igraph_small(&g, 1, 0, -1);
    igraph_matrix_init(&result, 0, 0);
    IGRAPH_ASSERT(igraph_layout_bcgl(&g, &result, /*use_seed*/ 0,
                  /*niter*/ 10, /*learning_rate*/ 0.01, /*momentum*/ 0.9,
                  IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T) == IGRAPH_SUCCESS);
    print_matrix(&result);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Ring graph of 10 vertices.\n");
    igraph_ring(&g, 10, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 0, 0);
    IGRAPH_ASSERT(igraph_layout_bcgl(&g, &result, /*use_seed*/ 0,
                  /*niter*/ 100, /*learning_rate*/ 0.01, /*momentum*/ 0.9,
                  IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T) == IGRAPH_SUCCESS);
    IGRAPH_ASSERT(igraph_matrix_nrow(&result) == 10);
    IGRAPH_ASSERT(igraph_matrix_ncol(&result) == 2);
    print_matrix(&result);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Ring graph of 10 vertices, 3D.\n");
    igraph_ring(&g, 10, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 0, 0);
    IGRAPH_ASSERT(igraph_layout_bcgl_3d(&g, &result, /*use_seed*/ 0,
                   /*niter*/ 100, /*learning_rate*/ 0.01, /*momentum*/ 0.9,
                   IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T) == IGRAPH_SUCCESS);
    IGRAPH_ASSERT(igraph_matrix_nrow(&result) == 10);
    IGRAPH_ASSERT(igraph_matrix_ncol(&result) == 3);
    print_matrix(&result);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Test with seed.\n");
    igraph_ring(&g, 5, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 5, 2);
    MATRIX(result, 0, 0) = 0.1; MATRIX(result, 0, 1) = 0.2;
    MATRIX(result, 1, 0) = 0.3; MATRIX(result, 1, 1) = 0.4;
    MATRIX(result, 2, 0) = 0.5; MATRIX(result, 2, 1) = 0.6;
    MATRIX(result, 3, 0) = 0.7; MATRIX(result, 3, 1) = 0.8;
    MATRIX(result, 4, 0) = 0.9; MATRIX(result, 4, 1) = 1.0;
    IGRAPH_ASSERT(igraph_layout_bcgl(&g, &result, /*use_seed*/ 1,
                  /*niter*/ 10, /*learning_rate*/ 0.01, /*momentum*/ 0.9,
                  IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T) == IGRAPH_SUCCESS);
    IGRAPH_ASSERT(igraph_matrix_nrow(&result) == 5);
    IGRAPH_ASSERT(igraph_matrix_ncol(&result) == 2);
    print_matrix(&result);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Test error: invalid niter.\n");
    igraph_ring(&g, 5, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 0, 0);
    CHECK_ERROR(igraph_layout_bcgl(&g, &result, 0, -1, 0.01, 0.9,
                IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T), IGRAPH_EINVAL);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Test error: invalid learning_rate.\n");
    igraph_ring(&g, 5, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 0, 0);
    CHECK_ERROR(igraph_layout_bcgl(&g, &result, 0, 10, 0.0, 0.9,
                IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T), IGRAPH_EINVAL);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Test error: invalid momentum.\n");
    igraph_ring(&g, 5, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 0, 0);
    CHECK_ERROR(igraph_layout_bcgl(&g, &result, 0, 10, 0.01, 1.5,
                IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T), IGRAPH_EINVAL);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Test error: invalid seed size.\n");
    igraph_ring(&g, 5, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 3, 2);
    CHECK_ERROR(igraph_layout_bcgl(&g, &result, 1, 10, 0.01, 0.9,
                IGRAPH_LAYOUT_BCGL_DISTRIBUTION_STUDENT_T), IGRAPH_EINVAL);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    printf("Test error: Gaussian not yet implemented.\n");
    igraph_ring(&g, 5, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&result, 0, 0);
    CHECK_ERROR(igraph_layout_bcgl(&g, &result, 0, 10, 0.01, 0.9,
                IGRAPH_LAYOUT_BCGL_DISTRIBUTION_GAUSSIAN), IGRAPH_UNIMPLEMENTED);
    igraph_matrix_destroy(&result);
    igraph_destroy(&g);

    VERIFY_FINALLY_STACK();

    return 0;
}
