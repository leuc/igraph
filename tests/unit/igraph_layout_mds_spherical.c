/*
   igraph library.
   Copyright (C) 2006-2012  Gabor Csardi <csardi.gabor@gmail.com>
   334 Harvard st, Cambridge MA, 02139 USA

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc.,  51 Franklin Street, Fifth Floor, Boston, MA
   02110-1301 USA

*/

#include <igraph.h>
#include <math.h>
#include <stdlib.h>

#include "test_utilities.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define sqr(x) ((x)*(x))

static double compute_stress(const igraph_matrix_t *coords, const igraph_matrix_t *dist,
                              igraph_int_t n, const char *label) {
    igraph_int_t i, j;
    double stress = 0.0, denom = 0.0;
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            double dot = MATRIX(*coords, i, 0) * MATRIX(*coords, j, 0)
                       + MATRIX(*coords, i, 1) * MATRIX(*coords, j, 1)
                       + MATRIX(*coords, i, 2) * MATRIX(*coords, j, 2);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            double pred = acos(dot);
            double target = MATRIX(*dist, i, j);
            stress += (pred - target) * (pred - target);
            denom += target * target;
        }
    }
    double nstress = (denom > 0.0) ? stress / denom : 0.0;
    printf("  %s: raw_stress=%g norm_stress=%g\n", label, stress, nstress);
    return nstress;
}

static void check_on_sphere(const igraph_matrix_t *coords, igraph_int_t n, const char *label) {
    igraph_int_t i, j;
    for (i = 0; i < n; i++) {
        double norm = 0.0;
        for (j = 0; j < 3; j++) {
            norm += MATRIX(*coords, i, j) * MATRIX(*coords, i, j);
        }
        if (fabs(sqrt(norm) - 1.0) > 1e-6) {
            printf("%s: Vertex %" IGRAPH_PRId " not on unit sphere: norm = %g\n", label, i, sqrt(norm));
            exit(1);
        }
    }
    printf("  %s: all on sphere [%lld nodes]\n", label, (long long)n);
}

int main(void) {
    igraph_t g;
    igraph_matrix_t coords, dist_mat;
    igraph_int_t i, j;

    igraph_rng_seed(igraph_rng_default(), 42); /* make tests deterministic */

    /* =================================================================== */
    /* Empty graph                                                          */
    /* =================================================================== */
    printf("=== Empty graph ===\n");
    igraph_small(&g, 0, 0, -1);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical(&g, &coords, NULL, 10, 0.1);
    print_matrix(&coords);
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    /* =================================================================== */
    /* 2-node graph                                                        */
    /* =================================================================== */
    printf("=== 2-node graph ===\n");
    igraph_small(&g, 2, IGRAPH_UNDIRECTED, 0, 1, -1);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical(&g, &coords, NULL, 10, 0.1);
    check_on_sphere(&coords, 2, "direct-2node");
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    igraph_small(&g, 2, IGRAPH_UNDIRECTED, 0, 1, -1);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical_interpolation(&g, &coords, NULL, 10, 0.1, 2);
    check_on_sphere(&coords, 2, "interp-2node");
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    /* =================================================================== */
    /* Tree (n=10) - both methods                                           */
    /* =================================================================== */
    printf("=== Tree(10) ===\n");
    igraph_kary_tree(&g, 10, 2, IGRAPH_TREE_UNDIRECTED);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical(&g, &coords, NULL, 100, 0.1);
    check_on_sphere(&coords, 10, "direct-tree10");
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    igraph_kary_tree(&g, 10, 2, IGRAPH_TREE_UNDIRECTED);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical_interpolation(&g, &coords, NULL, 100, 0.1, 5);
    check_on_sphere(&coords, 10, "interp-tree10");
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    /* =================================================================== */
    /* Ring(10) - both methods                                              */
    /* =================================================================== */
    printf("=== Ring(10) ===\n");
    igraph_ring(&g, 10, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical(&g, &coords, NULL, 100, 0.1);
    check_on_sphere(&coords, 10, "direct-ring10");
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    igraph_ring(&g, 10, IGRAPH_UNDIRECTED, 0, 1);
    igraph_matrix_init(&coords, 0, 0);
    igraph_layout_mds_spherical_interpolation(&g, &coords, NULL, 100, 0.1, 5);
    check_on_sphere(&coords, 10, "interp-ring10");
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    /* =================================================================== */
    /* Full(6) with precomputed distance matrix                              */
    /* =================================================================== */
    printf("=== Full(6) with precomputed dist ===\n");
    igraph_full(&g, 6, IGRAPH_UNDIRECTED, 0);
    igraph_matrix_init(&coords, 6, 2);
    igraph_matrix_init(&dist_mat, 6, 6);
    for (i = 0; i < 6; i++)
        for (j = 0; j < 2; j++) {
            MATRIX(coords, i, j) = RNG_INTEGER(0, 1000);
        }
    for (i = 0; i < 6; i++)
        for (j = i + 1; j < 6; j++) {
            double dist_sq = 0.0;
            dist_sq += sqr(MATRIX(coords, i, 0) - MATRIX(coords, j, 0));
            dist_sq += sqr(MATRIX(coords, i, 1) - MATRIX(coords, j, 1));
            MATRIX(dist_mat, i, j) = sqrt(dist_sq);
            MATRIX(dist_mat, j, i) = sqrt(dist_sq);
        }
    igraph_layout_mds_spherical(&g, &coords, &dist_mat, 200, 0.5);
    check_on_sphere(&coords, 6, "direct-full6-dist");
    igraph_matrix_destroy(&dist_mat);
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    igraph_full(&g, 6, IGRAPH_UNDIRECTED, 0);
    igraph_matrix_init(&coords, 6, 2);
    igraph_matrix_init(&dist_mat, 6, 6);
    for (i = 0; i < 6; i++)
        for (j = 0; j < 2; j++) {
            MATRIX(coords, i, j) = RNG_INTEGER(0, 1000);
        }
    for (i = 0; i < 6; i++)
        for (j = i + 1; j < 6; j++) {
            double dist_sq = 0.0;
            dist_sq += sqr(MATRIX(coords, i, 0) - MATRIX(coords, j, 0));
            dist_sq += sqr(MATRIX(coords, i, 1) - MATRIX(coords, j, 1));
            MATRIX(dist_mat, i, j) = sqrt(dist_sq);
            MATRIX(dist_mat, j, i) = sqrt(dist_sq);
        }
    igraph_layout_mds_spherical_interpolation(&g, &coords, &dist_mat, 200, 0.5, 3);
    check_on_sphere(&coords, 6, "interp-full6-dist");
    igraph_matrix_destroy(&dist_mat);
    igraph_matrix_destroy(&coords);
    igraph_destroy(&g);

    /* =================================================================== */
    /* Ring(50) - direct SGD vs interpolation stress comparison             */
    /* =================================================================== */
    printf("=== Ring(50) ===\n");
    {
        igraph_int_t n = 50;
        igraph_matrix_t d;
        igraph_ring(&g, n, IGRAPH_UNDIRECTED, 0, 1);
        igraph_matrix_init(&d, 0, 0);
        igraph_distances(&g, NULL, &d, igraph_vss_all(), igraph_vss_all(), IGRAPH_ALL);

        igraph_real_t maxd = 0.0;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                if (MATRIX(d, i, j) > maxd) maxd = MATRIX(d, i, j);
        if (maxd > 0.0) {
            igraph_real_t s = M_PI / maxd;
            igraph_matrix_scale(&d, s);
        }

        igraph_matrix_init(&coords, 0, 0);
        igraph_layout_mds_spherical(&g, &coords, NULL, 50, 0.1);
        check_on_sphere(&coords, n, "direct-ring50");
        compute_stress(&coords, &d, n, "stress-direct-ring50");
        igraph_matrix_destroy(&coords);
        igraph_destroy(&g);

        igraph_ring(&g, n, IGRAPH_UNDIRECTED, 0, 1);
        igraph_matrix_init(&coords, 0, 0);
        igraph_layout_mds_spherical_interpolation(&g, &coords, NULL, 50, 0.1, 10);
        check_on_sphere(&coords, n, "interp-ring50");
        compute_stress(&coords, &d, n, "stress-interp-ring50");
        igraph_matrix_destroy(&coords);
        igraph_destroy(&g);

        igraph_matrix_destroy(&d);
    }

    /* =================================================================== */
    /* Precomputed dist on ring(150)                                        */
    /* =================================================================== */
    printf("=== Ring(150) with precomputed dist ===\n");
    {
        igraph_int_t n = 150;
        igraph_ring(&g, n, IGRAPH_UNDIRECTED, 0, 1);
        igraph_matrix_init(&coords, 0, 0);
        igraph_matrix_init(&dist_mat, 0, 0);
        igraph_distances(&g, NULL, &dist_mat, igraph_vss_all(), igraph_vss_all(), IGRAPH_ALL);
        igraph_layout_mds_spherical_interpolation(&g, &coords, &dist_mat, 30, 0.1, 20);
        if (igraph_matrix_nrow(&coords) != n || igraph_matrix_ncol(&coords) != 3) {
            printf("FAIL: expected %lldx3, got %lldx%lld\n",
                   (long long)n,
                   (long long)igraph_matrix_nrow(&coords),
                   (long long)igraph_matrix_ncol(&coords));
            return 1;
        }
        check_on_sphere(&coords, n, "interp-ring150-dist");
        igraph_matrix_destroy(&dist_mat);
        igraph_matrix_destroy(&coords);
        igraph_destroy(&g);
    }

    /* =================================================================== */
    /* Direct SGD on ring(600) - tests that it works on any size graph      */
    /* =================================================================== */
    printf("=== Direct SGD on ring(600) ===\n");
    {
        igraph_int_t n = 600;
        igraph_ring(&g, n, IGRAPH_UNDIRECTED, 0, 1);
        igraph_matrix_init(&coords, 0, 0);
        igraph_layout_mds_spherical(&g, &coords, NULL, 20, 0.1);
        if (igraph_matrix_nrow(&coords) != n || igraph_matrix_ncol(&coords) != 3) {
            printf("FAIL: expected %lldx3, got %lldx%lld\n",
                   (long long)n,
                   (long long)igraph_matrix_nrow(&coords),
                   (long long)igraph_matrix_ncol(&coords));
            return 1;
        }
        check_on_sphere(&coords, n, "direct-ring600");
        igraph_matrix_destroy(&coords);
        igraph_destroy(&g);
    }

    VERIFY_FINALLY_STACK();

    printf("All tests passed.\n");
    return 0;
}
