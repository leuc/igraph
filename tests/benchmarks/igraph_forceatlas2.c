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
#include "bench.h"

#define TOSTR1(x) #x
#define TOSTR(x) TOSTR1(x)

int main(void) {
    igraph_t graph;
    igraph_vector_t weights;
    igraph_matrix_t layout;

    igraph_rng_seed(igraph_rng_default(), 42);
    BENCH_INIT();

    igraph_matrix_init(&layout, 0, 0);

    igraph_small(&graph, 12, IGRAPH_UNDIRECTED,
            0,1, 0,2, 0,3, 1,2, 1,3, 2,3,
            3,4, 4,5, 5,6,
            6,7, 7,8, 6,8, 7,9, 6,9, 8,9, 7,10, 8,10, 9,10, 10,11, 9,11, 8,11, 7,11,
            -1);

    igraph_vector_init_real(&weights,
            igraph_ecount(&graph),
            0.1, 0.09, 0.12, 0.09, 0.1, 0.1,
            0.9, 0.9, 0.9,
            0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.05, 0.1, 0.08, 0.12, 0.09, 0.11
            );

#define ITERS 500
#define REP 400

    /* Path 1: Base 2D exact calculation */
    BENCH("Small graph (2D, Exact O(N^2), Normal Physics)", \
        REPEAT(igraph_layout_forceatlas2(&graph, &layout, ITERS, 0, 1.0, 1.0, 0, 1.2, 2.0, 0, 1.0, &weights), REP);
    );

    /* Path 2: Base 3D exact calculation (Tests 3D vectorization) */
    BENCH("Small graph (3D, Exact O(N^2), Normal Physics)", \
        REPEAT(igraph_layout_forceatlas2_3d(&graph, &layout, ITERS, 0, 1.0, 1.0, 0, 1.2, 2.0, 0, 1.0, &weights), REP);
    );

    /* Path 3: Barnes-Hut 2D (QuadTree logic) */
    BENCH("Small graph (2D, Barnes-Hut, Normal Physics)", \
        REPEAT(igraph_layout_forceatlas2(&graph, &layout, ITERS, 0, 1.0, 1.0, 1, 1.2, 2.0, 0, 1.0, &weights), REP);
    );

    /* Path 4: Barnes-Hut 3D (OctTree logic) */
    BENCH("Small graph (3D, Barnes-Hut, Normal Physics)", \
        REPEAT(igraph_layout_forceatlas2_3d(&graph, &layout, ITERS, 0, 1.0, 1.0, 1, 1.2, 2.0, 0, 1.0, &weights), REP);
    );

    /* Path 5: Alternative Physics Modes (Strong Gravity & Outbound Distribution) */
    BENCH("Small graph (2D, Exact, Strong Gravity + Outbound Dist)", \
        REPEAT(igraph_layout_forceatlas2(&graph, &layout, ITERS, 1, 1.0, 1.0, 0, 1.2, 2.0, 1, 1.0, &weights), REP);
    );

    /* Path 6: Null weights (Unweighted edge path) */
    BENCH("Small graph (2D, Exact, Unweighted edges)", \
        REPEAT(igraph_layout_forceatlas2(&graph, &layout, ITERS, 0, 1.0, 1.0, 0, 1.2, 2.0, 0, 1.0, NULL), REP);
    );

#undef ITERS
#undef REP
#define ITERS 100
#define REP 3
#define VCOUNT 10000
#define DENS 0.001

    igraph_destroy(&graph);
    igraph_erdos_renyi_game_gnp(&graph, VCOUNT, DENS, IGRAPH_UNDIRECTED, IGRAPH_SIMPLE_SW, IGRAPH_EDGE_UNLABELED);

    igraph_vector_resize(&weights, igraph_ecount(&graph));
    for (igraph_int_t i=0; i < igraph_ecount(&graph); i++) {
        VECTOR(weights)[i] = RNG_UNIF(0.05, 0.15);
    }

    /* Path 7: Large Graph Barnes-Hut 2D */
    BENCH("Large ER graph 10k nodes (2D, Barnes-Hut ON, 60 iters)", \
        REPEAT(igraph_layout_forceatlas2(&graph, &layout, ITERS, 0, 1.0, 1.0, 1, 1.2, 2.0, 0, 1.0, &weights) , REP);
    );

    /* Path 8: Large Graph Barnes-Hut 3D */
    BENCH("Large ER graph 10k nodes (3D, Barnes-Hut ON, 60 iters)", \
        REPEAT(igraph_layout_forceatlas2_3d(&graph, &layout, ITERS, 0, 1.0, 1.0, 1, 1.2, 2.0, 0, 1.0, &weights) , REP);
    );

    /* Path 9: Large Graph Alternative Physics (Barnes-Hut, Strong Grav, Outbound) */
    BENCH("Large ER graph 10k nodes (2D, Barnes-Hut ON, Strong Grav)", \
        REPEAT(igraph_layout_forceatlas2(&graph, &layout, ITERS, 1, 1.0, 1.0, 1, 1.2, 2.0, 1, 1.0, &weights) , REP);
    );

    igraph_matrix_destroy(&layout);
    igraph_destroy(&graph);
    igraph_vector_destroy(&weights);

    return 0;
}
