/*
   igraph library.
   Copyright (C) 2024  The igraph development team <igraph@igraph.org>

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

#ifndef IGRAPH_BARNES_HUT_H
#define IGRAPH_BARNES_HUT_H

#include "igraph_decls.h"
#include "igraph_types.h"
#include "igraph_matrix.h"
#include "igraph_vector.h"

IGRAPH_BEGIN_C_DECLS

typedef struct igraph_bh_point_t {
    double coord[3];
    double mass;
    int id;
    void *data;
} igraph_bh_point_t;

typedef struct igraph_bh_node_t {
    double mass;
    double center[3];
    double size;
    int dim;
    int is_leaf;
    int point_count;
    igraph_integer_t *point_ids;
    struct igraph_bh_node_t **children;
} igraph_bh_node_t;

typedef struct {
    igraph_bh_node_t *root;
    int dim;
    int max_level;
    int point_count;
    igraph_bh_point_t *points;
    double bh_theta;
    igraph_integer_t capacity;
    igraph_integer_t node_count;
    igraph_bh_node_t *nodes;
} igraph_bh_tree_t;

typedef void (*igraph_bh_force_func_t)(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    double *force,
    void *user_data
);

igraph_error_t igraph_bh_tree_init(igraph_bh_tree_t *tree, int dim, double theta, int max_level);

void igraph_bh_tree_destroy(igraph_bh_tree_t *tree);

igraph_error_t igraph_bh_tree_build(
    igraph_bh_tree_t *tree,
    const igraph_matrix_t *coords,
    const igraph_vector_t *masses,
    int dim,
    int max_level,
    double bh_theta
);

igraph_error_t igraph_bh_calculate_repulsive_forces(
    const igraph_bh_tree_t *tree,
    igraph_matrix_t *forces,
    igraph_bh_force_func_t force_func,
    void *user_data
);

igraph_error_t igraph_bh_calculate_attractive_forces(
    const igraph_bh_tree_t *tree,
    const igraph_vector_int_t *from,
    const igraph_vector_int_t *to,
    const igraph_vector_t *weights,
    igraph_matrix_t *forces,
    igraph_bh_force_func_t force_func,
    void *user_data
);

void igraph_bh_tree_print_stats(const igraph_bh_tree_t *tree);

IGRAPH_END_C_DECLS

#endif