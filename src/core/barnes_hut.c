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

#include "barnes_hut.h"
#include "igraph_memory.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IGRAPH_BH_DEFAULT_THETA 0.6
#define IGRAPH_BH_DEFAULT_MAX_LEVEL 10
#define IGRAPH_BH_CHILDREN_COUNT(dim) (1 << (dim))

static igraph_integer_t build_bh_tree_recursive(
    igraph_bh_tree_t *tree,
    igraph_bh_node_t *nodes,
    igraph_integer_t *indices,
    igraph_integer_t count,
    double center[3],
    double size,
    int dim,
    int level,
    int max_level
);

static void calculate_force_node_to_tree(
    const igraph_bh_tree_t *tree,
    const igraph_bh_node_t *node,
    const igraph_bh_point_t *point,
    double *force,
    igraph_bh_force_func_t force_func,
    void *user_data
);

igraph_error_t igraph_bh_tree_init(igraph_bh_tree_t *tree, int dim, double theta, int max_level) {
    tree->points = NULL;
    tree->nodes = NULL;
    tree->root = NULL;
    tree->node_count = 0;
    tree->point_count = 0;
    tree->capacity = 0;

    igraph_bh_tree_destroy(tree);

    tree->dim = dim;
    tree->bh_theta = theta > 0 ? theta : IGRAPH_BH_DEFAULT_THETA;
    tree->max_level = max_level > 0 ? max_level : IGRAPH_BH_DEFAULT_MAX_LEVEL;
    tree->point_count = 0;
    tree->points = NULL;
    tree->root = NULL;
    tree->capacity = 0;
    tree->node_count = 0;
    tree->nodes = NULL;
    return IGRAPH_SUCCESS;
}

void igraph_bh_tree_destroy(igraph_bh_tree_t *tree) {
    if (tree->points) {
        IGRAPH_FREE(tree->points);
        tree->points = NULL;
    }
    if (tree->nodes) {
        for (igraph_integer_t i = 0; i < tree->node_count; i++) {
            if (tree->nodes[i].point_ids) {
                IGRAPH_FREE(tree->nodes[i].point_ids);
            }
            if (tree->nodes[i].children) {
                IGRAPH_FREE(tree->nodes[i].children);
            }
        }
        IGRAPH_FREE(tree->nodes);
        tree->nodes = NULL;
    }
    tree->root = NULL;
    tree->point_count = 0;
    tree->node_count = 0;
}

static double compute_center_of_mass(
    igraph_bh_node_t *node,
    igraph_bh_point_t *points,
    igraph_integer_t *indices,
    igraph_integer_t start,
    igraph_integer_t end,
    int dim
) {
    double total_mass = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;

    for (igraph_integer_t i = start; i < end; i++) {
        igraph_bh_point_t *p = &points[indices[i]];
        double m = p->mass > 0 ? p->mass : 1.0;
        total_mass += m;
        cx += p->coord[0] * m;
        cy += p->coord[1] * m;
        if (dim == 3) {
            cz += p->coord[2] * m;
        }
    }

    if (total_mass > 0) {
        cx /= total_mass;
        cy /= total_mass;
        node->center[0] = cx;
        node->center[1] = cy;
        if (dim == 3) {
            node->center[2] = cz / total_mass;
        } else {
            node->center[2] = 0.0;
        }
        node->mass = total_mass;
    } else {
        node->center[0] = 0;
        node->center[1] = 0;
        node->center[2] = 0;
        node->mass = 0;
    }

    double max_dist = 0;
    for (igraph_integer_t i = start; i < end; i++) {
        igraph_bh_point_t *p = &points[indices[i]];
        double dx = p->coord[0] - cx;
        double dy = p->coord[1] - cy;
        double dz = (dim == 3) ? (p->coord[2] - node->center[2]) : 0.0;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    return 2.0 * max_dist;
}

static igraph_integer_t build_bh_tree_recursive(
    igraph_bh_tree_t *tree,
    igraph_bh_node_t *nodes,
    igraph_integer_t *indices,
    igraph_integer_t count,
    double center[3],
    double size,
    int dim,
    int level,
    int max_level
) {
    igraph_integer_t node_idx = tree->node_count++;
    igraph_bh_node_t *node = &nodes[node_idx];

    node->dim = dim;
    node->size = size;
    node->point_ids = NULL;
    node->children = NULL;

    if (count == 1 || level >= max_level) {
        node->is_leaf = 1;
        node->point_count = count;
        node->point_ids = IGRAPH_CALLOC(count, igraph_integer_t);
        if (!node->point_ids) {
            return -1;
        }
        for (igraph_integer_t i = 0; i < count; i++) {
            node->point_ids[i] = indices[i];
        }

        node->mass = 0;
        node->center[0] = 0;
        node->center[1] = 0;
        node->center[2] = 0;
        for (igraph_integer_t i = 0; i < count; i++) {
            igraph_bh_point_t *p = &tree->points[indices[i]];
            node->mass += p->mass;
            node->center[0] += p->coord[0] * p->mass;
            node->center[1] += p->coord[1] * p->mass;
            if (dim == 3) {
                node->center[2] += p->coord[2] * p->mass;
            }
        }
        if (node->mass > 0) {
            node->center[0] /= node->mass;
            node->center[1] /= node->mass;
            if (dim == 3) {
                node->center[2] /= node->mass;
            }
        }

        return node_idx;
    }

    node->is_leaf = 0;
    node->point_count = 0;

    double total_mass = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (igraph_integer_t i = 0; i < count; i++) {
        igraph_bh_point_t *p = &tree->points[indices[i]];
        double m = p->mass > 0 ? p->mass : 1.0;
        total_mass += m;
        cx += p->coord[0] * m;
        cy += p->coord[1] * m;
        if (dim == 3) {
            cz += p->coord[2] * m;
        }
    }
    if (total_mass > 0) {
        node->center[0] = cx / total_mass;
        node->center[1] = cy / total_mass;
        if (dim == 3) {
            node->center[2] = cz / total_mass;
        } else {
            node->center[2] = 0.0;
        }
        node->mass = total_mass;
    } else {
        node->center[0] = center[0];
        node->center[1] = center[1];
        node->center[2] = center[2];
        node->mass = 0;
    }

    int n_children = IGRAPH_BH_CHILDREN_COUNT(dim);
    node->children = IGRAPH_CALLOC(n_children, igraph_bh_node_t*);
    if (!node->children) {
        return -1;
    }

    igraph_integer_t *counts = IGRAPH_CALLOC(n_children, igraph_integer_t);
    if (!counts) {
        IGRAPH_FREE(node->children);
        node->children = NULL;
        return -1;
    }

    for (igraph_integer_t i = 0; i < count; i++) {
        int child_idx = 0;
        igraph_bh_point_t *p = &tree->points[indices[i]];
        if (p->coord[0] >= node->center[0]) child_idx |= 1;
        if (p->coord[1] >= node->center[1]) child_idx |= 2;
        if (dim == 3 && p->coord[2] >= node->center[2]) child_idx |= 4;
        counts[child_idx]++;
    }

    igraph_integer_t *offsets = IGRAPH_CALLOC(n_children + 1, igraph_integer_t);
    if (!offsets) {
        IGRAPH_FREE(counts);
        IGRAPH_FREE(node->children);
        node->children = NULL;
        return -1;
    }

    for (int i = 1; i <= n_children; i++) {
        offsets[i] = offsets[i-1] + counts[i-1];
    }

    igraph_integer_t *temp = IGRAPH_MALLOC(count * sizeof(igraph_integer_t));
    if (!temp) {
        IGRAPH_FREE(offsets);
        IGRAPH_FREE(counts);
        IGRAPH_FREE(node->children);
        node->children = NULL;
        return -1;
    }

    igraph_integer_t *cursors = IGRAPH_CALLOC(n_children, igraph_integer_t);
    if (!cursors) {
        IGRAPH_FREE(temp);
        IGRAPH_FREE(offsets);
        IGRAPH_FREE(counts);
        IGRAPH_FREE(node->children);
        node->children = NULL;
        return -1;
    }
    for (int i = 0; i < n_children; i++) {
        cursors[i] = offsets[i];
    }

    for (igraph_integer_t i = 0; i < count; i++) {
        int child_idx = 0;
        igraph_bh_point_t *p = &tree->points[indices[i]];
        if (p->coord[0] >= node->center[0]) child_idx |= 1;
        if (p->coord[1] >= node->center[1]) child_idx |= 2;
        if (dim == 3 && p->coord[2] >= node->center[2]) child_idx |= 4;
        temp[cursors[child_idx]++] = indices[i];
    }

    for (igraph_integer_t i = 0; i < count; i++) {
        indices[i] = temp[i];
    }

    double half_size = size / 2.0;
    for (int i = 0; i < n_children; i++) {
        if (counts[i] > 0) {
            double child_center[3];
            child_center[0] = node->center[0] + ((i & 1) ? half_size : -half_size);
            child_center[1] = node->center[1] + ((i & 2) ? half_size : -half_size);
            child_center[2] = (dim == 3) ? (node->center[2] + ((i & 4) ? half_size : -half_size)) : 0.0;

            igraph_integer_t child_node_idx = build_bh_tree_recursive(
                tree, nodes, indices + offsets[i],
                counts[i],
                child_center, half_size, dim, level + 1, max_level
            );
            node->children[i] = &nodes[child_node_idx];
        } else {
            node->children[i] = NULL;
        }
    }

    IGRAPH_FREE(cursors);
    IGRAPH_FREE(offsets);
    IGRAPH_FREE(counts);
    IGRAPH_FREE(temp);

    return node_idx;
}

igraph_error_t igraph_bh_tree_build(
    igraph_bh_tree_t *tree,
    const igraph_matrix_t *coords,
    const igraph_vector_t *masses,
    int dim,
    int max_level,
    double bh_theta
) {
    igraph_bh_tree_destroy(tree);

    igraph_integer_t n = igraph_matrix_nrow(coords);

    if (n == 0) {
        tree->point_count = 0;
        tree->root = NULL;
        return IGRAPH_SUCCESS;
    }

    tree->dim = dim;
    tree->bh_theta = bh_theta > 0 ? bh_theta : IGRAPH_BH_DEFAULT_THETA;
    tree->max_level = max_level > 0 ? max_level : IGRAPH_BH_DEFAULT_MAX_LEVEL;

    tree->point_count = n;
    tree->points = IGRAPH_CALLOC(n, igraph_bh_point_t);
    if (!tree->points) {
        return IGRAPH_ENOMEM;
    }

    for (igraph_integer_t i = 0; i < n; i++) {
        tree->points[i].id = i;
        tree->points[i].coord[0] = MATRIX(*coords, i, 0);
        tree->points[i].coord[1] = MATRIX(*coords, i, 1);
        tree->points[i].coord[2] = (dim == 3) ? MATRIX(*coords, i, 2) : 0.0;
        tree->points[i].mass = masses ? VECTOR(*masses)[i] : 1.0;
        tree->points[i].data = NULL;
    }

    tree->capacity = n * IGRAPH_BH_CHILDREN_COUNT(dim);
    tree->nodes = IGRAPH_CALLOC(tree->capacity, igraph_bh_node_t);
    if (!tree->nodes) {
        IGRAPH_FREE(tree->points);
        tree->points = NULL;
        return IGRAPH_ENOMEM;
    }

    igraph_integer_t *indices = IGRAPH_MALLOC(n * sizeof(igraph_integer_t));
    igraph_integer_t *temp = IGRAPH_MALLOC(n * sizeof(igraph_integer_t));
    for (igraph_integer_t i = 0; i < n; i++) {
        indices[i] = i;
    }

    double min_x = tree->points[0].coord[0];
    double max_x = tree->points[0].coord[0];
    double min_y = tree->points[0].coord[1];
    double max_y = tree->points[0].coord[1];
    double min_z = (dim == 3) ? tree->points[0].coord[2] : 0.0;
    double max_z = min_z;

    for (igraph_integer_t i = 1; i < n; i++) {
        if (tree->points[i].coord[0] < min_x) min_x = tree->points[i].coord[0];
        if (tree->points[i].coord[0] > max_x) max_x = tree->points[i].coord[0];
        if (tree->points[i].coord[1] < min_y) min_y = tree->points[i].coord[1];
        if (tree->points[i].coord[1] > max_y) max_y = tree->points[i].coord[1];
        if (dim == 3) {
            if (tree->points[i].coord[2] < min_z) min_z = tree->points[i].coord[2];
            if (tree->points[i].coord[2] > max_z) max_z = tree->points[i].coord[2];
        }
    }

    double cx = (min_x + max_x) / 2.0;
    double cy = (min_y + max_y) / 2.0;
    double cz = (dim == 3) ? ((min_z + max_z) / 2.0) : 0.0;
    double size = max_x - min_x;
    if (max_y - min_y > size) size = max_y - min_y;
    if (dim == 3 && max_z - min_z > size) size = max_z - min_z;
    size *= 1.01;

    tree->node_count = 0;
    igraph_integer_t root_idx = build_bh_tree_recursive(
        tree, tree->nodes, indices, n,
        (double[3]){cx, cy, cz},
        size, dim, 0, tree->max_level
    );

    IGRAPH_FREE(indices);
    IGRAPH_FREE(temp);

    if (root_idx < 0) {
        IGRAPH_FREE(tree->points);
        IGRAPH_FREE(tree->nodes);
        tree->points = NULL;
        tree->nodes = NULL;
        return IGRAPH_ENOMEM;
    }

    tree->root = &tree->nodes[root_idx];

    return IGRAPH_SUCCESS;
}

static void calculate_force_node_to_tree(
    const igraph_bh_tree_t *tree,
    const igraph_bh_node_t *node,
    const igraph_bh_point_t *point,
    double *force,
    igraph_bh_force_func_t force_func,
    void *user_data
) {
    if (!node) return;

    double dx = point->coord[0] - node->center[0];
    double dy = point->coord[1] - node->center[1];
    double dz = (tree->dim == 3) ? (point->coord[2] - node->center[2]) : 0.0;
    double dist_sq = dx*dx + dy*dy + dz*dz;
    double dist = sqrt(dist_sq);

    if (node->is_leaf) {
        for (igraph_integer_t i = 0; i < node->point_count; i++) {
            igraph_bh_point_t *other = &tree->points[node->point_ids[i]];
            if (other->id != point->id) {
                double f[3] = {0, 0, 0};
                force_func(point, other, f, user_data);
                force[0] += f[0];
                force[1] += f[1];
                if (tree->dim == 3) {
                    force[2] += f[2];
                }
            }
        }
    } else {
        if (dist > 0 && tree->bh_theta * node->size < dist) {
            igraph_bh_point_t pseudo_point;
            pseudo_point.id = -1;
            pseudo_point.mass = node->mass;
            pseudo_point.coord[0] = node->center[0];
            pseudo_point.coord[1] = node->center[1];
            pseudo_point.coord[2] = node->center[2];
            pseudo_point.data = NULL;

            double f[3] = {0, 0, 0};
            force_func(point, &pseudo_point, f, user_data);
            force[0] += f[0];
            force[1] += f[1];
            if (tree->dim == 3) {
                force[2] += f[2];
            }
        } else {
            int n_children = IGRAPH_BH_CHILDREN_COUNT(tree->dim);
            for (int i = 0; i < n_children; i++) {
                if (node->children[i]) {
                    calculate_force_node_to_tree(tree, node->children[i], point, force, force_func, user_data);
                }
            }
        }
    }
}

igraph_error_t igraph_bh_calculate_repulsive_forces(
    const igraph_bh_tree_t *tree,
    igraph_matrix_t *forces,
    igraph_bh_force_func_t force_func,
    void *user_data
) {
    if (!tree || tree->point_count == 0) {
        return IGRAPH_SUCCESS;
    }

    igraph_integer_t n = tree->point_count;

    for (igraph_integer_t i = 0; i < n; i++) {
        double force[3] = {0, 0, 0};
        calculate_force_node_to_tree(tree, tree->root, &tree->points[i], force, force_func, user_data);
        MATRIX(*forces, i, 0) = force[0];
        MATRIX(*forces, i, 1) = force[1];
        if (tree->dim == 3) {
            MATRIX(*forces, i, 2) = force[2];
        }
    }

    return IGRAPH_SUCCESS;
}

igraph_error_t igraph_bh_calculate_attractive_forces(
    const igraph_bh_tree_t *tree,
    const igraph_vector_int_t *from,
    const igraph_vector_int_t *to,
    const igraph_vector_t *weights,
    igraph_matrix_t *forces,
    igraph_bh_force_func_t force_func,
    void *user_data
) {
    if (!from || !to || igraph_vector_int_size(from) == 0) {
        return IGRAPH_SUCCESS;
    }

    igraph_integer_t n_edges = igraph_vector_int_size(from);

    for (igraph_integer_t e = 0; e < n_edges; e++) {
        igraph_integer_t from_idx = VECTOR(*from)[e];
        igraph_integer_t to_idx = VECTOR(*to)[e];
        double w = weights ? VECTOR(*weights)[e] : 1.0;

        igraph_bh_point_t *p1 = &tree->points[from_idx];
        igraph_bh_point_t *p2 = &tree->points[to_idx];

        double f[3] = {0, 0, 0};
        force_func(p1, p2, f, user_data);

        f[0] *= w;
        f[1] *= w;
        if (tree->dim == 3) {
            f[2] *= w;
        }

        MATRIX(*forces, from_idx, 0) -= f[0];
        MATRIX(*forces, from_idx, 1) -= f[1];
        if (tree->dim == 3) {
            MATRIX(*forces, from_idx, 2) -= f[2];
        }

        MATRIX(*forces, to_idx, 0) += f[0];
        MATRIX(*forces, to_idx, 1) += f[1];
        if (tree->dim == 3) {
            MATRIX(*forces, to_idx, 2) += f[2];
        }
    }

    return IGRAPH_SUCCESS;
}

void igraph_bh_tree_print_stats(const igraph_bh_tree_t *tree) {
    printf("Barnes-Hut Tree Stats:\n");
    printf("  Points: %d\n", (int)tree->point_count);
    printf("  Nodes: %d\n", (int)tree->node_count);
    printf("  Dimension: %d\n", tree->dim);
    printf("  Theta: %f\n", tree->bh_theta);
    printf("  Max level: %d\n", tree->max_level);
}