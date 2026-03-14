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

#include "igraph_layout.h"

#define _USE_MATH_DEFINES

#include "igraph_random.h"
#include "igraph_interface.h"

#include "core/barnes_hut.h"
#include "core/interruption.h"
#include "layout/layout_internal.h"

#define IGRAPH_YHU_C 0.2
#define IGRAPH_YHU_BH 0.6
#define IGRAPH_YHU_TOL 0.001
#define IGRAPH_YHU_COOL 0.90
#define IGRAPH_YHU_QUADTREE_SIZE 45

typedef struct {
    double p;
    double KP;
    double CRK;
    double K;
    int dim;
    const igraph_t *graph;
    const igraph_vector_t *weights;
} yhu_data_t;

static void yhu_repulsive_force(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    double *force,
    void *user_data
) {
    yhu_data_t *data = (yhu_data_t *)user_data;

    if (p1->id == p2->id) {
        return;
    }

    double dx = p1->coord[0] - p2->coord[0];
    double dy = p1->coord[1] - p2->coord[1];
    double dist_sq = dx*dx + dy*dy;

    if (dist_sq < 1e-12) {
        return;
    }

    double dist = sqrt(dist_sq);
    double exp_factor = 1.0 - data->p;

    if (exp_factor < 0) {
        exp_factor = -exp_factor;
    }

    double scale = data->KP / pow(dist, exp_factor);

    if (!isfinite(scale)) {
        scale = 1e10;
    }

    force[0] = scale * dx / dist;
    force[1] = scale * dy / dist;
}

static void yhu_attractive_force(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    double *force,
    void *user_data
) {
    yhu_data_t *data = (yhu_data_t *)user_data;

    double dx = p1->coord[0] - p2->coord[0];
    double dy = p1->coord[1] - p2->coord[1];
    double dist = sqrt(dx*dx + dy*dy);

    if (dist < 1e-12) {
        return;
    }

    double scale = -data->CRK * dist;

    force[0] = scale * dx;
    force[1] = scale * dy;
}

static void yhu_repulsive_force_3d(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    double *force,
    void *user_data
) {
    yhu_data_t *data = (yhu_data_t *)user_data;

    if (p1->id == p2->id) {
        return;
    }

    double dx = p1->coord[0] - p2->coord[0];
    double dy = p1->coord[1] - p2->coord[1];
    double dz = p1->coord[2] - p2->coord[2];
    double dist_sq = dx*dx + dy*dy + dz*dz;

    if (dist_sq < 1e-12) {
        return;
    }

    double dist = sqrt(dist_sq);
    double exp_factor = 1.0 - data->p;

    if (exp_factor < 0) {
        exp_factor = -exp_factor;
    }

    double scale = data->KP / pow(dist, exp_factor);

    if (!isfinite(scale)) {
        scale = 1e10;
    }

    force[0] = scale * dx / dist;
    force[1] = scale * dy / dist;
    force[2] = scale * dz / dist;
}

static void yhu_attractive_force_3d(
    const igraph_bh_point_t *p1,
    const igraph_bh_point_t *p2,
    double *force,
    void *user_data
) {
    yhu_data_t *data = (yhu_data_t *)user_data;

    double dx = p1->coord[0] - p2->coord[0];
    double dy = p1->coord[1] - p2->coord[1];
    double dz = p1->coord[2] - p2->coord[2];
    double dist = sqrt(dx*dx + dy*dy + dz*dz);

    if (dist < 1e-12) {
        return;
    }

    double scale = -data->CRK * dist;

    force[0] = scale * dx;
    force[1] = scale * dy;
    force[2] = scale * dz;
}

static double update_step(igraph_bool_t adaptive_cooling, double step, double Fnorm, double Fnorm0) {
    if (!adaptive_cooling) {
        return IGRAPH_YHU_COOL * step;
    }
    if (Fnorm >= Fnorm0) {
        return IGRAPH_YHU_COOL * step;
    } else if (Fnorm > 0.95 * Fnorm0) {
        return step;
    } else {
        return 0.99 * step / IGRAPH_YHU_COOL;
    }
}

static igraph_error_t compute_average_edge_length(
    const igraph_t *graph,
    const igraph_matrix_t *coords,
    const igraph_vector_t *weights,
    double *avg_len
) {
    igraph_integer_t ecount = igraph_ecount(graph);
    if (ecount == 0) {
        *avg_len = 1.0;
        return IGRAPH_SUCCESS;
    }

    double total_len = 0.0;
    for (igraph_integer_t e = 0; e < ecount; e++) {
        igraph_integer_t from = IGRAPH_FROM(graph, e);
        igraph_integer_t to = IGRAPH_TO(graph, e);
        double dx = MATRIX(*coords, from, 0) - MATRIX(*coords, to, 0);
        double dy = MATRIX(*coords, from, 1) - MATRIX(*coords, to, 1);
        double w = weights ? VECTOR(*weights)[e] : 1.0;
        total_len += sqrt(dx*dx + dy*dy) * w;
    }

    *avg_len = total_len / ecount;
    return IGRAPH_SUCCESS;
}

static igraph_error_t beautify_leaves(
    const igraph_t *graph,
    igraph_matrix_t *coords
) {
    igraph_integer_t vcount = igraph_vcount(graph);
    igraph_integer_t ecount = igraph_ecount(graph);

    if (ecount == 0 || vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    igraph_vector_int_t degrees;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&degrees, vcount);

    IGRAPH_CHECK(igraph_degree(graph, &degrees, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS));

    for (igraph_integer_t i = 0; i < vcount; i++) {
        if (VECTOR(degrees)[i] == 1) {
            igraph_integer_t parent = -1;
            for (igraph_integer_t e = 0; e < ecount; e++) {
                igraph_integer_t from = IGRAPH_FROM(graph, e);
                igraph_integer_t to = IGRAPH_TO(graph, e);
                if (from == i) {
                    parent = to;
                    break;
                } else if (to == i) {
                    parent = from;
                    break;
                }
            }

            if (parent >= 0) {
                double px = MATRIX(*coords, parent, 0);
                double py = MATRIX(*coords, parent, 1);

                int leaf_count = 0;
                for (igraph_integer_t j = 0; j < vcount; j++) {
                    if (j != parent && VECTOR(degrees)[j] == 1) {
                        for (igraph_integer_t e = 0; e < ecount; e++) {
                            igraph_integer_t from = IGRAPH_FROM(graph, e);
                            igraph_integer_t to = IGRAPH_TO(graph, e);
                            if ((from == j && to == parent) || (to == j && from == parent)) {
                                leaf_count++;
                                break;
                            }
                        }
                    }
                }

                if (leaf_count > 0) {
                    double angle = 2.0 * 3.14159265358979323846 * (double)i / (double)vcount;
                    double radius = 5.0;
                    MATRIX(*coords, i, 0) = px + radius * cos(angle);
                    MATRIX(*coords, i, 1) = py + radius * sin(angle);
                }
            }
        }
    }

    igraph_vector_int_destroy(&degrees);
    IGRAPH_FINALLY_CLEAN(1);

    return IGRAPH_SUCCESS;
}

static igraph_error_t compute_average_edge_length_3d(
    const igraph_t *graph,
    const igraph_matrix_t *coords,
    const igraph_vector_t *weights,
    double *avg_len
) {
    igraph_integer_t ecount = igraph_ecount(graph);
    if (ecount == 0) {
        *avg_len = 1.0;
        return IGRAPH_SUCCESS;
    }

    double total_len = 0.0;
    for (igraph_integer_t e = 0; e < ecount; e++) {
        igraph_integer_t from = IGRAPH_FROM(graph, e);
        igraph_integer_t to = IGRAPH_TO(graph, e);
        double dx = MATRIX(*coords, from, 0) - MATRIX(*coords, to, 0);
        double dy = MATRIX(*coords, from, 1) - MATRIX(*coords, to, 1);
        double dz = MATRIX(*coords, from, 2) - MATRIX(*coords, to, 2);
        double w = weights ? VECTOR(*weights)[e] : 1.0;
        total_len += sqrt(dx*dx + dy*dy + dz*dz) * w;
    }

    *avg_len = total_len / ecount;
    return IGRAPH_SUCCESS;
}

static igraph_error_t beautify_leaves_3d(
    const igraph_t *graph,
    igraph_matrix_t *coords
) {
    igraph_integer_t vcount = igraph_vcount(graph);
    igraph_integer_t ecount = igraph_ecount(graph);

    if (ecount == 0 || vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    igraph_vector_int_t degrees;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&degrees, vcount);

    IGRAPH_CHECK(igraph_degree(graph, &degrees, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS));

    for (igraph_integer_t i = 0; i < vcount; i++) {
        if (VECTOR(degrees)[i] == 1) {
            igraph_integer_t parent = -1;
            for (igraph_integer_t e = 0; e < ecount; e++) {
                igraph_integer_t from = IGRAPH_FROM(graph, e);
                igraph_integer_t to = IGRAPH_TO(graph, e);
                if (from == i) {
                    parent = to;
                    break;
                } else if (to == i) {
                    parent = from;
                    break;
                }
            }

            if (parent >= 0) {
                double px = MATRIX(*coords, parent, 0);
                double py = MATRIX(*coords, parent, 1);
                double pz = MATRIX(*coords, parent, 2);

                int leaf_count = 0;
                for (igraph_integer_t j = 0; j < vcount; j++) {
                    if (j != parent && VECTOR(degrees)[j] == 1) {
                        for (igraph_integer_t e = 0; e < ecount; e++) {
                            igraph_integer_t from = IGRAPH_FROM(graph, e);
                            igraph_integer_t to = IGRAPH_TO(graph, e);
                            if ((from == j && to == parent) || (to == j && from == parent)) {
                                leaf_count++;
                                break;
                            }
                        }
                    }
                }

                if (leaf_count > 0) {
                    double theta = 2.0 * 3.14159265358979323846 * (double)i / (double)vcount;
                    double phi = 3.14159265358979323846 * (double)(i % 17) / 17.0;
                    double radius = 5.0;
                    MATRIX(*coords, i, 0) = px + radius * sin(phi) * cos(theta);
                    MATRIX(*coords, i, 1) = py + radius * sin(phi) * sin(theta);
                    MATRIX(*coords, i, 2) = pz + radius * cos(phi);
                }
            }
        }
    }

    igraph_vector_int_destroy(&degrees);
    IGRAPH_FINALLY_CLEAN(1);

    return IGRAPH_SUCCESS;
}

static igraph_error_t igraph_layout_i_yifan_hu_sfdp_3d(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t maxiter,
        igraph_real_t repulsive_exponent,
        igraph_real_t natural_length,
        igraph_real_t step,
        igraph_bool_t adaptive_cooling,
        igraph_real_t tolerance,
        igraph_quadtree_scheme_t quadtree_scheme,
        igraph_int_t max_qtree_level,
        igraph_bool_t beautify_leaves_flag,
        const igraph_vector_t *weights) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    if (!use_seed) {
        IGRAPH_CHECK(igraph_matrix_resize(res, vcount, 3));
        igraph_rng_seed(igraph_rng_default(), 42);
        for (igraph_integer_t i = 0; i < vcount; i++) {
            MATRIX(*res, i, 0) = RNG_UNIF(-1.0, 1.0);
            MATRIX(*res, i, 1) = RNG_UNIF(-1.0, 1.0);
            MATRIX(*res, i, 2) = RNG_UNIF(-1.0, 1.0);
        }
    }

    double K = natural_length;
    if (K < 0) {
        IGRAPH_CHECK(compute_average_edge_length_3d(graph, res, weights, &K));
    }

    double p = repulsive_exponent;
    if (p >= 0) {
        p = -1.0;
    }

    double KP = pow(K, 1.0 - p);
    double CRK = pow(IGRAPH_YHU_C, (2.0 - p) / 3.0) / K;

    yhu_data_t yhu_data = {
        .p = p,
        .KP = KP,
        .CRK = CRK,
        .K = K,
        .dim = 3,
        .graph = graph,
        .weights = weights
    };

    igraph_bh_tree_t tree;
    IGRAPH_CHECK(igraph_bh_tree_init(&tree, 3, IGRAPH_YHU_BH, (int)max_qtree_level, 1));
    IGRAPH_FINALLY(igraph_bh_tree_destroy, &tree);

    igraph_vector_int_t from, to;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&from, ecount);
    IGRAPH_VECTOR_INT_INIT_FINALLY(&to, ecount);

    for (igraph_integer_t e = 0; e < ecount; e++) {
        VECTOR(from)[e] = IGRAPH_FROM(graph, e);
        VECTOR(to)[e] = IGRAPH_TO(graph, e);
    }

    double Fnorm0 = IGRAPH_INFINITY;

    for (igraph_int_t iter = 0; iter < maxiter; iter++) {
        IGRAPH_ALLOW_INTERRUPTION();

        IGRAPH_CHECK(igraph_bh_tree_build(&tree, res, NULL));

        igraph_matrix_t forces;
        IGRAPH_CHECK(igraph_matrix_init(&forces, vcount, 3));
        IGRAPH_FINALLY(igraph_matrix_destroy, &forces);

        igraph_bh_calculate_repulsive_forces(&tree, &forces, yhu_repulsive_force_3d, &yhu_data);

        igraph_bh_calculate_attractive_forces(&tree, &from, &to, weights, &forces, yhu_attractive_force_3d, &yhu_data);

        double Fnorm = 0.0;
        for (igraph_integer_t i = 0; i < vcount; i++) {
            double fx = MATRIX(forces, i, 0);
            double fy = MATRIX(forces, i, 1);
            double fz = MATRIX(forces, i, 2);
            double fmag = sqrt(fx*fx + fy*fy + fz*fz);

            if (fmag > 1e-12) {
                MATRIX(forces, i, 0) = fx / fmag;
                MATRIX(forces, i, 1) = fy / fmag;
                MATRIX(forces, i, 2) = fz / fmag;
            }
            Fnorm += fmag;
        }

        for (igraph_integer_t i = 0; i < vcount; i++) {
            MATRIX(*res, i, 0) += step * MATRIX(forces, i, 0);
            MATRIX(*res, i, 1) += step * MATRIX(forces, i, 1);
            MATRIX(*res, i, 2) += step * MATRIX(forces, i, 2);
        }

        if (iter > 0 && adaptive_cooling) {
            step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
        }

        if (step < tolerance) {
            igraph_matrix_destroy(&forces);
            break;
        }

        Fnorm0 = Fnorm;
        igraph_matrix_destroy(&forces);
        IGRAPH_FINALLY_CLEAN(1);
    }

    if (beautify_leaves_flag) {
        IGRAPH_CHECK(beautify_leaves_3d(graph, res));
    }

    igraph_bh_tree_destroy(&tree);
    igraph_vector_int_destroy(&from);
    igraph_vector_int_destroy(&to);
    IGRAPH_FINALLY_CLEAN(3);

    return IGRAPH_SUCCESS;
}

static igraph_error_t igraph_layout_i_yifan_hu_sfdp(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t maxiter,
        igraph_real_t repulsive_exponent,
        igraph_real_t natural_length,
        igraph_real_t step,
        igraph_bool_t adaptive_cooling,
        igraph_real_t tolerance,
        igraph_quadtree_scheme_t quadtree_scheme,
        igraph_int_t max_qtree_level,
        igraph_bool_t beautify_leaves_flag,
        const igraph_vector_t *weights,
        const igraph_vector_t *minx,
        const igraph_vector_t *maxx,
        const igraph_vector_t *miny,
        const igraph_vector_t *maxy) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    if (!use_seed) {
        IGRAPH_CHECK(igraph_matrix_resize(res, vcount, 2));
        igraph_i_layout_random_bounded(graph, res, minx, maxx, miny, maxy);
    }

    double K = natural_length;
    if (K < 0) {
        IGRAPH_CHECK(compute_average_edge_length(graph, res, weights, &K));
    }

    double p = repulsive_exponent;
    if (p >= 0) {
        p = -1.0;
    }

    double KP = pow(K, 1.0 - p);
    double CRK = pow(IGRAPH_YHU_C, (2.0 - p) / 3.0) / K;

    yhu_data_t yhu_data = {
        .p = p,
        .KP = KP,
        .CRK = CRK,
        .K = K,
        .dim = 2,
        .graph = graph,
        .weights = weights
    };

    igraph_bh_tree_t tree;
    IGRAPH_CHECK(igraph_bh_tree_init(&tree, 2, IGRAPH_YHU_BH, (int)max_qtree_level, 1));
    IGRAPH_FINALLY(igraph_bh_tree_destroy, &tree);

    igraph_vector_int_t from, to;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&from, ecount);
    IGRAPH_VECTOR_INT_INIT_FINALLY(&to, ecount);

    for (igraph_integer_t e = 0; e < ecount; e++) {
        VECTOR(from)[e] = IGRAPH_FROM(graph, e);
        VECTOR(to)[e] = IGRAPH_TO(graph, e);
    }

    double Fnorm0 = IGRAPH_INFINITY;

    for (igraph_int_t iter = 0; iter < maxiter; iter++) {
        IGRAPH_ALLOW_INTERRUPTION();

        IGRAPH_CHECK(igraph_bh_tree_build(&tree, res, NULL));

        igraph_matrix_t forces;
        IGRAPH_CHECK(igraph_matrix_init(&forces, vcount, 2));
        IGRAPH_FINALLY(igraph_matrix_destroy, &forces);

        igraph_bh_calculate_repulsive_forces(&tree, &forces, yhu_repulsive_force, &yhu_data);

        igraph_bh_calculate_attractive_forces(&tree, &from, &to, weights, &forces, yhu_attractive_force, &yhu_data);

        double Fnorm = 0.0;
        for (igraph_integer_t i = 0; i < vcount; i++) {
            double fx = MATRIX(forces, i, 0);
            double fy = MATRIX(forces, i, 1);
            double fmag = sqrt(fx*fx + fy*fy);

            if (fmag > 1e-12) {
                MATRIX(forces, i, 0) = fx / fmag;
                MATRIX(forces, i, 1) = fy / fmag;
            }
            Fnorm += fmag;
        }

        for (igraph_integer_t i = 0; i < vcount; i++) {
            MATRIX(*res, i, 0) += step * MATRIX(forces, i, 0);
            MATRIX(*res, i, 1) += step * MATRIX(forces, i, 1);

            if (minx && MATRIX(*res, i, 0) < VECTOR(*minx)[i]) {
                MATRIX(*res, i, 0) = VECTOR(*minx)[i];
            }
            if (maxx && MATRIX(*res, i, 0) > VECTOR(*maxx)[i]) {
                MATRIX(*res, i, 0) = VECTOR(*maxx)[i];
            }
            if (miny && MATRIX(*res, i, 1) < VECTOR(*miny)[i]) {
                MATRIX(*res, i, 1) = VECTOR(*miny)[i];
            }
            if (maxy && MATRIX(*res, i, 1) > VECTOR(*maxy)[i]) {
                MATRIX(*res, i, 1) = VECTOR(*maxy)[i];
            }
        }

        igraph_matrix_destroy(&forces);
        IGRAPH_FINALLY_CLEAN(1);

        if (iter > 0 && step <= tolerance) {
            break;
        }

        step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
        Fnorm0 = Fnorm;
    }

    if (beautify_leaves_flag) {
        IGRAPH_CHECK(beautify_leaves(graph, res));
    }

    igraph_vector_int_destroy(&from);
    igraph_vector_int_destroy(&to);
    igraph_bh_tree_destroy(&tree);
    IGRAPH_FINALLY_CLEAN(3);

    return IGRAPH_SUCCESS;
}

static igraph_error_t igraph_layout_i_yifan_hu_exact(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t maxiter,
        igraph_real_t repulsive_exponent,
        igraph_real_t natural_length,
        igraph_real_t step,
        igraph_bool_t adaptive_cooling,
        igraph_real_t tolerance,
        igraph_bool_t beautify_leaves_flag,
        const igraph_vector_t *weights,
        const igraph_vector_t *minx,
        const igraph_vector_t *maxx,
        const igraph_vector_t *miny,
        const igraph_vector_t *maxy) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    if (!use_seed) {
        IGRAPH_CHECK(igraph_matrix_resize(res, vcount, 2));
        igraph_i_layout_random_bounded(graph, res, minx, maxx, miny, maxy);
    }

    double K = natural_length;
    if (K < 0) {
        IGRAPH_CHECK(compute_average_edge_length(graph, res, weights, &K));
    }

    double p = repulsive_exponent;
    if (p >= 0) {
        p = -1.0;
    }

    double KP = pow(K, 1.0 - p);
    double CRK = pow(IGRAPH_YHU_C, (2.0 - p) / 3.0) / K;

    igraph_vector_t disp_x, disp_y;
    IGRAPH_VECTOR_INIT_FINALLY(&disp_x, vcount);
    IGRAPH_VECTOR_INIT_FINALLY(&disp_y, vcount);

    igraph_vector_int_t from, to;
    IGRAPH_VECTOR_INT_INIT_FINALLY(&from, ecount);
    IGRAPH_VECTOR_INT_INIT_FINALLY(&to, ecount);

    for (igraph_integer_t e = 0; e < ecount; e++) {
        VECTOR(from)[e] = IGRAPH_FROM(graph, e);
        VECTOR(to)[e] = IGRAPH_TO(graph, e);
    }

    double Fnorm0 = IGRAPH_INFINITY;

    for (igraph_int_t iter = 0; iter < maxiter; iter++) {
        IGRAPH_ALLOW_INTERRUPTION();

        igraph_vector_null(&disp_x);
        igraph_vector_null(&disp_y);

        for (igraph_integer_t i = 0; i < vcount; i++) {
            for (igraph_integer_t j = i + 1; j < vcount; j++) {
                double dx = MATRIX(*res, i, 0) - MATRIX(*res, j, 0);
                double dy = MATRIX(*res, i, 1) - MATRIX(*res, j, 1);
                double d2 = dx*dx + dy*dy;

                if (d2 == 0) {
                    dx = RNG_UNIF(-1e-9, 1e-9);
                    dy = RNG_UNIF(-1e-9, 1e-9);
                    d2 = dx*dx + dy*dy;
                }

                double dist = sqrt(d2);
                double exp_factor = 1.0 - p;
                if (exp_factor < 0) {
                    exp_factor = -exp_factor;
                }

                double scale = KP / pow(dist, exp_factor);

                if (!isfinite(scale)) {
                    scale = 1e10;
                }

                VECTOR(disp_x)[i] += scale * dx / dist;
                VECTOR(disp_y)[i] += scale * dy / dist;
                VECTOR(disp_x)[j] -= scale * dx / dist;
                VECTOR(disp_y)[j] -= scale * dy / dist;
            }
        }

        for (igraph_integer_t e = 0; e < ecount; e++) {
            igraph_integer_t from_idx = VECTOR(from)[e];
            igraph_integer_t to_idx = VECTOR(to)[e];
            double dx = MATRIX(*res, from_idx, 0) - MATRIX(*res, to_idx, 0);
            double dy = MATRIX(*res, from_idx, 1) - MATRIX(*res, to_idx, 1);
            double w = weights ? VECTOR(*weights)[e] : 1.0;
            double dist = sqrt(dx*dx + dy*dy);
            if (dist == 0) {
                dist = 1e-9;
            }

            double scale = -CRK * dist * w;

            VECTOR(disp_x)[from_idx] += scale * dx;
            VECTOR(disp_y)[from_idx] += scale * dy;
            VECTOR(disp_x)[to_idx] -= scale * dx;
            VECTOR(disp_y)[to_idx] -= scale * dy;
        }

        double Fnorm = 0.0;
        for (igraph_integer_t i = 0; i < vcount; i++) {
            double fx = VECTOR(disp_x)[i];
            double fy = VECTOR(disp_y)[i];
            double fmag = sqrt(fx*fx + fy*fy);

            if (fmag > 1e-12) {
                VECTOR(disp_x)[i] /= fmag;
                VECTOR(disp_y)[i] /= fmag;
            }
            Fnorm += fmag;
        }

        for (igraph_integer_t i = 0; i < vcount; i++) {
            MATRIX(*res, i, 0) += step * VECTOR(disp_x)[i];
            MATRIX(*res, i, 1) += step * VECTOR(disp_y)[i];

            if (minx && MATRIX(*res, i, 0) < VECTOR(*minx)[i]) {
                MATRIX(*res, i, 0) = VECTOR(*minx)[i];
            }
            if (maxx && MATRIX(*res, i, 0) > VECTOR(*maxx)[i]) {
                MATRIX(*res, i, 0) = VECTOR(*maxx)[i];
            }
            if (miny && MATRIX(*res, i, 1) < VECTOR(*miny)[i]) {
                MATRIX(*res, i, 1) = VECTOR(*miny)[i];
            }
            if (maxy && MATRIX(*res, i, 1) > VECTOR(*maxy)[i]) {
                MATRIX(*res, i, 1) = VECTOR(*maxy)[i];
            }
        }

        if (iter > 0 && step <= tolerance) {
            break;
        }

        step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
        Fnorm0 = Fnorm;
    }

    if (beautify_leaves_flag) {
        IGRAPH_CHECK(beautify_leaves(graph, res));
    }

    igraph_vector_destroy(&disp_x);
    igraph_vector_destroy(&disp_y);
    igraph_vector_int_destroy(&from);
    igraph_vector_int_destroy(&to);
    IGRAPH_FINALLY_CLEAN(4);

    return IGRAPH_SUCCESS;
}

/**
 * \ingroup layout
 * \function igraph_layout_yifan_hu
 * \brief Yifan Hu layout algorithm.
 *
 * This function implements the Yifan Hu force-directed layout algorithm,
 * based on the Graphviz SFDP (Spring-Electrical Force-Directed Placement)
 * implementation.
 *
 * </para><para>
 * The algorithm uses an attraction force between connected vertices and
 * a repulsion force between all vertex pairs. The repulsion force uses
 * the Barnes-Hut quadtree optimization for O(n log n) performance on
 * larger graphs.
 *
 * </para><para>
 * The algorithm uses adaptive cooling: the step size is decreased when
 * the total force magnitude increases, and increased when it decreases.
 *
 * </para><para>
 * Reference:
 *
 * </para><para>
 * Hu, Y. F.:
 * Efficient, High-Quality Force-Directed Graph Drawing.
 * Software -- Practice and Experience, 2005.
 *
 * \param graph Pointer to an initialized graph object.
 * \param res Pointer to an initialized matrix object. This will
 *        contain the result and will be resized as needed.
 * \param use_seed If true the supplied values in the
 *        \p res argument are used as an initial layout, if
 *        false a random initial layout is used.
 * \param maxiter The number of iterations to perform. A reasonable
 *        default value is 500.
 * \param repulsive_exponent Repulsive force exponent. Default is -1.0 (SFDP).
 * \param natural_length Natural edge length. If negative, the average
 *        edge length is used. Default is -1.0 (auto).
 * \param step Initial step size. Default is 0.1.
 * \param adaptive_cooling Use adaptive step cooling. Default is true.
 * \param tolerance Convergence tolerance. If step size falls below
 *        this value, the algorithm stops. Default is 0.001.
 * \param quadtree_scheme Quadtree scheme to use. Options:
 *        \c IGRAPH_QUADTREE_NORMAL, \c IGRAPH_QUADTREE_FAST,
 *        \c IGRAPH_QUADTREE_HYBRID, \c IGRAPH_QUADTREE_NONE.
 *        Default is \c IGRAPH_QUADTREE_NORMAL.
 * \param max_qtree_level Maximum quadtree depth. Default is 10.
 * \param beautify_leaves Arrange degree-1 nodes around their parent.
 *        Default is false.
 * \param weights Pointer to a vector containing edge weights. Weights must
 *        be positive. If \c NULL, all edges are assumed to have weight 1.
 * \param minx Pointer to a vector, or a \c NULL pointer. If not a
 *        \c NULL pointer then the vector gives the minimum
 *        \quote x \endquote coordinate for every vertex.
 * \param maxx Same as \p minx, but the maximum \quote x \endquote
 *        coordinates.
 * \param miny Pointer to a vector, or a \c NULL pointer. If not a
 *        \c NULL pointer then the vector gives the minimum
 *        \quote y \endquote coordinate for every vertex.
 * \param maxy Same as \p miny, but the maximum \quote y \endquote
 *        coordinates.
 * \return Error code.
 *
 * Time complexity: O(n^2) per iteration for exact algorithm,
 * O(n log n) per iteration for Barnes-Hut, where n is the
 * number of vertices.
 */

igraph_error_t igraph_layout_yifan_hu(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t maxiter,
        igraph_real_t repulsive_exponent,
        igraph_real_t natural_length,
        igraph_real_t step,
        igraph_bool_t adaptive_cooling,
        igraph_real_t tolerance,
        igraph_quadtree_scheme_t quadtree_scheme,
        igraph_int_t max_qtree_level,
        igraph_bool_t beautify_leaves,
        const igraph_vector_t *weights,
        const igraph_vector_t *minx,
        const igraph_vector_t *maxx,
        const igraph_vector_t *miny,
        const igraph_vector_t *maxy) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (maxiter < 0) {
        IGRAPH_ERROR("Number of iterations must be non-negative in "
                     "Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (step <= 0) {
        IGRAPH_ERROR("Step size must be positive in "
                     "Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (tolerance <= 0) {
        IGRAPH_ERROR("Tolerance must be positive in "
                     "Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (use_seed && (igraph_matrix_nrow(res) != vcount ||
                     igraph_matrix_ncol(res) != 2)) {
        IGRAPH_ERROR("Invalid start position matrix size in "
                     "Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (weights && igraph_vector_size(weights) != ecount) {
        IGRAPH_ERROR("Invalid weight vector length.", IGRAPH_EINVAL);
    }
    if (weights && ecount > 0 && igraph_vector_min(weights) <= 0) {
        IGRAPH_ERROR("Weights must be positive for Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (minx && igraph_vector_size(minx) != vcount) {
        IGRAPH_ERROR("Invalid minx vector length.", IGRAPH_EINVAL);
    }
    if (maxx && igraph_vector_size(maxx) != vcount) {
        IGRAPH_ERROR("Invalid maxx vector length.", IGRAPH_EINVAL);
    }
    if (minx && maxx && !igraph_vector_all_le(minx, maxx)) {
        IGRAPH_ERROR("minx must not be greater than maxx.", IGRAPH_EINVAL);
    }
    if (miny && igraph_vector_size(miny) != vcount) {
        IGRAPH_ERROR("Invalid miny vector length.", IGRAPH_EINVAL);
    }
    if (maxy && igraph_vector_size(maxy) != vcount) {
        IGRAPH_ERROR("Invalid maxy vector length.", IGRAPH_EINVAL);
    }
    if (miny && maxy && !igraph_vector_all_le(miny, maxy)) {
        IGRAPH_ERROR("miny must not be greater than maxy.", IGRAPH_EINVAL);
    }

    if (vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    igraph_bool_t use_quadtree = (quadtree_scheme != IGRAPH_QUADTREE_NONE);

    if (quadtree_scheme == IGRAPH_QUADTREE_HYBRID && vcount > 10000) {
        use_quadtree = true;
    } else if (quadtree_scheme == IGRAPH_QUADTREE_HYBRID) {
        use_quadtree = false;
    }

    if (use_quadtree && vcount < IGRAPH_YHU_QUADTREE_SIZE) {
        use_quadtree = false;
    }

    if (use_quadtree) {
        return igraph_layout_i_yifan_hu_sfdp(graph, res, use_seed, maxiter,
                                             repulsive_exponent, natural_length,
                                             step, adaptive_cooling, tolerance,
                                             quadtree_scheme, max_qtree_level,
                                             beautify_leaves,
                                             weights, minx, maxx, miny, maxy);
    } else {
        return igraph_layout_i_yifan_hu_exact(graph, res, use_seed, maxiter,
                                              repulsive_exponent, natural_length,
                                              step, adaptive_cooling, tolerance,
                                              beautify_leaves,
                                              weights, minx, maxx, miny, maxy);
    }
}

/**
 * \function igraph_layout_yifan_hu_3d
 * \brief Yifan Hu layout (3D)
 *
 * \experimental
 *
 * This is the 3D variant of the Yifan Hu layout algorithm, based on
 * Graphviz's SFDP algorithm with Barnes-Hut optimization.
 *
 * \param graph Pointer to an initialized graph object.
 * \param res Pointer to an initialized matrix object. This will
 *        contain the result and will be resized as needed.
 * \param use_seed If true the supplied values in the
 *        \p res argument are used as an initial layout, if
 *        false a random initial layout is used.
 * \param maxiter The number of iterations to perform. A reasonable
 *        default value is 500.
 * \param repulsive_exponent Repulsive force exponent. Default is -1.0 (SFDP).
 * \param natural_length Natural edge length. If negative, the average
 *        edge length is used. Default is -1.0 (auto).
 * \param step Initial step size. Default is 0.1.
 * \param adaptive_cooling Use adaptive step cooling. Default is true.
 * \param tolerance Convergence tolerance. If step size falls below
 *        this value, the algorithm stops. Default is 0.001.
 * \param quadtree_scheme Quadtree scheme to use. Options:
 *        \c IGRAPH_QUADTREE_NORMAL, \c IGRAPH_QUADTREE_FAST,
 *        \c IGRAPH_QUADTREE_HYBRID, \c IGRAPH_QUADTREE_NONE.
 *        Default is \c IGRAPH_QUADTREE_NORMAL.
 * \param max_qtree_level Maximum octree depth. Default is 10.
 * \param beautify_leaves Arrange degree-1 nodes around their parent.
 *        Default is false.
 * \param weights Pointer to a vector containing edge weights. Weights must
 *        be positive. If \c NULL, all edges are assumed to have weight 1.
 * \return Error code.
 *
 * Time complexity: O(n^2) per iteration for exact algorithm,
 * O(n log n) per iteration for Barnes-Hut, where n is the
 * number of vertices.
 */

igraph_error_t igraph_layout_yifan_hu_3d(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t maxiter,
        igraph_real_t repulsive_exponent,
        igraph_real_t natural_length,
        igraph_real_t step,
        igraph_bool_t adaptive_cooling,
        igraph_real_t tolerance,
        igraph_quadtree_scheme_t quadtree_scheme,
        igraph_int_t max_qtree_level,
        igraph_bool_t beautify_leaves,
        const igraph_vector_t *weights) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (maxiter < 0) {
        IGRAPH_ERROR("Number of iterations must be non-negative in "
                     "Yifan Hu 3D layout.", IGRAPH_EINVAL);
    }

    if (step <= 0) {
        IGRAPH_ERROR("Step size must be positive in "
                     "Yifan Hu 3D layout.", IGRAPH_EINVAL);
    }

    if (tolerance <= 0) {
        IGRAPH_ERROR("Tolerance must be positive in "
                     "Yifan Hu 3D layout.", IGRAPH_EINVAL);
    }

    if (use_seed && (igraph_matrix_nrow(res) != vcount ||
                     igraph_matrix_ncol(res) != 3)) {
        IGRAPH_ERROR("Invalid start position matrix size in "
                     "Yifan Hu 3D layout.", IGRAPH_EINVAL);
    }

    if (weights && igraph_vector_size(weights) != ecount) {
        IGRAPH_ERROR("Invalid weight vector length.", IGRAPH_EINVAL);
    }
    if (weights && ecount > 0 && igraph_vector_min(weights) <= 0) {
        IGRAPH_ERROR("Weights must be positive for Yifan Hu 3D layout.", IGRAPH_EINVAL);
    }

    if (vcount == 0) {
        return IGRAPH_SUCCESS;
    }

    igraph_bool_t use_octree = (quadtree_scheme != IGRAPH_QUADTREE_NONE);

    if (quadtree_scheme == IGRAPH_QUADTREE_HYBRID && vcount > 10000) {
        use_octree = true;
    } else if (quadtree_scheme == IGRAPH_QUADTREE_HYBRID) {
        use_octree = false;
    }

    /* For 3D, always use octree when available since we don't have an exact implementation */
    if (use_octree || quadtree_scheme == IGRAPH_QUADTREE_NORMAL) {
        return igraph_layout_i_yifan_hu_sfdp_3d(graph, res, use_seed, maxiter,
                                                 repulsive_exponent, natural_length,
                                                 step, adaptive_cooling, tolerance,
                                                 quadtree_scheme, max_qtree_level,
                                                 beautify_leaves, weights);
    } else {
        IGRAPH_ERROR("Exact 3D Yifan Hu layout not implemented. "
                     "Use Barnes-Hut (quadtree_scheme != NONE).", IGRAPH_EINVAL);
        return IGRAPH_EINVAL;
    }
}
