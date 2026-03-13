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

#include "igraph_random.h"
#include "igraph_interface.h"

#include "core/grid.h"
#include "core/interruption.h"
#include "layout/layout_internal.h"

static igraph_error_t igraph_layout_i_yifan_hu(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t niter,
        igraph_real_t relative_strength,
        igraph_real_t step_ratio,
        igraph_real_t convergence_threshold,
        const igraph_vector_t *weights,
        const igraph_vector_t *minx,
        const igraph_vector_t *maxx,
        const igraph_vector_t *miny,
        const igraph_vector_t *maxy) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);
    igraph_vector_t disp_x, disp_y;
    igraph_real_t optimal_distance, step;
    igraph_real_t energy0 = IGRAPH_INFINITY, energy;
    igraph_int_t progress = 0;
    igraph_real_t avg_edge_length = 0;

    if (ecount > 0) {
        igraph_real_t total_len = 0;
        for (igraph_int_t e = 0; e < ecount; e++) {
            igraph_int_t from = IGRAPH_FROM(graph, e);
            igraph_int_t to = IGRAPH_TO(graph, e);
            igraph_real_t dx = MATRIX(*res, from, 0) - MATRIX(*res, to, 0);
            igraph_real_t dy = MATRIX(*res, from, 1) - MATRIX(*res, to, 1);
            igraph_real_t w = weights ? VECTOR(*weights)[e] : 1.0;
            total_len += sqrt(dx*dx + dy*dy) * w;
        }
        avg_edge_length = total_len / ecount;
    } else {
        avg_edge_length = 1.0;
    }

    optimal_distance = pow(relative_strength, 1.0/3.0) * avg_edge_length;
    step = optimal_distance / 5.0;

    IGRAPH_VECTOR_INIT_FINALLY(&disp_x, vcount);
    IGRAPH_VECTOR_INIT_FINALLY(&disp_y, vcount);

    for (igraph_int_t iter = 0; iter < niter; iter++) {
        IGRAPH_ALLOW_INTERRUPTION();

        igraph_vector_null(&disp_x);
        igraph_vector_null(&disp_y);

        for (igraph_int_t i = 0; i < vcount; i++) {
            for (igraph_int_t j = i + 1; j < vcount; j++) {
                igraph_real_t dx = MATRIX(*res, i, 0) - MATRIX(*res, j, 0);
                igraph_real_t dy = MATRIX(*res, i, 1) - MATRIX(*res, j, 1);
                igraph_real_t d2 = dx*dx + dy*dy;

                if (d2 == 0) {
                    dx = RNG_UNIF(-1e-9, 1e-9);
                    dy = RNG_UNIF(-1e-9, 1e-9);
                    d2 = dx*dx + dy*dy;
                }

                igraph_real_t dist = sqrt(d2);
                igraph_real_t scale = -relative_strength * optimal_distance * optimal_distance / d2;

                if (!isfinite(scale)) {
                    scale = -1.0;
                }

                VECTOR(disp_x)[i] += scale * dx / dist;
                VECTOR(disp_y)[i] += scale * dy / dist;
                VECTOR(disp_x)[j] -= scale * dx / dist;
                VECTOR(disp_y)[j] -= scale * dy / dist;
            }
        }

        for (igraph_int_t e = 0; e < ecount; e++) {
            igraph_int_t from = IGRAPH_FROM(graph, e);
            igraph_int_t to = IGRAPH_TO(graph, e);
            igraph_real_t dx = MATRIX(*res, from, 0) - MATRIX(*res, to, 0);
            igraph_real_t dy = MATRIX(*res, from, 1) - MATRIX(*res, to, 1);
            igraph_real_t w = weights ? VECTOR(*weights)[e] : 1.0;
            igraph_real_t dist = sqrt(dx*dx + dy*dy);
            if (dist == 0) {
                dist = 1e-9;
            }
            igraph_real_t scale = dist * w / optimal_distance;

            VECTOR(disp_x)[from] -= dx * scale;
            VECTOR(disp_y)[from] -= dy * scale;
            VECTOR(disp_x)[to] += dx * scale;
            VECTOR(disp_y)[to] += dy * scale;
        }

        energy = 0;
        igraph_real_t max_force = 1;
        for (igraph_int_t i = 0; i < vcount; i++) {
            igraph_real_t fx = VECTOR(disp_x)[i];
            igraph_real_t fy = VECTOR(disp_y)[i];
            igraph_real_t norm = sqrt(fx*fx + fy*fy);
            energy += norm;
            if (norm > max_force) {
                max_force = norm;
            }
        }

        for (igraph_int_t i = 0; i < vcount; i++) {
            VECTOR(disp_x)[i] *= step / max_force;
            VECTOR(disp_y)[i] *= step / max_force;
        }

        for (igraph_int_t i = 0; i < vcount; i++) {
            MATRIX(*res, i, 0) += VECTOR(disp_x)[i];
            MATRIX(*res, i, 1) += VECTOR(disp_y)[i];

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

        if (iter > 0 && convergence_threshold > 0) {
            if (fabs((energy - energy0) / energy) < convergence_threshold) {
                break;
            }
        }

        if (energy < energy0) {
            progress++;
            if (progress >= 5) {
                progress = 0;
                step /= step_ratio;
            }
        } else {
            progress = 0;
            step *= step_ratio;
        }

        energy0 = energy;
    }

    igraph_vector_destroy(&disp_x);
    igraph_vector_destroy(&disp_y);
    IGRAPH_FINALLY_CLEAN(2);

    return IGRAPH_SUCCESS;
}

static igraph_error_t igraph_layout_i_grid_yifan_hu(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t niter,
        igraph_real_t relative_strength,
        igraph_real_t step_ratio,
        igraph_real_t convergence_threshold,
        const igraph_vector_t *weights,
        const igraph_vector_t *minx,
        const igraph_vector_t *maxx,
        const igraph_vector_t *miny,
        const igraph_vector_t *maxy) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);
    const igraph_real_t width = sqrt(vcount), height = width;
    igraph_2dgrid_t grid;
    igraph_vector_t disp_x, disp_y;
    igraph_real_t optimal_distance, step;
    igraph_real_t energy0 = IGRAPH_INFINITY, energy;
    igraph_int_t progress = 0;
    igraph_2dgrid_iterator_t vidit;
    const igraph_real_t cellsize = 2.0;
    igraph_real_t avg_edge_length = 0;

    if (ecount > 0) {
        igraph_real_t total_len = 0;
        for (igraph_int_t e = 0; e < ecount; e++) {
            igraph_int_t from = IGRAPH_FROM(graph, e);
            igraph_int_t to = IGRAPH_TO(graph, e);
            igraph_real_t dx = MATRIX(*res, from, 0) - MATRIX(*res, to, 0);
            igraph_real_t dy = MATRIX(*res, from, 1) - MATRIX(*res, to, 1);
            igraph_real_t w = weights ? VECTOR(*weights)[e] : 1.0;
            total_len += sqrt(dx*dx + dy*dy) * w;
        }
        avg_edge_length = total_len / ecount;
    } else {
        avg_edge_length = 1.0;
    }

    optimal_distance = pow(relative_strength, 1.0/3.0) * avg_edge_length;
    step = optimal_distance / 5.0;

    IGRAPH_CHECK(igraph_2dgrid_init(&grid, res, -width/2, width/2, cellsize,
                                    -height/2, height/2, cellsize));
    IGRAPH_FINALLY(igraph_2dgrid_destroy, &grid);

    for (igraph_int_t i = 0; i < vcount; i++) {
        igraph_2dgrid_add2(&grid, i);
    }

    IGRAPH_VECTOR_INIT_FINALLY(&disp_x, vcount);
    IGRAPH_VECTOR_INIT_FINALLY(&disp_y, vcount);

    for (igraph_int_t iter = 0; iter < niter; iter++) {
        igraph_int_t v, u;

        IGRAPH_ALLOW_INTERRUPTION();

        igraph_vector_null(&disp_x);
        igraph_vector_null(&disp_y);

        igraph_2dgrid_reset(&grid, &vidit);
        while ((v = igraph_2dgrid_next(&grid, &vidit) - 1) != -1) {
            while ((u = igraph_2dgrid_next_nei(&grid, &vidit) - 1) != -1) {
                igraph_real_t dx = MATRIX(*res, v, 0) - MATRIX(*res, u, 0);
                igraph_real_t dy = MATRIX(*res, v, 1) - MATRIX(*res, u, 1);
                igraph_real_t d2 = dx*dx + dy*dy;

                if (d2 == 0) {
                    dx = RNG_UNIF(-1e-9, 1e-9);
                    dy = RNG_UNIF(-1e-9, 1e-9);
                    d2 = dx*dx + dy*dy;
                }

                if (d2 < cellsize * cellsize) {
                    igraph_real_t dist = sqrt(d2);
                    igraph_real_t scale = -relative_strength * optimal_distance * optimal_distance / d2;

                    if (!isfinite(scale)) {
                        scale = -1.0;
                    }

                    VECTOR(disp_x)[v] += scale * dx / dist;
                    VECTOR(disp_y)[v] += scale * dy / dist;
                    VECTOR(disp_x)[u] -= scale * dx / dist;
                    VECTOR(disp_y)[u] -= scale * dy / dist;
                }
            }
        }

        for (igraph_int_t e = 0; e < ecount; e++) {
            igraph_int_t from = IGRAPH_FROM(graph, e);
            igraph_int_t to = IGRAPH_TO(graph, e);
            igraph_real_t dx = MATRIX(*res, from, 0) - MATRIX(*res, to, 0);
            igraph_real_t dy = MATRIX(*res, from, 1) - MATRIX(*res, to, 1);
            igraph_real_t w = weights ? VECTOR(*weights)[e] : 1.0;
            igraph_real_t dist = sqrt(dx*dx + dy*dy);
            if (dist == 0) {
                dist = 1e-9;
            }
            igraph_real_t scale = dist * w / optimal_distance;

            VECTOR(disp_x)[from] -= dx * scale;
            VECTOR(disp_y)[from] -= dy * scale;
            VECTOR(disp_x)[to] += dx * scale;
            VECTOR(disp_y)[to] += dy * scale;
        }

        energy = 0;
        igraph_real_t max_force = 1;
        for (igraph_int_t i = 0; i < vcount; i++) {
            igraph_real_t fx = VECTOR(disp_x)[i];
            igraph_real_t fy = VECTOR(disp_y)[i];
            igraph_real_t norm = sqrt(fx*fx + fy*fy);
            energy += norm;
            if (norm > max_force) {
                max_force = norm;
            }
        }

        for (igraph_int_t i = 0; i < vcount; i++) {
            VECTOR(disp_x)[i] *= step / max_force;
            VECTOR(disp_y)[i] *= step / max_force;
        }

        for (igraph_int_t i = 0; i < vcount; i++) {
            MATRIX(*res, i, 0) += VECTOR(disp_x)[i];
            MATRIX(*res, i, 1) += VECTOR(disp_y)[i];

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

        if (iter > 0 && convergence_threshold > 0) {
            if (fabs((energy - energy0) / energy) < convergence_threshold) {
                break;
            }
        }

        if (energy < energy0) {
            progress++;
            if (progress >= 5) {
                progress = 0;
                step /= step_ratio;
            }
        } else {
            progress = 0;
            step *= step_ratio;
        }

        energy0 = energy;
    }

    igraph_vector_destroy(&disp_x);
    igraph_vector_destroy(&disp_y);
    igraph_2dgrid_destroy(&grid);
    IGRAPH_FINALLY_CLEAN(3);

    return IGRAPH_SUCCESS;
}

/**
 * \ingroup layout
 * \function igraph_layout_yifan_hu
 * \brief Yifan Hu layout algorithm.
 *
 * This function implements the Yifan Hu force-directed layout algorithm.
 *
 * </para><para>
 * The algorithm uses an attraction force between connected vertices and
 * a repulsion force between all vertex pairs. The attraction force is
 * modeled as a spring (Hooke's law) and the repulsion force as electrical
 * charge repulsion (Coulomb's law). The optimal distance between vertices
 * is computed from the average edge length and a relative strength parameter.
 *
 * </para><para>
 * The algorithm uses adaptive cooling: the step size is decreased when
 * the layout energy decreases for consecutive iterations, and increased
 * when the layout energy increases. This provides fast convergence while
 * avoiding oscillation.
 *
 * </para><para>
 * For large graphs, a grid-based approximation is used for the repulsion
 * calculation, providing O(n) performance.
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
 * \param niter The number of iterations to perform. A reasonable
 *        default value is 500.
 * \param relative_strength Relative strength of the repulsion force.
 *        Smaller values result in more spread out layouts. Default is 0.2.
 * \param step_ratio Step size ratio for adaptive cooling. Default is 0.95.
 * \param convergence_threshold Convergence threshold. If the relative
 *        energy difference is below this threshold, the algorithm stops.
 *        Default is 1e-4. Set to 0 to disable early stopping.
 * \param grid Whether to use the grid-based approximation for large graphs.
 *        Possible values: \c IGRAPH_LAYOUT_GRID, \c IGRAPH_LAYOUT_NOGRID,
 *        \c IGRAPH_LAYOUT_AUTOGRID. The last one uses the grid-based
 *        version for graphs with more than 1000 vertices.
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
 * Time complexity: O(n^2) per iteration for the exact algorithm,
 * O(n) per iteration for the grid-based approximation, where n is the
 * number of vertices.
 */

igraph_error_t igraph_layout_yifan_hu(
        const igraph_t *graph,
        igraph_matrix_t *res,
        igraph_bool_t use_seed,
        igraph_int_t niter,
        igraph_real_t relative_strength,
        igraph_real_t step_ratio,
        igraph_real_t convergence_threshold,
        igraph_layout_grid_t grid,
        const igraph_vector_t *weights,
        const igraph_vector_t *minx,
        const igraph_vector_t *maxx,
        const igraph_vector_t *miny,
        const igraph_vector_t *maxy) {

    const igraph_int_t vcount = igraph_vcount(graph);
    const igraph_int_t ecount = igraph_ecount(graph);

    if (niter < 0) {
        IGRAPH_ERROR("Number of iterations must be non-negative in "
                     "Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (relative_strength <= 0) {
        IGRAPH_ERROR("Relative strength must be positive in "
                     "Yifan Hu layout.", IGRAPH_EINVAL);
    }

    if (step_ratio <= 0 || step_ratio >= 1) {
        IGRAPH_ERROR("Step ratio must be in (0,1) in "
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

    if (!use_seed) {
        IGRAPH_CHECK(igraph_matrix_resize(res, vcount, 2));
        igraph_i_layout_random_bounded(graph, res, minx, maxx, miny, maxy);
    }

    if (grid == IGRAPH_LAYOUT_AUTOGRID) {
        if (vcount > 1000) {
            grid = IGRAPH_LAYOUT_GRID;
        } else {
            grid = IGRAPH_LAYOUT_NOGRID;
        }
    }

    if (grid == IGRAPH_LAYOUT_GRID) {
        return igraph_layout_i_grid_yifan_hu(graph, res, use_seed, niter,
                                             relative_strength, step_ratio,
                                             convergence_threshold,
                                             weights, minx, maxx, miny, maxy);
    } else {
        return igraph_layout_i_yifan_hu(graph, res, use_seed, niter,
                                        relative_strength, step_ratio,
                                        convergence_threshold,
                                        weights, minx, maxx, miny, maxy);
    }
}
