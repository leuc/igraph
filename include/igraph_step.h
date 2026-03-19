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

#ifndef IGRAPH_STEP_H
#define IGRAPH_STEP_H

#include "igraph_decls.h"
#include "igraph_error.h"

IGRAPH_BEGIN_C_DECLS

/**
 * \section about_step_handlers About execution step handlers
 *
 * <para>It is often useful to periodically yield control from
 * long-running algorithms to an external application, allowing
 * for progress monitoring, interruption, or cooperative multitasking.
 * A number of igraph functions support this at the time of writing.
 * </para>
 *
 * <para>
 * To handle step callbacks, the user has to install a
 * step handler, as there is none installed by default.
 * If an igraph function supports step handling, then it
 * calls the installed step handler periodically, and passes
 * a pointer to the current state. To install a step handler,
 * call \ref igraph_set_step_handler().
 * </para>
 */

/**
 * \section writing_step_handlers Writing step handlers
 *
 * <para>
 * To write a new step handler, one needs to create a function of
 * type \ref igraph_step_handler_t. The new step handler
 * can then be installed with the \ref igraph_set_step_handler()
 * function.
 * </para>
 *
 * <para>
 * The \p state parameter points to the current algorithm state,
 * which may be a graph, matrix, vector, or other data structure
 * depending on the algorithm. The handler can inspect this state
 * to report progress or determine whether to interrupt the computation.
 * </para>
 *
 * <para>
 * If the step handler returns a value other than \c IGRAPH_SUCCESS,
 * the algorithm will abort and return that error code, allowing
 * for clean interruption of long-running computations.
 * </para>
 */

/**
 * \section igraph_functions_with_step Writing igraph functions with step handling
 *
 * <para>
 * If you want to write a function that uses igraph and supports
 * step handling, you need to include \ref igraph_step()
 * calls in your function, usually via the \ref IGRAPH_STEP()
 * macro.
 * </para>
 *
 * <para>
 * It is good practice to call \ref igraph_step() at meaningful
 * points in the algorithm, such as after completing each iteration,
 * each level of a recursive algorithm, or after processing a
 * fixed number of elements.
 * </para>
 */

/**
 * \section step_and_threads Multi-threaded programs
 *
 * <para>
 * In multi-threaded programs, each thread has its own step
 * handler, if thread-local storage is supported and igraph is
 * thread-safe. See the \ref IGRAPH_THREAD_SAFE macro for checking
 * whether an igraph build is thread-safe.
 * </para>
 */

/* -------------------------------------------------- */
/* Step handlers                                       */
/* -------------------------------------------------- */

/**
 * \typedef igraph_step_handler_t
 * \brief Type of step handler functions
 *
 * This is the type of the igraph step handler functions.
 * \param state Pointer to the current algorithm state. The type
 *     of the state depends on the specific algorithm. Common
 *     types include \c igraph_t (graph), \c igraph_matrix_t,
 *     \c igraph_vector_int_t, or custom structures.
 * \param data User-defined data. Users can pass context
 *     information through this parameter.
 * \return If the return value is not \c IGRAPH_SUCCESS, then
 *     \ref igraph_step() returns the error code from the
 *     step handler intact. The \ref IGRAPH_STEP() macro also
 *     frees all allocated memory.
 */

typedef igraph_error_t igraph_step_handler_t(const void *state, void *data);

IGRAPH_EXPORT igraph_step_handler_t *igraph_set_step_handler(igraph_step_handler_t new_handler);

IGRAPH_EXPORT igraph_error_t igraph_step(const void *state, void *data);

/**
 * \define IGRAPH_STEP
 * \brief Report a step in a calculation from an igraph function (macro variant).
 *
 * The standard way to report a step from an igraph function.
 * \param state Pointer to the current algorithm state.
 * \param data User-defined data.
 * \return If the return value of the step handler is not
 *     \c IGRAPH_SUCCESS, then \ref igraph_step() returns the
 *     error code from the step handler intact. The \ref IGRAPH_STEP()
 *     macro also frees all allocated memory.
 */

#define IGRAPH_STEP(state, data) \
    do { \
        IGRAPH_CHECK(igraph_step((state), (data))); \
    } while (0)

IGRAPH_END_C_DECLS

#endif
