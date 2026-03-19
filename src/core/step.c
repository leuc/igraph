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

#include "igraph_step.h"

#include "config.h"

static IGRAPH_THREAD_LOCAL igraph_step_handler_t *igraph_i_step_handler = 0;

/**
 * \function igraph_step
 * \brief Report a step in a calculation from an igraph function.
 *
 * Note that the usual way to report a step is the \ref IGRAPH_STEP
 * macro, as that takes care of the return value of the step
 * handler.
 * \param state Pointer to the current algorithm state.
 * \param data User-defined data.
 * \return Error code from the step handler function, or \c IGRAPH_SUCCESS
 *         if no step handler function was registered.
 *
 * Time complexity: O(1).
 */

igraph_error_t igraph_step(const void *state, void *data) {
    if (igraph_i_step_handler) {
        return igraph_i_step_handler(state, data);
    }
    return IGRAPH_SUCCESS;
}

/**
 * \function igraph_set_step_handler
 * \brief Install a step handler, or remove the current handler.
 *
 * \param new_handler Pointer to a function of type
 *     \ref igraph_step_handler_t, the step handler function to
 *     install. To uninstall the current step handler, this argument
 *     can be a null pointer.
 * \return Pointer to the previously installed step handler function.
 *
 * Time complexity: O(1).
 */

igraph_step_handler_t *
igraph_set_step_handler(igraph_step_handler_t *new_handler) {
    igraph_step_handler_t *previous_handler = igraph_i_step_handler;
    igraph_i_step_handler = new_handler;
    return previous_handler;
}
