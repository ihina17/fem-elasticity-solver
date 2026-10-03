"""
Boundary-load utilities for linear elasticity.

This module currently supports constant traction applied
along a sequence of two-node boundary segments.
"""

import numpy as np


# ============================================================
# Constant edge traction
# ============================================================

def create_constant_edge_traction_loads(
    Coord,
    edge_nodes,
    traction,
    thickness=1.0,
):
    """
    Convert a constant boundary traction into equivalent
    nodal forces.

    Parameters
    ----------
    Coord : np.ndarray
        Nodal coordinates with shape (NumNodes, 2).

    edge_nodes : array-like
        Ordered 1-based node numbers along the loaded edge.

    traction : array-like
        Constant traction vector

            [tx, ty]

    thickness : float
        Out-of-plane thickness.

    Returns
    -------
    NodalLoads : np.ndarray
        Array with rows

            [node, dof, value]

        where

            dof = 1 : x direction
            dof = 2 : y direction

    Notes
    -----
    For a straight two-node boundary segment of length Le
    under constant traction t,

        f1 = t * thickness * Le / 2
        f2 = t * thickness * Le / 2

    This is the consistent nodal force vector for a linear edge.
    """

    # --------------------------------------------------------
    # Convert inputs
    # --------------------------------------------------------

    Coord = np.asarray(
        Coord,
        dtype=float,
    )

    edge_nodes = np.asarray(
        edge_nodes,
        dtype=int,
    ).ravel()

    traction = np.asarray(
        traction,
        dtype=float,
    ).ravel()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if Coord.ndim != 2:

        raise ValueError(
            "Coord must be a two-dimensional array."
        )


    if Coord.shape[1] != 2:

        raise ValueError(
            "Coord must contain two spatial coordinates."
        )


    if edge_nodes.size < 2:

        raise ValueError(
            "At least two edge nodes are required."
        )


    if traction.size != 2:

        raise ValueError(
            "traction must contain two components."
        )


    if thickness <= 0.0:

        raise ValueError(
            "thickness must be positive."
        )


    if np.any(
        edge_nodes < 1
    ):

        raise ValueError(
            "edge_nodes must use 1-based node numbering."
        )


    if np.any(
        edge_nodes > Coord.shape[0]
    ):

        raise ValueError(
            "edge_nodes contains an invalid node number."
        )


    # --------------------------------------------------------
    # Initialize accumulated nodal forces
    # --------------------------------------------------------

    nodal_force = {}


    for node in edge_nodes:

        nodal_force[
            int(node)
        ] = np.zeros(
            2,
            dtype=float,
        )


    # --------------------------------------------------------
    # Integrate traction over each edge segment
    # --------------------------------------------------------

    for i in range(
        edge_nodes.size - 1
    ):

        node_a = int(
            edge_nodes[i]
        )

        node_b = int(
            edge_nodes[i + 1]
        )


        xa = Coord[
            node_a - 1,
            :
        ]

        xb = Coord[
            node_b - 1,
            :
        ]


        edge_length = np.linalg.norm(
            xb - xa
        )


        if edge_length <= 0.0:

            raise ValueError(
                "Boundary edge length must be positive."
            )


        # ----------------------------------------------------
        # Consistent endpoint force
        # ----------------------------------------------------

        endpoint_force = (
            traction
            * thickness
            * edge_length
            / 2.0
        )


        nodal_force[
            node_a
        ] += endpoint_force


        nodal_force[
            node_b
        ] += endpoint_force


    # --------------------------------------------------------
    # Convert to [node, dof, value] rows
    # --------------------------------------------------------

    NodalLoads = []


    for node in edge_nodes:

        node = int(
            node
        )


        NodalLoads.append([
            node,
            1,
            nodal_force[node][0],
        ])


        NodalLoads.append([
            node,
            2,
            nodal_force[node][1],
        ])


    return np.array(
        NodalLoads,
        dtype=float,
    )