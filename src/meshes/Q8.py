"""
Q8 Mesh Generator

Structured eight-node serendipity quadrilateral mesh
for the cantilever beam problem.
"""

import numpy as np


# ============================================================
# Q8 cantilever mesh
# ============================================================

def generate_cantilever_q8_mesh(
    nx,
    length=1.0,
    height=0.1
):
    """
    Generate a structured Q8 mesh for a rectangular
    cantilever beam.

    The mesh uses one Q8 element through the beam depth
    and nx elements along the beam length.

    Q8 local node numbering:

        4 ----- 7 ----- 3
        |               |
        8               6
        |               |
        1 ----- 5 ----- 2


    Parameters
    ----------
    nx : int
        Number of Q8 elements along the beam length.

    length : float
        Beam length.

    height : float
        Beam height.

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Q8 element connectivity using 1-based numbering.
    """

    # --------------------------------------------------------
    # Input checks
    # --------------------------------------------------------

    if not isinstance(nx, int):
        raise TypeError(
            "nx must be an integer."
        )

    if nx < 1:
        raise ValueError(
            "nx must be at least 1."
        )

    if length <= 0.0:
        raise ValueError(
            "Beam length must be positive."
        )

    if height <= 0.0:
        raise ValueError(
            "Beam height must be positive."
        )


    # --------------------------------------------------------
    # Element dimensions
    # --------------------------------------------------------

    dx = (
        length / nx
    )

    y_bottom = 0.0
    y_top = height
    y_mid = 0.5 * height


    # --------------------------------------------------------
    # Node storage
    # --------------------------------------------------------

    coords = []

    node_map = {}


    def add_node(x, y):
        """
        Add a node only if it does not already exist.
        """

        key = (
            round(float(x), 12),
            round(float(y), 12)
        )

        if key not in node_map:

            node_map[key] = (
                len(coords) + 1
            )

            coords.append([
                float(x),
                float(y)
            ])

        return node_map[key]


    # --------------------------------------------------------
    # Connectivity
    # --------------------------------------------------------

    connectivity = []


    for ele in range(nx):

        x1 = ele * dx
        x2 = (ele + 1) * dx

        x_mid = (
            0.5 * (x1 + x2)
        )


        # ----------------------------------------------------
        # Corner nodes
        # ----------------------------------------------------

        n1 = add_node(
            x1,
            y_bottom
        )

        n2 = add_node(
            x2,
            y_bottom
        )

        n3 = add_node(
            x2,
            y_top
        )

        n4 = add_node(
            x1,
            y_top
        )


        # ----------------------------------------------------
        # Mid-side nodes
        # ----------------------------------------------------

        n5 = add_node(
            x_mid,
            y_bottom
        )

        n6 = add_node(
            x2,
            y_mid
        )

        n7 = add_node(
            x_mid,
            y_top
        )

        n8 = add_node(
            x1,
            y_mid
        )


        # ----------------------------------------------------
        # Q8 connectivity
        # ----------------------------------------------------

        connectivity.append([
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8
        ])


    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    Coord = np.array(
        coords,
        dtype=float
    )

    Connectivity = np.array(
        connectivity,
        dtype=int
    )


    return (
        Coord,
        Connectivity
    )