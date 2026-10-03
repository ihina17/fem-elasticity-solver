"""
Q4 Mesh Generator

Structured four-node quadrilateral mesh for rectangular domains.

Used for the cantilever beam comparison study.
"""

import numpy as np


# ============================================================
# Q4 rectangular mesh
# ============================================================

def generate_rectangular_q4_mesh(
    nx,
    ny,
    length=1.0,
    height=0.1
):
    """
    Generate a structured Q4 mesh for a rectangular domain.

    Q4 local node numbering:

        4 -------- 3
        |          |
        |          |
        1 -------- 2


    Parameters
    ----------
    nx : int
        Number of elements along the x direction.

    ny : int
        Number of elements along the y direction.

    length : float
        Length of the rectangular domain.

    height : float
        Height of the rectangular domain.

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Q4 element connectivity using 1-based numbering.
    """

    # --------------------------------------------------------
    # Input checks
    # --------------------------------------------------------

    if not isinstance(nx, int):
        raise TypeError(
            "nx must be an integer."
        )

    if not isinstance(ny, int):
        raise TypeError(
            "ny must be an integer."
        )

    if nx < 1:
        raise ValueError(
            "nx must be at least 1."
        )

    if ny < 1:
        raise ValueError(
            "ny must be at least 1."
        )

    if length <= 0.0:
        raise ValueError(
            "length must be positive."
        )

    if height <= 0.0:
        raise ValueError(
            "height must be positive."
        )


    # --------------------------------------------------------
    # Nodal coordinates
    # --------------------------------------------------------

    x_values = np.linspace(
        0.0,
        length,
        nx + 1
    )

    y_values = np.linspace(
        0.0,
        height,
        ny + 1
    )


    Coord = []


    for j in range(
        ny + 1
    ):

        for i in range(
            nx + 1
        ):

            Coord.append([
                x_values[i],
                y_values[j]
            ])


    Coord = np.array(
        Coord,
        dtype=float
    )


    # --------------------------------------------------------
    # Helper function for global node number
    #
    # Node numbering is 1-based.
    # --------------------------------------------------------

    def node_number(i, j):

        return (
            j * (nx + 1)
            + i
            + 1
        )


    # --------------------------------------------------------
    # Connectivity
    # --------------------------------------------------------

    Connectivity = []


    for j in range(ny):

        for i in range(nx):

            n1 = node_number(
                i,
                j
            )

            n2 = node_number(
                i + 1,
                j
            )

            n3 = node_number(
                i + 1,
                j + 1
            )

            n4 = node_number(
                i,
                j + 1
            )


            Connectivity.append([
                n1,
                n2,
                n3,
                n4
            ])


    Connectivity = np.array(
        Connectivity,
        dtype=int
    )


    return (
        Coord,
        Connectivity
    )