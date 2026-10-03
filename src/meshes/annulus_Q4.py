"""
Structured Q4 mesh generator for an annular domain.

The mesh is generated using radial and circumferential
divisions and uses 1-based element connectivity.
"""

import numpy as np


# ============================================================
# Annular Q4 mesh
# ============================================================

def generate_annulus_q4_mesh(
    n_radial,
    n_theta,
    inner_radius=0.1,
    outer_radius=1.0,
):
    """
    Generate a structured Q4 mesh for a complete annulus.

    Parameters
    ----------
    n_radial : int
        Number of elements in the radial direction.

    n_theta : int
        Number of elements around the circumference.

    inner_radius : float
        Inner radius of the annulus.

    outer_radius : float
        Outer radius of the annulus.

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates with shape (NumNodes, 2).

    Connectivity : np.ndarray
        Q4 connectivity using 1-based node numbering.

    Notes
    -----
    Local element ordering is

        4 -------- 3
        |          |
        |          |
        1 -------- 2

    mapped onto an annular sector as

        n4 ----- n3
        /         /
       /         /
      n1 ----- n2

    where nodes 1 and 4 lie on the inner radial side
    of each element and nodes 2 and 3 lie on the
    outer radial side.
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not isinstance(
        n_radial,
        int,
    ):
        raise TypeError(
            "n_radial must be an integer."
        )

    if not isinstance(
        n_theta,
        int,
    ):
        raise TypeError(
            "n_theta must be an integer."
        )

    if n_radial < 1:
        raise ValueError(
            "n_radial must be at least 1."
        )

    if n_theta < 3:
        raise ValueError(
            "n_theta must be at least 3."
        )

    if inner_radius <= 0.0:
        raise ValueError(
            "inner_radius must be positive."
        )

    if outer_radius <= inner_radius:
        raise ValueError(
            "outer_radius must be greater than inner_radius."
        )


    # --------------------------------------------------------
    # Radial coordinates
    # --------------------------------------------------------

    radial_values = np.linspace(
        inner_radius,
        outer_radius,
        n_radial + 1,
    )


    # --------------------------------------------------------
    # Angular coordinates
    #
    # Do not repeat theta = 2*pi because it is the same
    # physical location as theta = 0.
    # --------------------------------------------------------

    theta_values = np.linspace(
        0.0,
        2.0 * np.pi,
        n_theta,
        endpoint=False,
    )


    # --------------------------------------------------------
    # Nodal coordinates
    #
    # Node numbering:
    # radial ring first, then angle.
    # --------------------------------------------------------

    Coord = []


    for radius in radial_values:

        for theta in theta_values:

            x = (
                radius
                * np.cos(theta)
            )

            y = (
                radius
                * np.sin(theta)
            )

            Coord.append([
                x,
                y,
            ])


    Coord = np.array(
        Coord,
        dtype=float,
    )


    # --------------------------------------------------------
    # Helper function
    #
    # Returns 1-based global node number.
    # --------------------------------------------------------

    def node_number(
        radial_index,
        theta_index,
    ):

        theta_index = (
            theta_index
            % n_theta
        )

        return (
            radial_index
            * n_theta
            + theta_index
            + 1
        )


    # --------------------------------------------------------
    # Element connectivity
    # --------------------------------------------------------

    Connectivity = []


    for j in range(
        n_radial
    ):

        for i in range(
            n_theta
        ):

            next_i = (
                i + 1
            ) % n_theta


            # Inner radius, theta_i
            n1 = node_number(
                j,
                i,
            )


            # Outer radius, theta_i
            n2 = node_number(
                j + 1,
                i,
            )


            # Outer radius, theta_(i+1)
            n3 = node_number(
                j + 1,
                next_i,
            )


            # Inner radius, theta_(i+1)
            n4 = node_number(
                j,
                next_i,
            )


            Connectivity.append([
                n1,
                n2,
                n3,
                n4,
            ])


    Connectivity = np.array(
        Connectivity,
        dtype=int,
    )


    return (
        Coord,
        Connectivity,
    )