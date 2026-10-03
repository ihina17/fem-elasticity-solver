"""
Uniform L2 Mesh Generator
"""

import numpy as np


def generate_uniform_l2_mesh(
    length: float,
    n_elements: int
):
    """
    Generate a uniform one-dimensional mesh using
    two-node linear L2 elements.

    Parameters
    ----------
    length : float
        Total length of the domain.

    n_elements : int
        Number of finite elements.

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates with shape (NumNodes, 1).

    Connectivity : np.ndarray
        Element connectivity using 1-based node numbering.
    """

    # --------------------------------------------------------
    # Input checks
    # --------------------------------------------------------

    if length <= 0.0:
        raise ValueError(
            "Length must be positive."
        )

    if not isinstance(n_elements, int):
        raise TypeError(
            "n_elements must be an integer."
        )

    if n_elements < 1:
        raise ValueError(
            "n_elements must be at least 1."
        )


    # --------------------------------------------------------
    # Nodal coordinates
    # --------------------------------------------------------

    NumNodes = n_elements + 1

    Coord = np.linspace(
        0.0,
        length,
        NumNodes
    ).reshape(-1, 1)


    # --------------------------------------------------------
    # Element connectivity
    # --------------------------------------------------------

    Connectivity = np.zeros(
        (n_elements, 2),
        dtype=int
    )

    for ele in range(n_elements):

        Connectivity[ele, :] = [
            ele + 1,
            ele + 2
        ]


    return Coord, Connectivity