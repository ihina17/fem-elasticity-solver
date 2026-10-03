import numpy as np

from src.meshes.L2 import generate_uniform_l2_mesh


def test_uniform_l2_mesh():

    Coord, Connectivity = generate_uniform_l2_mesh(
        length=1.0,
        n_elements=4
    )

    Coord_exact = np.array([
        [0.00],
        [0.25],
        [0.50],
        [0.75],
        [1.00]
    ])

    Connectivity_exact = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])

    assert np.allclose(
        Coord,
        Coord_exact
    )

    assert np.array_equal(
        Connectivity,
        Connectivity_exact
    )