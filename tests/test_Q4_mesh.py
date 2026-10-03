import numpy as np

from src.meshes.Q4 import generate_rectangular_q4_mesh


def test_single_q4_element():

    Coord, Connectivity = (
        generate_rectangular_q4_mesh(
            nx=1,
            ny=1,
            length=1.0,
            height=0.1
        )
    )

    assert Coord.shape == (
        4,
        2
    )

    assert Connectivity.shape == (
        1,
        4
    )

    expected_coord = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.1],
        [1.0, 0.1]
    ])

    assert np.allclose(
        Coord,
        expected_coord
    )

    expected_connectivity = np.array([
        [1, 2, 4, 3]
    ])

    assert np.array_equal(
        Connectivity,
        expected_connectivity
    )


def test_multiple_q4_elements():

    Coord, Connectivity = (
        generate_rectangular_q4_mesh(
            nx=4,
            ny=2,
            length=1.0,
            height=0.1
        )
    )

    # Number of nodes:
    # (nx + 1)(ny + 1)
    assert Coord.shape == (
        15,
        2
    )

    # Number of elements:
    # nx * ny
    assert Connectivity.shape == (
        8,
        4
    )

    assert np.isclose(
        Coord[:, 0].min(),
        0.0
    )

    assert np.isclose(
        Coord[:, 0].max(),
        1.0
    )

    assert np.isclose(
        Coord[:, 1].min(),
        0.0
    )

    assert np.isclose(
        Coord[:, 1].max(),
        0.1
    )


def test_q4_positive_orientation():

    Coord, Connectivity = (
        generate_rectangular_q4_mesh(
            nx=3,
            ny=2,
            length=1.0,
            height=0.1
        )
    )

    for element in Connectivity:

        ids = (
            element
            - 1
        )

        x = Coord[
            ids,
            :
        ]

        # Polygon orientation using the first three nodes
        v1 = (
            x[1]
            - x[0]
        )

        v2 = (
            x[2]
            - x[0]
        )

        cross = (
            v1[0] * v2[1]
            - v1[1] * v2[0]
        )

        assert cross > 0.0