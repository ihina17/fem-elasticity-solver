import numpy as np

from src.meshes.Q8 import generate_cantilever_q8_mesh


def test_single_q8_element():

    Coord, Connectivity = (
        generate_cantilever_q8_mesh(
            nx=1,
            length=1.0,
            height=0.1
        )
    )

    # One Q8 element
    assert Connectivity.shape == (
        1,
        8
    )

    # Eight nodes
    assert Coord.shape == (
        8,
        2
    )

    # All node numbers should appear once
    assert np.array_equal(
        np.sort(
            Connectivity[0]
        ),
        np.arange(
            1,
            9
        )
    )


def test_multiple_q8_elements():

    Coord, Connectivity = (
        generate_cantilever_q8_mesh(
            nx=4,
            length=1.0,
            height=0.1
        )
    )

    assert Connectivity.shape == (
        4,
        8
    )

    # Beam should start at x = 0
    assert np.isclose(
        Coord[:, 0].min(),
        0.0
    )

    # Beam should end at x = 1
    assert np.isclose(
        Coord[:, 0].max(),
        1.0
    )

    # Beam depth
    assert np.isclose(
        Coord[:, 1].min(),
        0.0
    )

    assert np.isclose(
        Coord[:, 1].max(),
        0.1
    )