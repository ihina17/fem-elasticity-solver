import numpy as np

from src.meshes.cooks_Q4 import (
    generate_cooks_q4_mesh,
)


# ============================================================
# Single element
# ============================================================

def test_single_cooks_q4_element():

    Coord, Connectivity = (
        generate_cooks_q4_mesh(
            nx=1,
            ny=1,
        )
    )


    assert Coord.shape == (
        4,
        2,
    )


    assert Connectivity.shape == (
        1,
        4,
    )


    expected_coord = np.array([
        [0.0, 0.0],
        [48.0, 44.0],
        [0.0, 44.0],
        [48.0, 60.0],
    ])


    assert np.allclose(
        Coord,
        expected_coord,
    )


    expected_connectivity = np.array([
        [1, 2, 4, 3]
    ])


    assert np.array_equal(
        Connectivity,
        expected_connectivity,
    )


# ============================================================
# Mesh dimensions
# ============================================================

def test_cooks_q4_mesh_size():

    Coord, Connectivity = (
        generate_cooks_q4_mesh(
            nx=4,
            ny=4,
        )
    )


    assert Coord.shape == (
        25,
        2,
    )


    assert Connectivity.shape == (
        16,
        4,
    )


# ============================================================
# Corner coordinates
# ============================================================

def test_cooks_q4_corner_coordinates():

    Coord, Connectivity = (
        generate_cooks_q4_mesh(
            nx=4,
            ny=4,
        )
    )


    corners = np.array([
        [0.0, 0.0],
        [0.0, 44.0],
        [48.0, 44.0],
        [48.0, 60.0],
    ])


    for corner in corners:

        distance = np.linalg.norm(
            Coord
            - corner,
            axis=1,
        )


        assert np.min(
            distance
        ) < 1e-12


# ============================================================
# Line BC
# ============================================================

def test_cooks_q4_line_bc():

    Coord, Connectivity = (
        generate_cooks_q4_mesh(
            nx=4,
            ny=4,
        )
    )


    # BC lies at x = 24
    bc_nodes = np.where(
        np.isclose(
            Coord[:, 0],
            24.0,
        )
    )[0]


    # ny + 1 nodes should lie on this line
    assert len(
        bc_nodes
    ) == 5


    # Lower intersection B
    assert np.isclose(
        Coord[
            bc_nodes,
            1
        ].min(),
        22.0,
    )


    # Upper intersection C
    assert np.isclose(
        Coord[
            bc_nodes,
            1
        ].max(),
        52.0,
    )


# ============================================================
# Positive element orientation
# ============================================================

def test_cooks_q4_positive_orientation():

    Coord, Connectivity = (
        generate_cooks_q4_mesh(
            nx=8,
            ny=8,
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


        v1 = (
            x[1]
            - x[0]
        )


        v2 = (
            x[3]
            - x[0]
        )


        cross = (
            v1[0]
            * v2[1]
            - v1[1]
            * v2[0]
        )


        assert cross > 0.0