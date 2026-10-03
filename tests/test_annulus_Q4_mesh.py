import numpy as np

from src.meshes.annulus_Q4 import (
    generate_annulus_q4_mesh
)


# ============================================================
# Basic mesh size
# ============================================================

def test_annulus_q4_mesh_size():

    Coord, Connectivity = (
        generate_annulus_q4_mesh(
            n_radial=2,
            n_theta=8,
            inner_radius=0.1,
            outer_radius=1.0,
        )
    )

    # Nodes:
    # (n_radial + 1) * n_theta
    assert Coord.shape == (
        24,
        2,
    )

    # Elements:
    # n_radial * n_theta
    assert Connectivity.shape == (
        16,
        4,
    )


# ============================================================
# Inner and outer radii
# ============================================================

def test_annulus_q4_radii():

    Coord, Connectivity = (
        generate_annulus_q4_mesh(
            n_radial=3,
            n_theta=12,
            inner_radius=0.1,
            outer_radius=1.0,
        )
    )

    radii = np.sqrt(
        Coord[:, 0]**2
        + Coord[:, 1]**2
    )

    assert np.isclose(
        radii.min(),
        0.1,
    )

    assert np.isclose(
        radii.max(),
        1.0,
    )


# ============================================================
# Periodic circumferential connectivity
# ============================================================

def test_annulus_q4_periodic_connectivity():

    n_theta = 8

    Coord, Connectivity = (
        generate_annulus_q4_mesh(
            n_radial=1,
            n_theta=n_theta,
            inner_radius=0.1,
            outer_radius=1.0,
        )
    )

    # Last element must connect back to the first
    # circumferential nodes.

    last_element = Connectivity[-1]

    assert last_element[0] == 8
    assert last_element[1] == 16
    assert last_element[2] == 9
    assert last_element[3] == 1


# ============================================================
# Element orientation
# ============================================================

def test_annulus_q4_positive_orientation():

    Coord, Connectivity = (
        generate_annulus_q4_mesh(
            n_radial=3,
            n_theta=24,
            inner_radius=0.1,
            outer_radius=1.0,
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
            v1[0] * v2[1]
            - v1[1] * v2[0]
        )

        assert cross > 0.0