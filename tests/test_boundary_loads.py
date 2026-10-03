import numpy as np

from src.physics_models.boundary_loads import (
    create_constant_edge_traction_loads,
)

from src.physics_models.elasticity_driver import (
    Apply_NodalLoads,
)


# ============================================================
# Single edge segment
# ============================================================

def test_constant_traction_single_segment():

    Coord = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
    ])


    loads = (
        create_constant_edge_traction_loads(
            Coord=Coord,
            edge_nodes=np.array([
                1,
                2,
            ]),
            traction=np.array([
                0.0,
                100.0,
            ]),
            thickness=1.0,
        )
    )


    # Edge length = 2
    #
    # Total vertical force = 100 * 2 = 200
    #
    # Each endpoint receives 100.

    expected = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 100.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 100.0],
    ])


    assert np.allclose(
        loads,
        expected,
    )


# ============================================================
# Two boundary segments
# ============================================================

def test_constant_traction_multiple_segments():

    Coord = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
    ])


    loads = (
        create_constant_edge_traction_loads(
            Coord=Coord,
            edge_nodes=np.array([
                1,
                2,
                3,
            ]),
            traction=np.array([
                0.0,
                100.0,
            ]),
            thickness=1.0,
        )
    )


    # Vertical nodal forces should be:
    #
    # node 1 : 50
    # node 2 : 100
    # node 3 : 50

    vertical_loads = loads[
        loads[:, 1] == 2,
        2,
    ]


    assert np.allclose(
        vertical_loads,
        np.array([
            50.0,
            100.0,
            50.0,
        ]),
    )


    assert np.isclose(
        np.sum(
            vertical_loads
        ),
        200.0,
    )


# ============================================================
# Apply nodal loads to equation vectors
# ============================================================

def test_apply_nodal_loads():

    # Node 1 is prescribed.
    # Nodes 2 and 3 are free.

    GlobalID = np.array([
        [-1, -2],
        [ 1,  2],
        [ 3,  4],
    ])


    R_F = np.zeros(
        (
            4,
            1,
        )
    )


    R_P = np.zeros(
        (
            2,
            1,
        )
    )


    NodalLoads = np.array([
        [2, 2, 10.0],
        [3, 1, -2.0],
        [1, 1, 5.0],
    ])


    R_F, R_P = Apply_NodalLoads(
        NodalLoads,
        GlobalID,
        R_F,
        R_P,
    )


    assert np.isclose(
        R_F[
            1,
            0
        ],
        10.0,
    )


    assert np.isclose(
        R_F[
            2,
            0
        ],
        -2.0,
    )


    assert np.isclose(
        R_P[
            0,
            0
        ],
        5.0,
    )