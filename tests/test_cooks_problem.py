import numpy as np

from src.meshes.cooks_Q4 import (
    generate_cooks_q4_mesh,
)

from problems.cooks_problem import (
    V,
    create_cooks_constraints,
    create_cooks_edge_loads,
    total_applied_force,
    solve_cooks_problem,
    get_tip_A_displacement,
)


# ============================================================
# Left boundary constraints
# ============================================================

def test_cooks_constraints():

    Coord, _ = (
        generate_cooks_q4_mesh(
            nx=4,
            ny=4,
        )
    )


    Constraints = (
        create_cooks_constraints(
            Coord
        )
    )


    # Five left-boundary nodes,
    # two displacement constraints per node.
    assert Constraints.shape == (
        10,
        3,
    )


# ============================================================
# Correct total boundary load
# ============================================================

def test_cooks_total_edge_load():

    Coord, _ = (
        generate_cooks_q4_mesh(
            nx=4,
            ny=4,
        )
    )


    NodalLoads = (
        create_cooks_edge_loads(
            Coord
        )
    )


    force = (
        total_applied_force(
            NodalLoads
        )
    )


    assert np.isclose(
        force[0],
        0.0,
    )


    assert np.isclose(
        force[1],
        V,
    )


# ============================================================
# FEM solution
# ============================================================

def test_cooks_solution_is_finite():

    (
        Coord,
        Connectivity,
        U,
        NodalLoads,
    ) = solve_cooks_problem(
        nx=2,
        ny=2,
    )


    assert np.all(
        np.isfinite(
            U
        )
    )


# ============================================================
# Upward tip displacement
# ============================================================

def test_cooks_tip_moves_upward():

    (
        Coord,
        Connectivity,
        U,
        NodalLoads,
    ) = solve_cooks_problem(
        nx=4,
        ny=4,
    )


    ux_A, uy_A = (
        get_tip_A_displacement(
            Coord,
            U,
        )
    )


    assert uy_A > 0.0