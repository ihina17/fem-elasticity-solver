import numpy as np

from problems.thick_cylinder import (
    ri,
    ro,
    u0,
    analytical_constants,
    analytical_radial_displacement,
    analytical_strain,
    analytical_stress,
    analytical_displacement_xy,
)


# ============================================================
# Analytical constants
# ============================================================

def test_analytical_constants():

    A, B = analytical_constants()

    assert np.isclose(
        A,
        0.050505050505050504,
    )

    assert np.isclose(
        B,
        -0.0005050505050505052,
    )


# ============================================================
# Boundary conditions
# ============================================================

def test_radial_displacement_boundary_conditions():

    assert np.isclose(
        analytical_radial_displacement(
            ri
        ),
        0.0,
    )

    assert np.isclose(
        analytical_radial_displacement(
            ro
        ),
        u0,
    )


# ============================================================
# Strain relationship
# ============================================================

def test_strain_trace_is_constant():

    radii = np.array([
        0.1,
        0.2,
        0.5,
        1.0,
    ])

    epsilon_rr, epsilon_tt, _ = (
        analytical_strain(
            radii
        )
    )

    A, _ = analytical_constants()

    assert np.allclose(
        epsilon_rr
        + epsilon_tt,
        2.0 * A,
    )


# ============================================================
# Axisymmetric shear stress
# ============================================================

def test_shear_stress_is_zero():

    radii = np.array([
        0.1,
        0.5,
        1.0,
    ])

    _, _, sigma_rt = (
        analytical_stress(
            radii
        )
    )

    assert np.allclose(
        sigma_rt,
        0.0,
    )


# ============================================================
# Cartesian displacement
# ============================================================

def test_cartesian_displacement_on_x_axis():

    ux, uy = (
        analytical_displacement_xy(
            ro,
            0.0,
        )
    )

    assert np.isclose(
        ux,
        u0,
    )

    assert np.isclose(
        uy,
        0.0,
    )


def test_cartesian_displacement_on_y_axis():

    ux, uy = (
        analytical_displacement_xy(
            0.0,
            ro,
        )
    )

    assert np.isclose(
        ux,
        0.0,
    )

    assert np.isclose(
        uy,
        u0,
    )


from problems.thick_cylinder import (
    create_thick_cylinder_constraints,
    solve_thick_cylinder,
    fem_radial_displacement,
)

from src.meshes.annulus_Q4 import (
    generate_annulus_q4_mesh,
)


# ============================================================
# FEM constraints
# ============================================================

def test_thick_cylinder_constraints():

    Coord, _ = generate_annulus_q4_mesh(
        n_radial=2,
        n_theta=16,
        inner_radius=ri,
        outer_radius=ro,
    )

    Constraints = (
        create_thick_cylinder_constraints(
            Coord
        )
    )

    # 16 inner nodes + 16 outer nodes
    # 2 displacement constraints per node
    assert Constraints.shape == (
        64,
        3,
    )


# ============================================================
# FEM boundary conditions
# ============================================================

def test_thick_cylinder_fem_boundary_conditions():

    Coord, Connectivity, U = (
        solve_thick_cylinder(
            n_radial=2,
            n_theta=24,
        )
    )


    radii = np.sqrt(
        Coord[:, 0]**2
        + Coord[:, 1]**2
    )


    # --------------------------------------------------------
    # Inner boundary
    # --------------------------------------------------------

    inner_nodes = np.where(
        np.isclose(
            radii,
            ri
        )
    )[0]


    assert np.allclose(
        U[
            inner_nodes,
            :
        ],
        0.0
    )


    # --------------------------------------------------------
    # Outer boundary
    # --------------------------------------------------------

    outer_nodes = np.where(
        np.isclose(
            radii,
            ro
        )
    )[0]


    ur = fem_radial_displacement(
        Coord,
        U
    )


    assert np.allclose(
        ur[
            outer_nodes
        ],
        u0
    )


# ============================================================
# FEM solution is finite
# ============================================================

def test_thick_cylinder_fem_solution_is_finite():

    Coord, Connectivity, U = (
        solve_thick_cylinder(
            n_radial=3,
            n_theta=24,
        )
    )

    assert np.all(
        np.isfinite(
            U
        )
    )