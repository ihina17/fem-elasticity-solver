"""
Modified Lame Problem: Thick Cylinder

A two-dimensional annular elastic body is subjected to prescribed
radial displacement.

Boundary conditions:

    u = 0                       at r = ri

    ur = u0
    utheta = 0                  at r = ro

The material is modeled using plane-stress linearized elasticity.

The script includes:

    1. Analytical displacement solution
    2. Analytical strain and stress
    3. Q4 finite element solution
    4. FEM versus analytical displacement comparison
    5. Total strain energy
    6. h-refinement convergence study
    7. Observed convergence rates
    8. Mesh and displacement plots
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.fem.gauss_quadrature import GaussPoints
from src.fem.shape_functions import ShapeFunctions

from src.meshes.annulus_Q4 import (
    generate_annulus_q4_mesh,
)

from src.physics_models.elasticity_driver import (
    Driver_LE,
)

from src.physics_models.elasticity_kernel import (
    CalculateLocalMatrices,
)


# ============================================================
# Output directory
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# Problem parameters
# ============================================================

ri = 0.1
ro = 1.0

u0 = 0.05

E = 100000.0
nu = 0.3

thickness = 1.0


# ============================================================
# FEM settings
# ============================================================

dim = 2
dofs_per_node = 2


# ============================================================
# Material properties
# ============================================================

medium_set = {
    "type": "elastic_moduli",
    "E": E,
    "nu": nu,
    "analysis_type": "plane_stress",
    "thickness": thickness,
}


# ============================================================
# Body force
# ============================================================

load_type = {
    "type": "none",
}


# ============================================================
# Analytical constants
# ============================================================

def analytical_constants():
    """
    Return constants A and B in the analytical solution

        ur(r) = A r + B/r.
    """

    denominator = (
        ro**2
        - ri**2
    )

    A = (
        u0
        * ro
        / denominator
    )

    B = (
        -u0
        * ro
        * ri**2
        / denominator
    )

    return (
        A,
        B,
    )


# ============================================================
# Analytical radial displacement
# ============================================================

def analytical_radial_displacement(r):
    """
    Analytical radial displacement

        ur(r) = A r + B/r.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    if np.any(
        r <= 0.0
    ):
        raise ValueError(
            "Radius must be positive."
        )

    A, B = analytical_constants()

    return (
        A * r
        + B / r
    )


# ============================================================
# Analytical strain
# ============================================================

def analytical_strain(r):
    """
    Return the axisymmetric strain components

        epsilon_rr
        epsilon_tt
        gamma_rt
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    if np.any(
        r <= 0.0
    ):
        raise ValueError(
            "Radius must be positive."
        )

    A, B = analytical_constants()


    # --------------------------------------------------------
    # Radial strain
    #
    # epsilon_rr = dur/dr
    # --------------------------------------------------------

    epsilon_rr = (
        A
        - B / r**2
    )


    # --------------------------------------------------------
    # Hoop strain
    #
    # epsilon_tt = ur/r
    # --------------------------------------------------------

    epsilon_tt = (
        A
        + B / r**2
    )


    # --------------------------------------------------------
    # Axisymmetry gives zero shear strain
    # --------------------------------------------------------

    gamma_rt = np.zeros_like(
        r,
        dtype=float,
    )


    return (
        epsilon_rr,
        epsilon_tt,
        gamma_rt,
    )


# ============================================================
# Analytical stress
# ============================================================

def analytical_stress(r):
    """
    Return the plane-stress components

        sigma_rr
        sigma_tt
        sigma_rt
    """

    (
        epsilon_rr,
        epsilon_tt,
        gamma_rt,
    ) = analytical_strain(
        r
    )


    # --------------------------------------------------------
    # Plane-stress constitutive factor
    # --------------------------------------------------------

    factor = (
        E
        / (
            1.0
            - nu**2
        )
    )


    # --------------------------------------------------------
    # Radial normal stress
    # --------------------------------------------------------

    sigma_rr = (
        factor
        * (
            epsilon_rr
            + nu * epsilon_tt
        )
    )


    # --------------------------------------------------------
    # Hoop normal stress
    # --------------------------------------------------------

    sigma_tt = (
        factor
        * (
            epsilon_tt
            + nu * epsilon_rr
        )
    )


    # --------------------------------------------------------
    # Shear stress
    # --------------------------------------------------------

    sigma_rt = (
        E
        / (
            2.0
            * (
                1.0
                + nu
            )
        )
        * gamma_rt
    )


    return (
        sigma_rr,
        sigma_tt,
        sigma_rt,
    )


# ============================================================
# Analytical Cartesian displacement
# ============================================================

def analytical_displacement_xy(
    x,
    y,
):
    """
    Convert the analytical radial displacement into
    Cartesian displacement components.

        ux = ur x/r
        uy = ur y/r
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )


    r = np.sqrt(
        x**2
        + y**2
    )


    if np.any(
        r <= 0.0
    ):
        raise ValueError(
            "Analytical solution is undefined at r = 0."
        )


    ur = analytical_radial_displacement(
        r
    )


    ux = (
        ur
        * x
        / r
    )

    uy = (
        ur
        * y
        / r
    )


    return (
        ux,
        uy,
    )


# ============================================================
# Analytical total strain energy
# ============================================================

def analytical_strain_energy():
    """
    Analytical total strain energy of the annulus
    under plane stress.

    Unit out-of-plane thickness is used.
    """

    A, B = analytical_constants()


    term_1 = (
        A**2
        * (
            ro**2
            - ri**2
        )
        / (
            1.0
            - nu
        )
    )


    term_2 = (
        B**2
        * (
            1.0 / ri**2
            - 1.0 / ro**2
        )
        / (
            1.0
            + nu
        )
    )


    energy = (
        np.pi
        * thickness
        * E
        * (
            term_1
            + term_2
        )
    )


    return float(
        energy
    )


# ============================================================
# Boundary conditions
# ============================================================

def create_thick_cylinder_constraints(
    Coord,
):
    """
    Prescribe the displacement boundary conditions.

    Inner boundary:

        ux = 0
        uy = 0

    Outer boundary:

        ur = u0
        utheta = 0

    The outer radial displacement is converted to
    Cartesian components:

        ux = u0 x/r
        uy = u0 y/r
    """

    Constraints = []


    for node in range(
        Coord.shape[0]
    ):

        x = Coord[
            node,
            0
        ]

        y = Coord[
            node,
            1
        ]


        r = np.sqrt(
            x**2
            + y**2
        )


        # ----------------------------------------------------
        # Inner boundary
        # ----------------------------------------------------

        if np.isclose(
            r,
            ri
        ):

            Constraints.append([
                node + 1,
                1,
                0.0,
            ])

            Constraints.append([
                node + 1,
                2,
                0.0,
            ])


        # ----------------------------------------------------
        # Outer boundary
        # ----------------------------------------------------

        elif np.isclose(
            r,
            ro
        ):

            ux = (
                u0
                * x
                / r
            )

            uy = (
                u0
                * y
                / r
            )


            Constraints.append([
                node + 1,
                1,
                ux,
            ])

            Constraints.append([
                node + 1,
                2,
                uy,
            ])


    return np.array(
        Constraints,
        dtype=float,
    )


# ============================================================
# FEM solution
# ============================================================

def solve_thick_cylinder(
    n_radial,
    n_theta,
):
    """
    Solve the modified Lame problem using Q4 elements.
    """

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    Coord, Connectivity = (
        generate_annulus_q4_mesh(
            n_radial=n_radial,
            n_theta=n_theta,
            inner_radius=ri,
            outer_radius=ro,
        )
    )


    NumNodes = Coord.shape[0]

    Nele = Connectivity.shape[0]


    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    Constraints = (
        create_thick_cylinder_constraints(
            Coord
        )
    )


    NCons = Constraints.shape[0]


    # --------------------------------------------------------
    # Q4 quadrature
    # --------------------------------------------------------

    NGPTS = 2


    # --------------------------------------------------------
    # FEM solution
    # --------------------------------------------------------

    U = Driver_LE(
        Connectivity,
        Constraints,
        Coord,
        medium_set,
        dim,
        dofs_per_node,
        "Q4",
        load_type,
        NCons,
        Nele,
        NGPTS,
        NumNodes,
    )


    return (
        Coord,
        Connectivity,
        U,
    )


# ============================================================
# FEM radial displacement
# ============================================================

def fem_radial_displacement(
    Coord,
    U,
):
    """
    Convert Cartesian FEM displacement to radial displacement.

        ur = ux x/r + uy y/r
    """

    x = Coord[
        :,
        0
    ]

    y = Coord[
        :,
        1
    ]


    r = np.sqrt(
        x**2
        + y**2
    )


    ux = U[
        :,
        0
    ]

    uy = U[
        :,
        1
    ]


    ur = (
        ux
        * x
        / r
        + uy
        * y
        / r
    )


    return ur


# ============================================================
# Nodal displacement error
# ============================================================

def compute_displacement_error(
    Coord,
    U,
):
    """
    Compute relative nodal displacement error.

        ||U_FEM - U_exact||_2
        ---------------------
             ||U_exact||_2
    """

    ux_exact, uy_exact = (
        analytical_displacement_xy(
            Coord[:, 0],
            Coord[:, 1],
        )
    )


    U_exact = np.column_stack([
        ux_exact,
        uy_exact,
    ])


    numerator = np.linalg.norm(
        U
        - U_exact
    )


    denominator = np.linalg.norm(
        U_exact
    )


    if np.isclose(
        denominator,
        0.0
    ):
        raise ValueError(
            "Analytical displacement norm is zero."
        )


    return float(
        numerator
        / denominator
    )


# ============================================================
# Radial displacement data
# ============================================================

def radial_displacement_data(
    Coord,
    U,
):
    """
    Return radius and FEM radial displacement,
    sorted by radius.
    """

    radius = np.sqrt(
        Coord[:, 0]**2
        + Coord[:, 1]**2
    )


    ur_fem = fem_radial_displacement(
        Coord,
        U,
    )


    order = np.argsort(
        radius
    )


    return (
        radius[order],
        ur_fem[order],
    )


# ============================================================
# Integrated L2 displacement error
# ============================================================

def compute_l2_displacement_error(
    Coord,
    Connectivity,
    U,
):
    """
    Compute the relative L2 displacement error using
    Gaussian integration over the domain.

        ||u_h - u_exact||_L2
        --------------------
            ||u_exact||_L2
    """

    r_gauss, w_gauss = GaussPoints(
        dim=2,
        EleType="Q4",
        NGPTS=2,
    )


    numerator = 0.0

    denominator = 0.0


    for element in Connectivity:

        ids = (
            element
            - 1
        )


        xCap = Coord[
            ids,
            :
        ]


        UCap = U[
            ids,
            :
        ]


        for gpt in range(
            len(w_gauss)
        ):

            zeta = r_gauss[
                gpt,
                :
            ]


            # ------------------------------------------------
            # Shape functions
            # ------------------------------------------------

            N, DN = ShapeFunctions(
                "Q4",
                zeta,
            )


            # ------------------------------------------------
            # Gauss-point physical coordinates
            # ------------------------------------------------

            x = (
                N
                @ xCap
            ).reshape(-1)


            # ------------------------------------------------
            # Jacobian
            # ------------------------------------------------

            J = (
                DN.T
                @ xCap
            )


            detJ = float(
                np.linalg.det(
                    J
                )
            )


            if detJ <= 0.0:

                raise ValueError(
                    "Element Jacobian must be positive."
                )


            # ------------------------------------------------
            # FEM displacement
            # ------------------------------------------------

            u_fem = (
                N
                @ UCap
            ).reshape(-1)


            # ------------------------------------------------
            # Analytical displacement
            # ------------------------------------------------

            ux_exact, uy_exact = (
                analytical_displacement_xy(
                    x[0],
                    x[1],
                )
            )


            u_exact = np.array([
                float(ux_exact),
                float(uy_exact),
            ])


            difference = (
                u_fem
                - u_exact
            )


            weight = (
                w_gauss[gpt]
                * detJ
                * thickness
            )


            numerator += (
                difference
                @ difference
                * weight
            )


            denominator += (
                u_exact
                @ u_exact
                * weight
            )


    if denominator <= 0.0:

        raise ValueError(
            "Analytical displacement L2 norm is zero."
        )


    return float(
        np.sqrt(
            numerator
            / denominator
        )
    )


# ============================================================
# FEM total strain energy
# ============================================================

def compute_fem_strain_energy(
    Coord,
    Connectivity,
    U,
):
    """
    Compute total finite element strain energy

        U = 1/2 sum_e ue^T Ke ue.
    """

    r_gauss, w_gauss = GaussPoints(
        dim=2,
        EleType="Q4",
        NGPTS=2,
    )


    total_energy = 0.0


    for element in Connectivity:

        ids = (
            element
            - 1
        )


        xCap = Coord[
            ids,
            :
        ]


        EleNodes = element


        # ----------------------------------------------------
        # Element stiffness matrix
        # ----------------------------------------------------

        KLocal, _ = CalculateLocalMatrices(
            medium_set,
            dofs_per_node,
            EleNodes,
            "Q4",
            load_type,
            r_gauss,
            w_gauss,
            xCap,
        )


        # ----------------------------------------------------
        # Element displacement vector
        #
        # [u1, v1, u2, v2, ...]
        # ----------------------------------------------------

        ULocal = U[
            ids,
            :
        ].reshape(-1)


        # ----------------------------------------------------
        # Element strain energy
        # ----------------------------------------------------

        element_energy = (
            0.5
            * ULocal
            @ KLocal
            @ ULocal
        )


        total_energy += (
            element_energy
        )


    return float(
        total_energy
    )


# ============================================================
# Convergence study
# ============================================================

def run_convergence_study(
    radial_meshes=(
        2,
        4,
        8,
        16,
    )
):
    """
    Perform an h-refinement study.

    Circumferential refinement is chosen as

        n_theta = 8 * n_radial.

    Characteristic radial element size is

        h = (ro - ri) / n_radial.
    """

    exact_energy = (
        analytical_strain_energy()
    )


    results = {
        "n_radial": [],
        "n_theta": [],
        "h": [],
        "error": [],
        "energy": [],
        "energy_error": [],
    }


    print()
    print("Thick-cylinder convergence study")
    print("--------------------------------")
    print()


    print(
        f"{'Nr':>6} "
        f"{'Ntheta':>8} "
        f"{'h':>12} "
        f"{'L2 error':>16} "
        f"{'Strain energy':>18} "
        f"{'Energy error':>16}"
    )


    for n_radial in radial_meshes:

        n_theta = (
            8
            * n_radial
        )


        # ----------------------------------------------------
        # FEM solution
        # ----------------------------------------------------

        Coord, Connectivity, U = (
            solve_thick_cylinder(
                n_radial=n_radial,
                n_theta=n_theta,
            )
        )


        # ----------------------------------------------------
        # Characteristic mesh size
        # ----------------------------------------------------

        h = (
            ro
            - ri
        ) / n_radial


        # ----------------------------------------------------
        # L2 displacement error
        # ----------------------------------------------------

        error = (
            compute_l2_displacement_error(
                Coord,
                Connectivity,
                U,
            )
        )


        # ----------------------------------------------------
        # Strain energy
        # ----------------------------------------------------

        energy = (
            compute_fem_strain_energy(
                Coord,
                Connectivity,
                U,
            )
        )


        energy_error = (
            abs(
                energy
                - exact_energy
            )
            / abs(
                exact_energy
            )
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results[
            "n_radial"
        ].append(
            n_radial
        )

        results[
            "n_theta"
        ].append(
            n_theta
        )

        results[
            "h"
        ].append(
            h
        )

        results[
            "error"
        ].append(
            error
        )

        results[
            "energy"
        ].append(
            energy
        )

        results[
            "energy_error"
        ].append(
            energy_error
        )


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"{n_radial:6d} "
            f"{n_theta:8d} "
            f"{h:12.6e} "
            f"{error:16.8e} "
            f"{energy:18.8e} "
            f"{energy_error:16.8e}"
        )


    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    for key in results:

        results[key] = np.array(
            results[key]
        )


    # --------------------------------------------------------
    # Observed convergence rates
    # --------------------------------------------------------

    rates = np.full(
        len(
            results["error"]
        ),
        np.nan,
    )


    energy_rates = np.full(
        len(
            results["energy_error"]
        ),
        np.nan,
    )


    for i in range(
        1,
        len(
            results["error"]
        ),
    ):

        rates[i] = (
            np.log(
                results["error"][i - 1]
                / results["error"][i]
            )
            / np.log(
                results["h"][i - 1]
                / results["h"][i]
            )
        )


        energy_rates[i] = (
            np.log(
                results["energy_error"][i - 1]
                / results["energy_error"][i]
            )
            / np.log(
                results["h"][i - 1]
                / results["h"][i]
            )
        )


    results[
        "rate"
    ] = rates


    results[
        "energy_rate"
    ] = energy_rates


    # --------------------------------------------------------
    # Print convergence rates
    # --------------------------------------------------------

    print()
    print("Observed convergence rates")
    print("--------------------------")


    print(
        f"{'Nr':>6} "
        f"{'L2 rate':>14} "
        f"{'Energy rate':>14}"
    )


    for i in range(
        len(
            results["n_radial"]
        )
    ):

        if i == 0:

            print(
                f"{int(results['n_radial'][i]):6d} "
                f"{'---':>14} "
                f"{'---':>14}"
            )

        else:

            print(
                f"{int(results['n_radial'][i]):6d} "
                f"{results['rate'][i]:14.6f} "
                f"{results['energy_rate'][i]:14.6f}"
            )


    return results


# ============================================================
# Print analytical information
# ============================================================

def print_analytical_solution():
    """
    Print the analytical solution parameters and
    verify the prescribed radial displacement.
    """

    A, B = analytical_constants()


    print()
    print("Modified Lame thick-cylinder problem")
    print("------------------------------------")


    print(
        f"Inner radius     = "
        f"{ri:.6f}"
    )


    print(
        f"Outer radius     = "
        f"{ro:.6f}"
    )


    print(
        f"Outer displacement = "
        f"{u0:.6f}"
    )


    print(
        f"Young's modulus  = "
        f"{E:.6f}"
    )


    print(
        f"Poisson's ratio  = "
        f"{nu:.6f}"
    )


    print()
    print("Analytical constants")


    print(
        f"A = "
        f"{A:.10e}"
    )


    print(
        f"B = "
        f"{B:.10e}"
    )


    print()
    print("Boundary check")


    print(
        f"ur(ri) = "
        f"{analytical_radial_displacement(ri):.10e}"
    )


    print(
        f"ur(ro) = "
        f"{analytical_radial_displacement(ro):.10e}"
    )


# ============================================================
# Radial displacement plot
# ============================================================

def plot_radial_displacement(
    Coord,
    U,
):
    """
    Compare FEM radial displacement with the analytical
    radial displacement.
    """

    radius_fem, ur_fem = (
        radial_displacement_data(
            Coord,
            U,
        )
    )


    radius_exact = np.linspace(
        ri,
        ro,
        400,
    )


    ur_exact = (
        analytical_radial_displacement(
            radius_exact
        )
    )


    plt.figure()


    plt.plot(
        radius_exact,
        ur_exact,
        label="Analytical",
    )


    plt.plot(
        radius_fem,
        ur_fem,
        "o",
        markersize=3,
        label="Q4 FEM",
    )


    plt.xlabel(
        "Radius, r"
    )


    plt.ylabel(
        "Radial displacement, $u_r$"
    )


    plt.title(
        "Thick Cylinder Radial Displacement"
    )


    plt.legend()


    plt.grid(
        True
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "thick_cylinder_radial_displacement.png",
        dpi=300,
    )


# ============================================================
# Annulus mesh plot
# ============================================================

def plot_annulus_mesh(
    Coord,
    Connectivity,
    U=None,
    scale=1.0,
    filename="thick_cylinder_mesh.png",
):
    """
    Plot the annular finite element mesh.

    If U is supplied, both the undeformed and scaled
    deformed meshes are shown.
    """

    plt.figure()


    order = [
        0,
        1,
        2,
        3,
        0,
    ]


    # --------------------------------------------------------
    # Original mesh
    # --------------------------------------------------------

    for element in Connectivity:

        ids = (
            element
            - 1
        )


        x = Coord[
            ids[order],
            0
        ]


        y = Coord[
            ids[order],
            1
        ]


        plt.plot(
            x,
            y,
            linestyle="--",
        )


    # --------------------------------------------------------
    # Deformed mesh
    # --------------------------------------------------------

    if U is not None:

        deformed_coord = (
            Coord
            + scale * U
        )


        for element in Connectivity:

            ids = (
                element
                - 1
            )


            x = deformed_coord[
                ids[order],
                0
            ]


            y = deformed_coord[
                ids[order],
                1
            ]


            plt.plot(
                x,
                y,
            )


    plt.xlabel(
        "x"
    )


    plt.ylabel(
        "y"
    )


    if U is None:

        plt.title(
            "Thick Cylinder Q4 Mesh"
        )

    else:

        plt.title(
            f"Thick Cylinder Deformed Mesh "
            f"(scale = {scale:g})"
        )


    plt.axis(
        "equal"
    )


    plt.grid(
        True
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / filename,
        dpi=300,
    )


# ============================================================
# Convergence plot
# ============================================================

def plot_convergence(
    results,
):
    """
    Plot relative L2 displacement error versus mesh size.
    """

    plt.figure()


    plt.loglog(
        results["h"],
        results["error"],
        "o-",
        label="Q4 FEM",
    )


    plt.xlabel(
        "Element size, h"
    )


    plt.ylabel(
        "Relative L2 displacement error"
    )


    plt.title(
        "Thick Cylinder Mesh Convergence"
    )


    plt.legend()


    plt.grid(
        True,
        which="both",
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "thick_cylinder_convergence.png",
        dpi=300,
    )


# ============================================================
# Strain energy plot
# ============================================================

def plot_strain_energy(
    results,
):
    """
    Plot FEM total strain energy versus element size.
    """

    exact_energy = (
        analytical_strain_energy()
    )


    plt.figure()


    plt.plot(
        results["h"],
        results["energy"],
        "o-",
        label="Q4 FEM",
    )


    plt.axhline(
        exact_energy,
        linestyle="--",
        label="Analytical",
    )


    plt.xlabel(
        "Element size, h"
    )


    plt.ylabel(
        "Total strain energy"
    )


    plt.title(
        "Thick Cylinder Strain Energy"
    )


    plt.legend()


    plt.grid(
        True
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "thick_cylinder_strain_energy.png",
        dpi=300,
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Analytical solution
    # --------------------------------------------------------

    print_analytical_solution()


    # ========================================================
    # Representative FEM solution
    # ========================================================

    n_radial = 8
    n_theta = 64


    Coord, Connectivity, U = (
        solve_thick_cylinder(
            n_radial=n_radial,
            n_theta=n_theta,
        )
    )


    # --------------------------------------------------------
    # Nodal displacement error
    # --------------------------------------------------------

    relative_error = (
        compute_displacement_error(
            Coord,
            U,
        )
    )


    print()
    print("Finite element solution")
    print("-----------------------")


    print(
        f"Radial elements       = "
        f"{n_radial}"
    )


    print(
        f"Circumferential elems = "
        f"{n_theta}"
    )


    print(
        f"Number of nodes       = "
        f"{Coord.shape[0]}"
    )


    print(
        f"Number of elements    = "
        f"{Connectivity.shape[0]}"
    )


    print(
        f"Relative nodal error  = "
        f"{relative_error:.8e}"
    )


    # ========================================================
    # Strain energy
    # ========================================================

    fem_energy = (
        compute_fem_strain_energy(
            Coord,
            Connectivity,
            U,
        )
    )


    exact_energy = (
        analytical_strain_energy()
    )


    print()


    print(
        f"FEM strain energy       = "
        f"{fem_energy:.8e}"
    )


    print(
        f"Analytical strain energy = "
        f"{exact_energy:.8e}"
    )


    # ========================================================
    # Convergence study
    # ========================================================

    results = run_convergence_study(
        radial_meshes=(
            2,
            4,
            8,
            16,
        )
    )


    # ========================================================
    # Figures
    # ========================================================

    plot_radial_displacement(
        Coord,
        U,
    )


    plot_annulus_mesh(
        Coord,
        Connectivity,
        filename="thick_cylinder_mesh.png",
    )


    plot_annulus_mesh(
        Coord,
        Connectivity,
        U=U,
        scale=1.0,
        filename="thick_cylinder_deformed.png",
    )


    plot_convergence(
        results
    )


    plot_strain_energy(
        results
    )


    # ========================================================
    # Display
    # ========================================================

    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()