"""
Deformation of a Rod Under Self-Weight

A one-dimensional hanging elastic rod is solved using
two-node linear finite elements (L2).

The problem includes:
    1. FEM displacement solution
    2. Comparison with the analytical displacement
    3. L2 and H1 convergence analysis
    4. Stress evaluated at Gauss points
    5. Stress evaluated at element nodes
    6. Recovered nodal stress using averaging
    7. Comparison with the analytical stress
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.meshes.L2 import generate_uniform_l2_mesh
from src.physics_models.elasticity_driver import Driver_LE
from src.error_calculation import Calculate_Error


# ============================================================
# Output directory
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# Problem parameters
# ============================================================

L = 1.0                 # Rod length (m)
A = 0.5                 # Cross-sectional area (m^2)

rho = 1500.0            # Density (kg/m^3)
g = 10.0                # Gravitational acceleration (m/s^2)

E = 250e6               # Young's modulus (Pa)
nu = 0.3                # Poisson's ratio


# ============================================================
# FEM settings
# ============================================================

dim = 1
dofs_per_node = 1

EleType = "L2"
NGPTS = 2


# ============================================================
# Material properties
# ============================================================

medium_set = {
    "type": "elastic_moduli",
    "E": E,
    "nu": nu,
    "A": A
}


# ============================================================
# Body force
# ============================================================

load_type = {
    "type": "gravity",
    "rho": rho,
    "g": g
}


# ============================================================
# Analytical solution
# ============================================================

def analytical_displacement(x):
    """
    Analytical displacement of the hanging rod.

    x is measured downward from the fixed support.
    """

    return (
        rho * g / E
        * (
            L * x
            - 0.5 * x**2
        )
    )


def analytical_gradient(x):
    """
    Analytical displacement gradient du/dx.
    """

    return np.array([
        rho * g / E
        * (L - x)
    ])


def analytical_stress(x):
    """
    Analytical axial stress.
    """

    return (
        rho * g
        * (L - x)
    )


# ============================================================
# Solve rod problem
# ============================================================

def solve_rod_problem(n_elements):
    """
    Solve the rod-under-self-weight problem using
    a uniform L2 finite element mesh.
    """

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    Coord, Connectivity = generate_uniform_l2_mesh(
        length=L,
        n_elements=n_elements
    )

    NumNodes = Coord.shape[0]


    # --------------------------------------------------------
    # Boundary condition
    #
    # Fixed support:
    # u(0) = 0
    # --------------------------------------------------------

    Constraints = np.array([
        [1, 1, 0.0]
    ])

    NCons = Constraints.shape[0]


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
        EleType,
        load_type,
        NCons,
        n_elements,
        NGPTS,
        NumNodes
    )

    return (
        Coord,
        Connectivity,
        U
    )


# ============================================================
# FE stress at Gauss points
# ============================================================

def calculate_gauss_point_stress(
    Coord,
    Connectivity,
    U
):
    """
    Calculate FE stress at the two Gauss points
    of every L2 element.

    For an L2 element, the strain is constant
    within the element. Therefore both Gauss
    points have the same stress value.
    """

    Nele = Connectivity.shape[0]

    xi_gauss = np.array([
        -1.0 / np.sqrt(3.0),
         1.0 / np.sqrt(3.0)
    ])

    gauss_x = []
    gauss_stress = []


    for ele in range(Nele):

        node1 = Connectivity[ele, 0] - 1
        node2 = Connectivity[ele, 1] - 1


        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        x1 = Coord[node1, 0]
        x2 = Coord[node2, 0]

        Le = x2 - x1


        # ----------------------------------------------------
        # Displacements
        # ----------------------------------------------------

        u1 = U[node1, 0]
        u2 = U[node2, 0]


        # ----------------------------------------------------
        # Element strain
        # ----------------------------------------------------

        strain = (
            u2 - u1
        ) / Le


        # ----------------------------------------------------
        # Element stress
        # ----------------------------------------------------

        stress = (
            E * strain
        )


        # ----------------------------------------------------
        # Physical Gauss-point coordinates
        # ----------------------------------------------------

        for xi in xi_gauss:

            N1 = 0.5 * (
                1.0 - xi
            )

            N2 = 0.5 * (
                1.0 + xi
            )

            x_gp = (
                N1 * x1
                + N2 * x2
            )

            gauss_x.append(
                x_gp
            )

            gauss_stress.append(
                stress
            )


    return (
        np.array(gauss_x),
        np.array(gauss_stress)
    )


# ============================================================
# FE stress at element nodes
# ============================================================

def calculate_element_node_stress(
    Coord,
    Connectivity,
    U
):
    """
    Calculate stress at the nodes of each individual element.

    No averaging is performed.

    An interior global node therefore has two values:
    one from the element on the left and one from
    the element on the right.
    """

    Nele = Connectivity.shape[0]

    element_node_x = []
    element_node_stress = []


    for ele in range(Nele):

        node1 = Connectivity[ele, 0] - 1
        node2 = Connectivity[ele, 1] - 1


        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        x1 = Coord[node1, 0]
        x2 = Coord[node2, 0]

        Le = x2 - x1


        # ----------------------------------------------------
        # Displacements
        # ----------------------------------------------------

        u1 = U[node1, 0]
        u2 = U[node2, 0]


        # ----------------------------------------------------
        # Strain and stress
        # ----------------------------------------------------

        strain = (
            u2 - u1
        ) / Le

        stress = (
            E * strain
        )


        # ----------------------------------------------------
        # Store stress at both local element nodes
        # ----------------------------------------------------

        element_node_x.extend([
            x1,
            x2
        ])

        element_node_stress.extend([
            stress,
            stress
        ])


    return (
        np.array(element_node_x),
        np.array(element_node_stress)
    )


# ============================================================
# Recovered nodal stress
# ============================================================

def recover_nodal_stress(
    Coord,
    Connectivity,
    U
):
    """
    Recover one stress value at each global node.

    Stress contributions from all elements connected
    to a node are averaged.
    """

    NumNodes = Coord.shape[0]
    Nele = Connectivity.shape[0]

    nodal_stress = np.zeros(
        NumNodes
    )

    contribution_count = np.zeros(
        NumNodes,
        dtype=int
    )


    for ele in range(Nele):

        node1 = Connectivity[ele, 0] - 1
        node2 = Connectivity[ele, 1] - 1


        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        x1 = Coord[node1, 0]
        x2 = Coord[node2, 0]

        Le = x2 - x1


        # ----------------------------------------------------
        # Displacements
        # ----------------------------------------------------

        u1 = U[node1, 0]
        u2 = U[node2, 0]


        # ----------------------------------------------------
        # Element stress
        # ----------------------------------------------------

        strain = (
            u2 - u1
        ) / Le

        stress = (
            E * strain
        )


        # ----------------------------------------------------
        # Add element contribution to connected nodes
        # ----------------------------------------------------

        nodal_stress[node1] += stress
        nodal_stress[node2] += stress

        contribution_count[node1] += 1
        contribution_count[node2] += 1


    # --------------------------------------------------------
    # Average stress contributions
    # --------------------------------------------------------

    nodal_stress = (
        nodal_stress
        / contribution_count
    )

    return nodal_stress


# ============================================================
# Convergence study
# ============================================================

def run_convergence_study(
    mesh_sizes=(4, 8, 16, 32)
):
    """
    Perform a uniform mesh-refinement study.

    Computes:
        L2 displacement error
        H1 displacement seminorm error
    """

    h_values = []
    L2_errors = []
    H1_errors = []


    print()
    print("Convergence study")
    print("-----------------")

    print(
        f"{'Nele':>8} "
        f"{'h':>12} "
        f"{'L2 error':>16} "
        f"{'H1 error':>16}"
    )


    for Nele in mesh_sizes:

        # ----------------------------------------------------
        # Solve
        # ----------------------------------------------------

        Coord, Connectivity, U = (
            solve_rod_problem(
                Nele
            )
        )


        # ----------------------------------------------------
        # Calculate errors
        # ----------------------------------------------------

        L2_error, H1_error = Calculate_Error(
            Connectivity,
            Coord,
            EleType,
            3,
            U,
            analytical_displacement,
            analytical_gradient
        )


        # ----------------------------------------------------
        # Mesh size
        # ----------------------------------------------------

        h = (
            L / Nele
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        h_values.append(
            h
        )

        L2_errors.append(
            L2_error
        )

        H1_errors.append(
            H1_error
        )


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"{Nele:8d} "
            f"{h:12.6f} "
            f"{L2_error:16.8e} "
            f"{H1_error:16.8e}"
        )


    return (
        np.array(h_values),
        np.array(L2_errors),
        np.array(H1_errors)
    )


# ============================================================
# Convergence rates
# ============================================================

def print_convergence_rates(
    h_values,
    L2_errors,
    H1_errors
):
    """
    Calculate and print observed convergence rates.
    """

    print()
    print("Observed convergence rates")
    print("--------------------------")

    print(
        f"{'Refinement':>15} "
        f"{'L2 rate':>12} "
        f"{'H1 rate':>12}"
    )


    for i in range(
        1,
        len(h_values)
    ):

        L2_rate = (
            np.log(
                L2_errors[i - 1]
                / L2_errors[i]
            )
            / np.log(
                h_values[i - 1]
                / h_values[i]
            )
        )


        H1_rate = (
            np.log(
                H1_errors[i - 1]
                / H1_errors[i]
            )
            / np.log(
                h_values[i - 1]
                / h_values[i]
            )
        )


        old_n = int(
            round(
                L / h_values[i - 1]
            )
        )

        new_n = int(
            round(
                L / h_values[i]
            )
        )


        print(
            f"{old_n:4d} -> {new_n:4d} "
            f"{L2_rate:12.4f} "
            f"{H1_rate:12.4f}"
        )


# ============================================================
# Print displacement results
# ============================================================

def print_displacement_results(
    Coord,
    U
):
    """
    Print FEM and analytical nodal displacements.
    """

    x_nodes = Coord[:, 0]

    U_exact = analytical_displacement(
        x_nodes
    )


    print()
    print("Rod under self-weight")
    print("---------------------")

    print(
        f"Number of elements: "
        f"{len(x_nodes) - 1}"
    )

    print()


    print(
        f"{'Node':>6} "
        f"{'x':>12} "
        f"{'FEM u':>16} "
        f"{'Exact u':>16} "
        f"{'Error':>16}"
    )


    for node in range(
        len(x_nodes)
    ):

        error = abs(
            U[node, 0]
            - U_exact[node]
        )

        print(
            f"{node + 1:6d} "
            f"{x_nodes[node]:12.6f} "
            f"{U[node, 0]:16.8e} "
            f"{U_exact[node]:16.8e} "
            f"{error:16.8e}"
        )


    # --------------------------------------------------------
    # Tip displacement
    # --------------------------------------------------------

    tip_fem = U[-1, 0]

    tip_exact = analytical_displacement(
        L
    )

    tip_error = abs(
        tip_fem
        - tip_exact
    )


    print()

    print(
        f"FEM tip displacement   = "
        f"{tip_fem:.8e} m"
    )

    print(
        f"Exact tip displacement = "
        f"{tip_exact:.8e} m"
    )

    print(
        f"Absolute tip error     = "
        f"{tip_error:.8e} m"
    )


# ============================================================
# Print stress results
# ============================================================

def print_stress_results(
    Coord,
    gauss_x,
    gauss_stress,
    element_node_x,
    element_node_stress,
    recovered_stress
):
    """
    Print Gauss-point, element-node, and recovered stresses.
    """

    x_nodes = Coord[:, 0]


    # --------------------------------------------------------
    # Gauss-point stress
    # --------------------------------------------------------

    print()
    print("FE stress at Gauss points")
    print("-------------------------")

    print(
        f"{'Point':>8} "
        f"{'x':>12} "
        f"{'FE stress':>16} "
        f"{'Exact stress':>16}"
    )


    for i in range(
        len(gauss_x)
    ):

        sigma_exact = analytical_stress(
            gauss_x[i]
        )

        print(
            f"{i + 1:8d} "
            f"{gauss_x[i]:12.6f} "
            f"{gauss_stress[i]:16.8e} "
            f"{sigma_exact:16.8e}"
        )


    # --------------------------------------------------------
    # Element-node stress
    # --------------------------------------------------------

    print()
    print("FE stress at element nodes")
    print("--------------------------")

    print(
        f"{'Point':>8} "
        f"{'x':>12} "
        f"{'FE stress':>16} "
        f"{'Exact stress':>16}"
    )


    for i in range(
        len(element_node_x)
    ):

        sigma_exact = analytical_stress(
            element_node_x[i]
        )

        print(
            f"{i + 1:8d} "
            f"{element_node_x[i]:12.6f} "
            f"{element_node_stress[i]:16.8e} "
            f"{sigma_exact:16.8e}"
        )


    # --------------------------------------------------------
    # Recovered nodal stress
    # --------------------------------------------------------

    print()
    print("Recovered nodal stress")
    print("----------------------")

    print(
        f"{'Node':>8} "
        f"{'x':>12} "
        f"{'AvgNodes':>16} "
        f"{'Exact stress':>16}"
    )


    for node in range(
        len(x_nodes)
    ):

        sigma_exact = analytical_stress(
            x_nodes[node]
        )

        print(
            f"{node + 1:8d} "
            f"{x_nodes[node]:12.6f} "
            f"{recovered_stress[node]:16.8e} "
            f"{sigma_exact:16.8e}"
        )


# ============================================================
# Plot displacement
# ============================================================

def plot_displacement(
    Coord,
    U
):
    """
    Plot FEM and analytical displacement.
    """

    x_nodes = Coord[:, 0]


    x_plot = np.linspace(
        0.0,
        L,
        300
    )

    u_exact = analytical_displacement(
        x_plot
    )


    plt.figure()

    plt.plot(
        x_plot,
        u_exact,
        label="Analytical"
    )

    plt.plot(
        x_nodes,
        U[:, 0],
        "o--",
        label="FEM (L2)"
    )


    plt.xlabel(
        "Position, x (m)"
    )

    plt.ylabel(
        "Displacement, u (m)"
    )

    plt.title(
        "Rod Under Self-Weight"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "rod_self_weight_displacement.png",
        dpi=300
    )


# ============================================================
# Plot convergence
# ============================================================

def plot_convergence(
    h_values,
    L2_errors,
    H1_errors
):
    """
    Plot error versus mesh size.
    """

    plt.figure()


    plt.loglog(
        h_values,
        L2_errors,
        "o-",
        label="L2 error"
    )

    plt.loglog(
        h_values,
        H1_errors,
        "s-",
        label="H1 seminorm error"
    )


    plt.xlabel(
        "Element size, h"
    )

    plt.ylabel(
        "Error"
    )

    plt.title(
        "Rod Mesh Convergence"
    )

    plt.legend()

    plt.grid(
        True,
        which="both"
    )

    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "rod_self_weight_convergence.png",
        dpi=300
    )


# ============================================================
# Plot stress recovery
# ============================================================

def plot_stress(
    Coord,
    gauss_x,
    gauss_stress,
    element_node_x,
    element_node_stress,
    recovered_stress
):
    """
    Compare:
        FE stress at Gauss points
        FE stress at element nodes
        FE stress after nodal averaging
        Analytical stress
    """

    x_nodes = Coord[:, 0]


    # --------------------------------------------------------
    # Analytical stress
    # --------------------------------------------------------

    x_exact = np.linspace(
        0.0,
        L,
        300
    )

    sigma_exact = analytical_stress(
        x_exact
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure()


    plt.plot(
        gauss_x,
        gauss_stress,
        "^",
        label="FE stress (Gauss Points)"
    )


    plt.plot(
        element_node_x,
        element_node_stress,
        "s-",
        label="FE stress (Element Nodes)"
    )


    plt.plot(
        x_nodes,
        recovered_stress,
        "o-",
        label="FE stress (AvgNodes)"
    )


    plt.plot(
        x_exact,
        sigma_exact,
        "--",
        label="Analytical Stress"
    )


    plt.xlabel(
        "Position, x (m)"
    )

    plt.ylabel(
        "Axial Stress (Pa)"
    )

    plt.title(
        "Stress Recovery for Rod Under Self-Weight"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "rod_self_weight_stress.png",
        dpi=300
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Base problem
    # --------------------------------------------------------

    Nele = 4

    Coord, Connectivity, U = (
        solve_rod_problem(
            Nele
        )
    )


    # --------------------------------------------------------
    # Displacement results
    # --------------------------------------------------------

    print_displacement_results(
        Coord,
        U
    )


    # --------------------------------------------------------
    # Gauss-point stress
    # --------------------------------------------------------

    gauss_x, gauss_stress = (
        calculate_gauss_point_stress(
            Coord,
            Connectivity,
            U
        )
    )


    # --------------------------------------------------------
    # Element-node stress
    # --------------------------------------------------------

    (
        element_node_x,
        element_node_stress
    ) = calculate_element_node_stress(
        Coord,
        Connectivity,
        U
    )


    # --------------------------------------------------------
    # Recovered nodal stress
    # --------------------------------------------------------

    recovered_stress = (
        recover_nodal_stress(
            Coord,
            Connectivity,
            U
        )
    )


    # --------------------------------------------------------
    # Print stress results
    # --------------------------------------------------------

    print_stress_results(
        Coord,
        gauss_x,
        gauss_stress,
        element_node_x,
        element_node_stress,
        recovered_stress
    )


    # --------------------------------------------------------
    # Convergence study
    # --------------------------------------------------------

    h_values, L2_errors, H1_errors = (
        run_convergence_study(
            mesh_sizes=(
                4,
                8,
                16,
                32
            )
        )
    )


    print_convergence_rates(
        h_values,
        L2_errors,
        H1_errors
    )


    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_displacement(
        Coord,
        U
    )


    plot_convergence(
        h_values,
        L2_errors,
        H1_errors
    )


    plot_stress(
        Coord,
        gauss_x,
        gauss_stress,
        element_node_x,
        element_node_stress,
        recovered_stress
    )


    # --------------------------------------------------------
    # Show figures
    # --------------------------------------------------------

    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()