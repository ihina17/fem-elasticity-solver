"""
Bending of a Cantilever Beam

A two-dimensional cantilever beam subjected to gravity is solved
using Q4 and Q8 finite elements.

The study compares:

    1. Q4 elements with one layer through the depth
    2. Q4 elements with four layers through the depth
    3. Q8 elements with one layer through the depth

The FEM tip displacement is compared with the Euler-Bernoulli
beam solution.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.meshes.Q4 import generate_rectangular_q4_mesh
from src.meshes.Q8 import generate_cantilever_q8_mesh
from src.physics_models.elasticity_driver import Driver_LE


# ============================================================
# Output directory
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# Problem parameters
# ============================================================

L = 1.0
H = 0.1
width = 1.0

E = 200e6
nu = 0.0

rho = 1000.0
g = 10.0


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
    "thickness": width,
}


# ============================================================
# Gravity body force
# ============================================================

load_type = {
    "type": "gravity",
    "rho": rho,
    "g": np.array([
        0.0,
        -g,
    ]),
}


# ============================================================
# Analytical beam solution
# ============================================================

def second_moment_of_area():
    """
    Second moment of area of the rectangular cross-section.

    I = b H^3 / 12
    """

    return (
        width
        * H**3
        / 12.0
    )


def distributed_load():
    """
    Equivalent gravity load per unit beam length.

    q = rho g H b
    """

    return (
        rho
        * g
        * H
        * width
    )


def analytical_tip_displacement():
    """
    Euler-Bernoulli tip displacement for a cantilever
    under a uniform distributed load.

    Positive y is upward, so the displacement is negative.
    """

    q = distributed_load()
    I = second_moment_of_area()

    return (
        -q
        * L**4
        / (
            8.0
            * E
            * I
        )
    )


def analytical_deflection(x):
    """
    Euler-Bernoulli deflection along the beam.

    v(x) =
        -q x^2 (6L^2 - 4Lx + x^2)
        --------------------------------
                    24 E I
    """

    q = distributed_load()
    I = second_moment_of_area()

    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        -q
        * x**2
        * (
            6.0 * L**2
            - 4.0 * L * x
            + x**2
        )
        / (
            24.0
            * E
            * I
        )
    )


# ============================================================
# Boundary conditions
# ============================================================

def create_fixed_end_constraints(Coord):
    """
    Fix both displacement components at x = 0.

        u = 0
        v = 0
    """

    left_nodes = np.where(
        np.isclose(
            Coord[:, 0],
            0.0,
        )
    )[0]

    Constraints = []

    for node in left_nodes:

        # x displacement
        Constraints.append([
            node + 1,
            1,
            0.0,
        ])

        # y displacement
        Constraints.append([
            node + 1,
            2,
            0.0,
        ])

    return np.array(
        Constraints,
        dtype=float,
    )


# ============================================================
# Mesh generation
# ============================================================

def generate_beam_mesh(
    element_type,
    nx,
    ny=1,
):
    """
    Generate a Q4 or Q8 cantilever mesh.

    Q4:
        nx elements along the length
        ny elements through the depth

    Q8:
        nx elements along the length
        one element through the depth
    """

    element_type = (
        element_type
        .strip()
        .upper()
    )

    if element_type == "Q4":

        return generate_rectangular_q4_mesh(
            nx=nx,
            ny=ny,
            length=L,
            height=H,
        )

    if element_type == "Q8":

        if ny != 1:

            raise ValueError(
                "The current Q8 mesh generator uses "
                "one element layer through the depth."
            )

        return generate_cantilever_q8_mesh(
            nx=nx,
            length=L,
            height=H,
        )

    raise ValueError(
        "element_type must be 'Q4' or 'Q8'."
    )


# ============================================================
# Solve cantilever problem
# ============================================================

def solve_cantilever(
    element_type,
    nx,
    ny=1,
):
    """
    Solve the cantilever beam problem.
    """

    element_type = (
        element_type
        .strip()
        .upper()
    )

    Coord, Connectivity = generate_beam_mesh(
        element_type,
        nx,
        ny,
    )

    NumNodes = Coord.shape[0]
    Nele = Connectivity.shape[0]

    Constraints = create_fixed_end_constraints(
        Coord
    )

    NCons = Constraints.shape[0]

    if element_type == "Q4":
        NGPTS = 2

    elif element_type == "Q8":
        NGPTS = 3

    else:
        raise ValueError(
            "Unsupported element type."
        )

    U = Driver_LE(
        Connectivity,
        Constraints,
        Coord,
        medium_set,
        dim,
        dofs_per_node,
        element_type,
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
# Tip displacement
# ============================================================

def get_tip_displacement(
    Coord,
    U,
):
    """
    Determine the vertical displacement at x = L.

    If a centerline node exists at the free end, use it.
    Otherwise average the free-edge vertical displacements.
    """

    tip_nodes = np.where(
        np.isclose(
            Coord[:, 0],
            L,
        )
    )[0]

    if len(tip_nodes) == 0:

        raise ValueError(
            "No nodes were found at the free end."
        )

    y_tip = Coord[
        tip_nodes,
        1,
    ]

    distance_from_center = np.abs(
        y_tip
        - H / 2.0
    )

    closest_index = np.argmin(
        distance_from_center
    )

    closest_node = tip_nodes[
        closest_index
    ]

    if np.isclose(
        Coord[
            closest_node,
            1,
        ],
        H / 2.0,
    ):

        return float(
            U[
                closest_node,
                1,
            ]
        )

    return float(
        np.mean(
            U[
                tip_nodes,
                1,
            ]
        )
    )


# ============================================================
# Convergence study
# ============================================================

def run_convergence_study(
    mesh_sizes=(
        2,
        5,
        10,
        20,
        40,
    )
):
    """
    Compare:

        Q4 - one layer
        Q4 - four layers
        Q8 - one layer
    """

    exact_tip = analytical_tip_displacement()

    results = {
        "nx": [],
        "h": [],
        "Q4_1": [],
        "Q4_4": [],
        "Q8_1": [],
    }

    print()
    print("Cantilever beam convergence study")
    print("---------------------------------")
    print()

    print(
        f"{'Nx':>6} "
        f"{'Q4 (1 layer)':>16} "
        f"{'Q4 (4 layers)':>16} "
        f"{'Q8 (1 layer)':>16}"
    )

    for nx in mesh_sizes:

        # ----------------------------------------------------
        # Q4 - one layer
        # ----------------------------------------------------

        Coord, Connectivity, U = solve_cantilever(
            element_type="Q4",
            nx=nx,
            ny=1,
        )

        tip_q4_1 = get_tip_displacement(
            Coord,
            U,
        )

        # ----------------------------------------------------
        # Q4 - four layers
        # ----------------------------------------------------

        Coord, Connectivity, U = solve_cantilever(
            element_type="Q4",
            nx=nx,
            ny=4,
        )

        tip_q4_4 = get_tip_displacement(
            Coord,
            U,
        )

        # ----------------------------------------------------
        # Q8 - one layer
        # ----------------------------------------------------

        Coord, Connectivity, U = solve_cantilever(
            element_type="Q8",
            nx=nx,
            ny=1,
        )

        tip_q8_1 = get_tip_displacement(
            Coord,
            U,
        )

        # ----------------------------------------------------
        # Normalize using Euler-Bernoulli result
        # ----------------------------------------------------

        normalized_q4_1 = abs(
            tip_q4_1
            / exact_tip
        )

        normalized_q4_4 = abs(
            tip_q4_4
            / exact_tip
        )

        normalized_q8_1 = abs(
            tip_q8_1
            / exact_tip
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results["nx"].append(
            nx
        )

        results["h"].append(
            L / nx
        )

        results["Q4_1"].append(
            normalized_q4_1
        )

        results["Q4_4"].append(
            normalized_q4_4
        )

        results["Q8_1"].append(
            normalized_q8_1
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"{nx:6d} "
            f"{normalized_q4_1:16.6f} "
            f"{normalized_q4_4:16.6f} "
            f"{normalized_q8_1:16.6f}"
        )

    for key in results:

        results[key] = np.array(
            results[key]
        )

    return results


# ============================================================
# Problem information
# ============================================================

def print_problem_information():
    """
    Print analytical beam quantities.
    """

    I = second_moment_of_area()
    q = distributed_load()
    exact_tip = analytical_tip_displacement()

    print()
    print("Cantilever beam under gravity")
    print("-----------------------------")

    print(
        f"Length                  = "
        f"{L:.6f} m"
    )

    print(
        f"Height                  = "
        f"{H:.6f} m"
    )

    print(
        f"Width                   = "
        f"{width:.6f} m"
    )

    print(
        f"Young's modulus         = "
        f"{E:.6e} Pa"
    )

    print(
        f"Poisson's ratio         = "
        f"{nu:.6f}"
    )

    print(
        f"Density                 = "
        f"{rho:.6f} kg/m^3"
    )

    print(
        f"Gravity                 = "
        f"{g:.6f} m/s^2"
    )

    print(
        f"Distributed beam load   = "
        f"{q:.6f} N/m"
    )

    print(
        f"Second moment of area   = "
        f"{I:.8e} m^4"
    )

    print(
        f"Analytical tip displacement = "
        f"{exact_tip:.8e} m"
    )


# ============================================================
# Single solution information
# ============================================================

def print_single_solution(
    element_type,
    nx,
    ny,
    Coord,
    Connectivity,
    U,
):
    """
    Print results for one FEM mesh.
    """

    tip_fem = get_tip_displacement(
        Coord,
        U,
    )

    tip_exact = analytical_tip_displacement()

    normalized = abs(
        tip_fem
        / tip_exact
    )

    error_percent = (
        abs(
            tip_fem
            - tip_exact
        )
        / abs(
            tip_exact
        )
        * 100.0
    )

    print()
    print(
        f"{element_type} cantilever solution"
    )

    print(
        "----------------------------"
    )

    print(
        f"Elements along length  = "
        f"{nx}"
    )

    print(
        f"Layers through depth   = "
        f"{ny}"
    )

    print(
        f"Number of nodes        = "
        f"{Coord.shape[0]}"
    )

    print(
        f"Number of elements     = "
        f"{Connectivity.shape[0]}"
    )

    print(
        f"FEM tip displacement   = "
        f"{tip_fem:.8e} m"
    )

    print(
        f"Exact tip displacement = "
        f"{tip_exact:.8e} m"
    )

    print(
        f"Normalized displacement = "
        f"{normalized:.6f}"
    )

    print(
        f"Relative error          = "
        f"{error_percent:.4f} %"
    )


# ============================================================
# Convergence plot
# ============================================================

def plot_convergence(
    results,
):
    """
    Plot normalized tip displacement.
    """

    nx = results["nx"]

    plt.figure()

    plt.plot(
        nx,
        results["Q4_1"],
        "o-",
        label="Q4 - 1 layer",
    )

    plt.plot(
        nx,
        results["Q4_4"],
        "s-",
        label="Q4 - 4 layers",
    )

    plt.plot(
        nx,
        results["Q8_1"],
        "^-",
        label="Q8 - 1 layer",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Euler-Bernoulli",
    )

    plt.xlabel(
        "Number of elements along beam, Nx"
    )

    plt.ylabel(
        "|v_FEM / v_EB|"
    )

    plt.title(
        "Cantilever Tip-Displacement Convergence"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cantilever_convergence.png",
        dpi=300,
    )


# ============================================================
# Error plot
# ============================================================

def plot_error(
    results,
):
    """
    Plot difference from the Euler-Bernoulli tip displacement.
    """

    h = results["h"]

    error_q4_1 = np.abs(
        results["Q4_1"]
        - 1.0
    )

    error_q4_4 = np.abs(
        results["Q4_4"]
        - 1.0
    )

    error_q8_1 = np.abs(
        results["Q8_1"]
        - 1.0
    )

    plt.figure()

    plt.loglog(
        h,
        error_q4_1,
        "o-",
        label="Q4 - 1 layer",
    )

    plt.loglog(
        h,
        error_q4_4,
        "s-",
        label="Q4 - 4 layers",
    )

    plt.loglog(
        h,
        error_q8_1,
        "^-",
        label="Q8 - 1 layer",
    )

    plt.xlabel(
        "Element size, h = L / Nx"
    )

    plt.ylabel(
        "Difference from Euler-Bernoulli tip displacement"
    )

    plt.title(
        "Cantilever Mesh Refinement"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cantilever_error.png",
        dpi=300,
    )


# ============================================================
# Element plotting order
# ============================================================

def element_plot_order(
    element_type,
):
    """
    Local node order used to draw element boundaries.
    """

    element_type = (
        element_type
        .strip()
        .upper()
    )

    if element_type == "Q4":

        return [
            0,
            1,
            2,
            3,
            0,
        ]

    if element_type == "Q8":

        return [
            0,
            4,
            1,
            5,
            2,
            6,
            3,
            7,
            0,
        ]

    raise ValueError(
        "Unsupported element type."
    )


# ============================================================
# Mesh plot
# ============================================================

def plot_mesh(
    Coord,
    Connectivity,
    element_type,
    filename,
):
    """
    Plot the undeformed finite element mesh.
    """

    order = element_plot_order(
        element_type
    )

    plt.figure()

    for element in Connectivity:

        ids = (
            element
            - 1
        )

        x = Coord[
            ids[order],
            0,
        ]

        y = Coord[
            ids[order],
            1,
        ]

        plt.plot(
            x,
            y,
        )

    plt.xlabel(
        "x (m)"
    )

    plt.ylabel(
        "y (m)"
    )

    plt.title(
        f"{element_type} Cantilever Mesh"
    )

    plt.axis(
        "equal"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / filename,
        dpi=300,
    )


# ============================================================
# Deformed mesh
# ============================================================

def plot_deformed_mesh(
    Coord,
    Connectivity,
    U,
    element_type,
    filename,
    scale=10.0,
):
    """
    Plot undeformed and scaled deformed finite element meshes.
    """

    order = element_plot_order(
        element_type
    )

    deformed_coord = (
        Coord
        + scale * U
    )

    plt.figure()

    for element in Connectivity:

        ids = (
            element
            - 1
        )

        # ----------------------------------------------------
        # Undeformed mesh
        # ----------------------------------------------------

        x_original = Coord[
            ids[order],
            0,
        ]

        y_original = Coord[
            ids[order],
            1,
        ]

        plt.plot(
            x_original,
            y_original,
            linestyle="--",
        )

        # ----------------------------------------------------
        # Deformed mesh
        # ----------------------------------------------------

        x_deformed = deformed_coord[
            ids[order],
            0,
        ]

        y_deformed = deformed_coord[
            ids[order],
            1,
        ]

        plt.plot(
            x_deformed,
            y_deformed,
        )

    plt.xlabel(
        "x (m)"
    )

    plt.ylabel(
        "y (m)"
    )

    plt.title(
        f"{element_type} Cantilever Beam "
        f"(deformation scale = {scale:g})"
    )

    plt.axis(
        "equal"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / filename,
        dpi=300,
    )


# ============================================================
# Q8 centerline deflection
# ============================================================

def plot_q8_deflection(
    Coord,
    U,
):
    """
    Compare Q8 centerline displacement with the
    Euler-Bernoulli beam solution.
    """

    center_nodes = np.where(
        np.isclose(
            Coord[:, 1],
            H / 2.0,
        )
    )[0]

    x_fem = Coord[
        center_nodes,
        0,
    ]

    v_fem = U[
        center_nodes,
        1,
    ]

    # Sort nodes from left to right
    order = np.argsort(
        x_fem
    )

    x_fem = x_fem[
        order
    ]

    v_fem = v_fem[
        order
    ]

    x_exact = np.linspace(
        0.0,
        L,
        300,
    )

    v_exact = analytical_deflection(
        x_exact
    )

    plt.figure()

    plt.plot(
        x_exact,
        v_exact,
        label="Euler-Bernoulli",
    )

    plt.plot(
        x_fem,
        v_fem,
        "o--",
        label="Q8 FEM",
    )

    plt.xlabel(
        "Position, x (m)"
    )

    plt.ylabel(
        "Vertical displacement, v (m)"
    )

    plt.title(
        "Cantilever Beam Deflection"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cantilever_q8_deflection.png",
        dpi=300,
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Problem information
    # --------------------------------------------------------

    print_problem_information()


    # ========================================================
    # Representative Q8 solution
    # ========================================================

    nx_q8 = 5

    (
        Coord_q8,
        Connectivity_q8,
        U_q8,
    ) = solve_cantilever(
        element_type="Q8",
        nx=nx_q8,
        ny=1,
    )

    print_single_solution(
        element_type="Q8",
        nx=nx_q8,
        ny=1,
        Coord=Coord_q8,
        Connectivity=Connectivity_q8,
        U=U_q8,
    )


    # ========================================================
    # Representative Q4 solution
    # ========================================================

    nx_q4 = 20
    ny_q4 = 4

    (
        Coord_q4,
        Connectivity_q4,
        U_q4,
    ) = solve_cantilever(
        element_type="Q4",
        nx=nx_q4,
        ny=ny_q4,
    )

    print_single_solution(
        element_type="Q4",
        nx=nx_q4,
        ny=ny_q4,
        Coord=Coord_q4,
        Connectivity=Connectivity_q4,
        U=U_q4,
    )


    # ========================================================
    # Convergence study
    # ========================================================

    results = run_convergence_study(
        mesh_sizes=(
            2,
            5,
            10,
            20,
            40,
        )
    )


    # ========================================================
    # Meshes used for visual element comparison
    # ========================================================

    Coord_q4_mesh, Connectivity_q4_mesh = (
        generate_beam_mesh(
            element_type="Q4",
            nx=5,
            ny=1,
        )
    )

    Coord_q8_mesh, Connectivity_q8_mesh = (
        generate_beam_mesh(
            element_type="Q8",
            nx=5,
            ny=1,
        )
    )


    # ========================================================
    # Plots
    # ========================================================

    plot_convergence(
        results
    )

    plot_error(
        results
    )

    plot_q8_deflection(
        Coord_q8,
        U_q8,
    )


    # --------------------------------------------------------
    # Mesh comparison
    # --------------------------------------------------------

    plot_mesh(
        Coord_q4_mesh,
        Connectivity_q4_mesh,
        element_type="Q4",
        filename="cantilever_q4_mesh.png",
    )

    plot_mesh(
        Coord_q8_mesh,
        Connectivity_q8_mesh,
        element_type="Q8",
        filename="cantilever_q8_mesh.png",
    )


    # --------------------------------------------------------
    # Deformed meshes
    # --------------------------------------------------------

    plot_deformed_mesh(
        Coord_q8,
        Connectivity_q8,
        U_q8,
        element_type="Q8",
        filename="cantilever_q8_deformed.png",
        scale=10.0,
    )

    plot_deformed_mesh(
        Coord_q4,
        Connectivity_q4,
        U_q4,
        element_type="Q4",
        filename="cantilever_q4_deformed.png",
        scale=10.0,
    )


    # ========================================================
    # Show figures
    # ========================================================

    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()