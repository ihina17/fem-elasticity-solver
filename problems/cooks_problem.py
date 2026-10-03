r"""
Cook's Membrane Problem

A tapered two-dimensional cantilever is subjected to a
uniform upward shear traction on the right boundary.

Geometry:

    upper-left  (0, 44) -------- (48, 60)  A
                         \         |
                          \        |
                           \       |
                            \      |
    lower-left  (0, 0) ----- (48, 44)

Boundary conditions:

    ux = 0
    uy = 0

on the left boundary.

The total upward shear force on the right boundary is

    V = 100.

The problem is modeled using plane-strain linear elasticity.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.meshes.cooks_Q4 import (
    generate_cooks_q4_mesh,
)

from src.physics_models.boundary_loads import (
    create_constant_edge_traction_loads,
)

from src.physics_models.elasticity_driver import (
    Driver_LE,
)

from src.fem.shape_functions import (
    ShapeFunctions,
)

from src.physics_models.elasticity_kernel import (
    Get_ConstitutiveMatrix,
)


# ============================================================
# Output directory
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# Problem parameters
# ============================================================

E = 100.0
nu = 0.4999

thickness = 1.0

V = 100.0


# ============================================================
# Geometry
# ============================================================

X_LEFT = 0.0
X_RIGHT = 48.0

Y_LEFT_BOTTOM = 0.0
Y_LEFT_TOP = 44.0

Y_RIGHT_BOTTOM = 44.0
Y_RIGHT_TOP = 60.0


# ============================================================
# FEM settings
# ============================================================

dim = 2
dofs_per_node = 2


# ============================================================
# No body force
# ============================================================

load_type = {
    "type": "none",
}


# ============================================================
# Material
# ============================================================

def create_material(
    poisson_ratio=nu,
):
    """
    Create the plane-strain material definition.
    """

    return {
        "type": "elastic_moduli",
        "E": E,
        "nu": poisson_ratio,
        "analysis_type": "plane_strain",
        "thickness": thickness,
    }


# ============================================================
# Fixed left boundary
# ============================================================

def create_cooks_constraints(
    Coord,
):
    """
    Fix both displacement components on x = 0.

        ux = 0
        uy = 0
    """

    left_nodes = np.where(
        np.isclose(
            Coord[:, 0],
            X_LEFT,
        )
    )[0]

    Constraints = []

    for node in left_nodes:

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

    return np.array(
        Constraints,
        dtype=float,
    )


# ============================================================
# Right-edge nodes
# ============================================================

def get_right_edge_nodes(
    Coord,
):
    """
    Return 1-based node numbers on the loaded right edge.

    Nodes are ordered from bottom to top.
    """

    nodes = np.where(
        np.isclose(
            Coord[:, 0],
            X_RIGHT,
        )
    )[0]

    if len(nodes) < 2:

        raise ValueError(
            "At least two nodes are required on the right edge."
        )

    order = np.argsort(
        Coord[
            nodes,
            1,
        ]
    )

    nodes = nodes[
        order
    ]

    return (
        nodes
        + 1
    )


# ============================================================
# Right-edge traction
# ============================================================

def create_cooks_edge_loads(
    Coord,
):
    """
    Create the consistent nodal forces produced by the
    distributed right-edge shear load.

    The specified V = 100 is treated as the total resultant
    vertical force.

    Therefore,

        traction_y = V / (right-edge length * thickness).
    """

    edge_nodes = get_right_edge_nodes(
        Coord
    )

    # --------------------------------------------------------
    # Right-edge length
    # --------------------------------------------------------

    bottom_node = (
        edge_nodes[0]
        - 1
    )

    top_node = (
        edge_nodes[-1]
        - 1
    )

    edge_length = np.linalg.norm(
        Coord[
            top_node,
            :
        ]
        - Coord[
            bottom_node,
            :
        ]
    )

    if edge_length <= 0.0:

        raise ValueError(
            "Right-edge length must be positive."
        )

    # --------------------------------------------------------
    # Uniform traction producing total resultant V
    # --------------------------------------------------------

    traction_y = (
        V
        / (
            edge_length
            * thickness
        )
    )

    traction = np.array([
        0.0,
        traction_y,
    ])

    # --------------------------------------------------------
    # Equivalent consistent nodal loads
    # --------------------------------------------------------

    NodalLoads = (
        create_constant_edge_traction_loads(
            Coord=Coord,
            edge_nodes=edge_nodes,
            traction=traction,
            thickness=thickness,
        )
    )

    return NodalLoads


# ============================================================
# Total applied force
# ============================================================

def total_applied_force(
    NodalLoads,
):
    """
    Return the sum of externally applied nodal forces.
    """

    total_fx = np.sum(
        NodalLoads[
            NodalLoads[:, 1] == 1,
            2,
        ]
    )

    total_fy = np.sum(
        NodalLoads[
            NodalLoads[:, 1] == 2,
            2,
        ]
    )

    return np.array([
        total_fx,
        total_fy,
    ])


# ============================================================
# Solve Cook's problem
# ============================================================

def solve_cooks_problem(
    nx,
    ny=None,
    poisson_ratio=nu,
):
    """
    Solve Cook's membrane using Q4 finite elements.

    Parameters
    ----------
    nx : int
        Number of elements in the horizontal direction.

    ny : int or None
        Number of elements through the vertical direction.
        If None, ny = nx.

    poisson_ratio : float
        Poisson's ratio.
    """

    if ny is None:

        ny = nx

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    Coord, Connectivity = (
        generate_cooks_q4_mesh(
            nx=nx,
            ny=ny,
        )
    )

    NumNodes = Coord.shape[0]

    Nele = Connectivity.shape[0]

    # --------------------------------------------------------
    # Essential boundary conditions
    # --------------------------------------------------------

    Constraints = (
        create_cooks_constraints(
            Coord
        )
    )

    NCons = Constraints.shape[0]

    # --------------------------------------------------------
    # Natural boundary load
    # --------------------------------------------------------

    NodalLoads = (
        create_cooks_edge_loads(
            Coord
        )
    )

    # --------------------------------------------------------
    # Material
    # --------------------------------------------------------

    medium_set = create_material(
        poisson_ratio
    )

    # --------------------------------------------------------
    # Q4 Gaussian quadrature
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
        NodalLoads=NodalLoads,
    )

    return (
        Coord,
        Connectivity,
        U,
        NodalLoads,
    )


# ============================================================
# Tip A displacement
# ============================================================

def get_tip_A_displacement(
    Coord,
    U,
):
    """
    Return displacement at point

        A = (48, 60).
    """

    candidates = np.where(
        np.logical_and(
            np.isclose(
                Coord[:, 0],
                X_RIGHT,
            ),
            np.isclose(
                Coord[:, 1],
                Y_RIGHT_TOP,
            ),
        )
    )[0]

    if len(candidates) != 1:

        raise ValueError(
            "Could not uniquely identify point A."
        )

    node = candidates[0]

    ux = float(
        U[
            node,
            0,
        ]
    )

    uy = float(
        U[
            node,
            1,
        ]
    )

    return (
        ux,
        uy,
    )


# ============================================================
# Mesh plot
# ============================================================

def plot_mesh(
    Coord,
    Connectivity,
):
    """
    Plot the Cook membrane mesh.
    """

    order = [
        0,
        1,
        2,
        3,
        0,
    ]

    plt.figure()

    for element in Connectivity:

        ids = (
            element
            - 1
        )

        plt.plot(
            Coord[
                ids[order],
                0,
            ],
            Coord[
                ids[order],
                1,
            ],
        )

    plt.xlabel(
        "x"
    )

    plt.ylabel(
        "y"
    )

    plt.title(
        "Cook's Membrane Q4 Mesh"
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
        / "cooks_mesh.png",
        dpi=300,
    )


# ============================================================
# Deformed mesh
# ============================================================

def plot_deformed_mesh(
    Coord,
    Connectivity,
    U,
    scale=1.0,
):
    """
    Plot original and scaled deformed Cook membrane.
    """

    order = [
        0,
        1,
        2,
        3,
        0,
    ]

    deformed = (
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
        # Original mesh
        # ----------------------------------------------------

        plt.plot(
            Coord[
                ids[order],
                0,
            ],
            Coord[
                ids[order],
                1,
            ],
            linestyle="--",
        )

        # ----------------------------------------------------
        # Deformed mesh
        # ----------------------------------------------------

        plt.plot(
            deformed[
                ids[order],
                0,
            ],
            deformed[
                ids[order],
                1,
            ],
        )

    plt.xlabel(
        "x"
    )

    plt.ylabel(
        "y"
    )

    plt.title(
        f"Cook's Membrane Deformation "
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
        / "cooks_deformed.png",
        dpi=300,
    )


# ============================================================
# Mesh convergence study
# ============================================================

def run_mesh_convergence(
    mesh_sizes=(
        2,
        4,
        8,
        16,
        32,
    ),
    poisson_ratio=nu,
):
    """
    Compute the Cook tip displacement on a hierarchy
    of square structured meshes.
    """

    results = {
        "n": [],
        "uy_A": [],
    }

    print()
    print("Cook's membrane mesh-convergence study")
    print("--------------------------------------")
    print()

    print(
        f"{'Mesh':>10} "
        f"{'uy(A)':>18}"
    )

    for n in mesh_sizes:

        (
            Coord,
            Connectivity,
            U,
            NodalLoads,
        ) = solve_cooks_problem(
            nx=n,
            ny=n,
            poisson_ratio=poisson_ratio,
        )

        ux_A, uy_A = (
            get_tip_A_displacement(
                Coord,
                U,
            )
        )

        results["n"].append(
            n
        )

        results["uy_A"].append(
            uy_A
        )

        print(
            f"{n:4d} x {n:<4d} "
            f"{uy_A:18.8e}"
        )

    for key in results:

        results[key] = np.array(
            results[key]
        )

    return results


# ============================================================
# Poisson-ratio study
# ============================================================

def run_poisson_study(
    nx=32,
    ny=32,
    poisson_values=(
        0.0,
        0.1,
        0.25,
        0.4,
        0.45,
        0.49,
        0.4999,
    ),
):
    """
    Compute the Cook tip displacement for several
    Poisson-ratio values on one fixed fine mesh.
    """

    results = {
        "nu": [],
        "uy_A": [],
    }

    print()
    print("Cook's membrane Poisson-ratio study")
    print("-----------------------------------")
    print()

    print(
        f"{'nu':>12} "
        f"{'uy(A)':>18}"
    )

    for poisson_ratio in poisson_values:

        (
            Coord,
            Connectivity,
            U,
            NodalLoads,
        ) = solve_cooks_problem(
            nx=nx,
            ny=ny,
            poisson_ratio=poisson_ratio,
        )

        ux_A, uy_A = (
            get_tip_A_displacement(
                Coord,
                U,
            )
        )

        results["nu"].append(
            poisson_ratio
        )

        results["uy_A"].append(
            uy_A
        )

        print(
            f"{poisson_ratio:12.4f} "
            f"{uy_A:18.8e}"
        )

    for key in results:

        results[key] = np.array(
            results[key]
        )

    return results


# ============================================================
# Mesh-convergence plot
# ============================================================

def plot_mesh_convergence(
    results,
):
    """
    Plot tip displacement versus mesh resolution.
    """

    plt.figure()

    plt.plot(
        results["n"],
        results["uy_A"],
        "o-",
    )

    plt.xlabel(
        "Number of elements per direction"
    )

    plt.ylabel(
        "Vertical tip displacement, uy(A)"
    )

    plt.title(
        "Cook's Membrane Mesh Convergence"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cooks_convergence.png",
        dpi=300,
    )


# ============================================================
# Poisson-ratio plot
# ============================================================

def plot_poisson_study(
    results,
):
    """
    Plot tip displacement versus Poisson's ratio.
    """

    plt.figure()

    plt.plot(
        results["nu"],
        results["uy_A"],
        "o-",
    )

    plt.xlabel(
        "Poisson's ratio"
    )

    plt.ylabel(
        "Vertical tip displacement, uy(A)"
    )

    plt.title(
        "Cook's Membrane: Effect of Poisson's Ratio"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cooks_poisson_study.png",
        dpi=300,
    )


# ============================================================
# Stress at one point inside a Q4 element
# ============================================================

def q4_stress_at_point(
    element_coord,
    element_displacement,
    xi,
    eta,
    poisson_ratio,
):
    """
    Compute strain and stress at a specified Q4 natural
    coordinate (xi, eta).

    Returns
    -------
    strain : ndarray
        [epsilon_xx, epsilon_yy, gamma_xy]

    stress : ndarray
        [sigma_xx, sigma_yy, tau_xy]

    sigma_zz : float
        Out-of-plane normal stress for plane strain.
    """

    N, DN = ShapeFunctions(
        "Q4",
        np.array([
            xi,
            eta,
        ]),
    )

    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    J = (
        DN.T
        @ element_coord
    )

    detJ = np.linalg.det(
        J
    )

    if detJ <= 0.0:

        raise ValueError(
            "Element Jacobian must be positive."
        )

    # --------------------------------------------------------
    # Shape-function derivatives in physical coordinates
    # --------------------------------------------------------

    dNdx = (
        DN
        @ np.linalg.inv(J).T
    )

    # --------------------------------------------------------
    # Strain-displacement matrix
    # --------------------------------------------------------

    B = np.zeros(
        (
            3,
            8,
        )
    )

    for a in range(4):

        B[
            0,
            2 * a,
        ] = dNdx[
            a,
            0,
        ]

        B[
            1,
            2 * a + 1,
        ] = dNdx[
            a,
            1,
        ]

        B[
            2,
            2 * a,
        ] = dNdx[
            a,
            1,
        ]

        B[
            2,
            2 * a + 1,
        ] = dNdx[
            a,
            0,
        ]

    # --------------------------------------------------------
    # Element displacement vector
    #
    # [ux1, uy1, ux2, uy2, ...]
    # --------------------------------------------------------

    Ue = element_displacement.reshape(
        -1
    )

    # --------------------------------------------------------
    # Strain
    # --------------------------------------------------------

    strain = (
        B
        @ Ue
    )

    # --------------------------------------------------------
    # Plane-strain constitutive matrix
    # --------------------------------------------------------

    material = create_material(
        poisson_ratio
    )

    C = Get_ConstitutiveMatrix(
        material
    )

    # --------------------------------------------------------
    # In-plane stress
    # --------------------------------------------------------

    stress = (
        C
        @ strain
    )

    # --------------------------------------------------------
    # Out-of-plane stress for plane strain
    #
    # epsilon_zz = 0
    #
    # sigma_zz = lambda * (epsilon_xx + epsilon_yy)
    # --------------------------------------------------------

    lame_lambda = (
        E
        * poisson_ratio
        / (
            (1.0 + poisson_ratio)
            * (
                1.0
                - 2.0 * poisson_ratio
            )
        )
    )

    sigma_zz = (
        lame_lambda
        * (
            strain[0]
            + strain[1]
        )
    )

    return (
        strain,
        stress,
        sigma_zz,
    )


# ============================================================
# Nodal stress trace
# ============================================================

def compute_nodal_stress_trace(
    Coord,
    Connectivity,
    U,
    poisson_ratio,
):
    """
    Compute an averaged nodal value of

        tr(T) = sigma_xx + sigma_yy + sigma_zz

    for plane strain.

    Element stresses are evaluated at the local element nodes
    and averaged over all elements sharing each global node.
    """

    NumNodes = Coord.shape[0]

    trace_sum = np.zeros(
        NumNodes,
        dtype=float,
    )

    contribution_count = np.zeros(
        NumNodes,
        dtype=int,
    )

    # --------------------------------------------------------
    # Q4 natural coordinates corresponding to local nodes
    #
    # 4 -------- 3
    # |          |
    # |          |
    # 1 -------- 2
    # --------------------------------------------------------

    local_coordinates = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0],
    ])

    # --------------------------------------------------------
    # Loop over elements
    # --------------------------------------------------------

    for element in Connectivity:

        ids = (
            element
            - 1
        )

        element_coord = Coord[
            ids,
            :
        ]

        element_displacement = U[
            ids,
            :
        ]

        # ----------------------------------------------------
        # Evaluate stress at each local node
        # ----------------------------------------------------

        for local_node in range(4):

            xi = local_coordinates[
                local_node,
                0,
            ]

            eta = local_coordinates[
                local_node,
                1,
            ]

            (
                strain,
                stress,
                sigma_zz,
            ) = q4_stress_at_point(
                element_coord,
                element_displacement,
                xi,
                eta,
                poisson_ratio,
            )

            sigma_xx = stress[0]

            sigma_yy = stress[1]

            stress_trace = (
                sigma_xx
                + sigma_yy
                + sigma_zz
            )

            global_node = ids[
                local_node
            ]

            trace_sum[
                global_node
            ] += stress_trace

            contribution_count[
                global_node
            ] += 1

    # --------------------------------------------------------
    # Average contributions
    # --------------------------------------------------------

    if np.any(
        contribution_count == 0
    ):

        raise ValueError(
            "A mesh node has no stress contributions."
        )

    nodal_trace = (
        trace_sum
        / contribution_count
    )

    return nodal_trace


# ============================================================
# Stress trace along BC
# ============================================================

def get_stress_trace_along_BC(
    Coord,
    Connectivity,
    U,
    poisson_ratio,
):
    """
    Extract the averaged nodal stress trace along

        BC : x = 24.
    """

    nodal_trace = (
        compute_nodal_stress_trace(
            Coord,
            Connectivity,
            U,
            poisson_ratio,
        )
    )

    # --------------------------------------------------------
    # Nodes on x = 24
    # --------------------------------------------------------

    bc_nodes = np.where(
        np.isclose(
            Coord[:, 0],
            24.0,
        )
    )[0]

    if len(bc_nodes) == 0:

        raise ValueError(
            "No mesh nodes were found on line BC."
        )

    # --------------------------------------------------------
    # Sort from B to C
    # --------------------------------------------------------

    order = np.argsort(
        Coord[
            bc_nodes,
            1,
        ]
    )

    bc_nodes = bc_nodes[
        order
    ]

    y = Coord[
        bc_nodes,
        1,
    ]

    trace = nodal_trace[
        bc_nodes
    ]

    return (
        y,
        trace,
    )


# ============================================================
# Plot stress trace along BC
# ============================================================

def plot_stress_trace_BC(
    Coord,
    Connectivity,
    U,
    poisson_ratio,
):
    """
    Plot averaged nodal tr(T) along line BC.
    """

    y, trace = (
        get_stress_trace_along_BC(
            Coord,
            Connectivity,
            U,
            poisson_ratio,
        )
    )

    plt.figure()

    plt.plot(
        y,
        trace,
        "o-",
    )

    plt.xlabel(
        "y coordinate along BC"
    )

    plt.ylabel(
        "tr(T)"
    )

    plt.title(
        "Cook's Membrane: Stress Trace Along BC"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cooks_stress_trace_BC.png",
        dpi=300,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Representative solution
    # ========================================================

    nx = 8
    ny = 8

    (
        Coord,
        Connectivity,
        U,
        NodalLoads,
    ) = solve_cooks_problem(
        nx=nx,
        ny=ny,
        poisson_ratio=nu,
    )

    # --------------------------------------------------------
    # Tip displacement
    # --------------------------------------------------------

    ux_A, uy_A = (
        get_tip_A_displacement(
            Coord,
            U,
        )
    )

    # --------------------------------------------------------
    # Load check
    # --------------------------------------------------------

    applied_force = (
        total_applied_force(
            NodalLoads
        )
    )

    # --------------------------------------------------------
    # Print representative solution
    # --------------------------------------------------------

    print()
    print("Cook's membrane problem")
    print("-----------------------")

    print(
        f"Young's modulus       = "
        f"{E:.6f}"
    )

    print(
        f"Poisson's ratio       = "
        f"{nu:.6f}"
    )

    print(
        f"Analysis type         = "
        f"plane strain"
    )

    print(
        f"Mesh                  = "
        f"{nx} x {ny}"
    )

    print(
        f"Number of nodes       = "
        f"{Coord.shape[0]}"
    )

    print(
        f"Number of elements    = "
        f"{Connectivity.shape[0]}"
    )

    print()
    print("Applied load check")

    print(
        f"Total Fx              = "
        f"{applied_force[0]:.8e}"
    )

    print(
        f"Total Fy              = "
        f"{applied_force[1]:.8e}"
    )

    print()
    print("Tip A displacement")

    print(
        f"ux(A)                 = "
        f"{ux_A:.8e}"
    )

    print(
        f"uy(A)                 = "
        f"{uy_A:.8e}"
    )

    # ========================================================
    # Mesh-convergence study
    # ========================================================

    mesh_results = run_mesh_convergence(
        mesh_sizes=(
            2,
            4,
            8,
            16,
            32,
        ),
        poisson_ratio=nu,
    )

    # ========================================================
    # Poisson-ratio study
    # ========================================================

    poisson_results = run_poisson_study(
        nx=32,
        ny=32,
        poisson_values=(
            0.0,
            0.1,
            0.25,
            0.4,
            0.45,
            0.49,
            0.4999,
        ),
    )

    # ========================================================
    # Stress trace along BC using finest mesh
    # ========================================================

    (
        Coord_fine,
        Connectivity_fine,
        U_fine,
        NodalLoads_fine,
    ) = solve_cooks_problem(
        nx=32,
        ny=32,
        poisson_ratio=nu,
    )

    # ========================================================
    # Figures
    # ========================================================

    plot_mesh(
        Coord,
        Connectivity,
    )

    plot_deformed_mesh(
        Coord,
        Connectivity,
        U,
        scale=0.1,
    )

    plot_mesh_convergence(
        mesh_results
    )

    plot_poisson_study(
        poisson_results
    )

    plot_stress_trace_BC(
        Coord_fine,
        Connectivity_fine,
        U_fine,
        poisson_ratio=nu,
    )

    # ========================================================
    # Display figures
    # ========================================================

    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()