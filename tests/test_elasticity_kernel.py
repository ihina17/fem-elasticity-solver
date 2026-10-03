import numpy as np

from src.fem.gauss_quadrature import GaussPoints
from src.physics_models.elasticity_kernel import (
    CalculateLocalMatrices,
    Assemble,
)


# ============================================================
# Test local matrices for a 1D L2 rod element
# ============================================================

def test_l2_rod_local_matrices():

    # --------------------------------------------------------
    # Single L2 element: x = 0 to x = 1
    # --------------------------------------------------------

    EleNodes = np.array([1, 2])

    xCap = np.array([
        [0.0],
        [1.0]
    ])

    dim = 1
    dofs_per_node = 1
    EleType = "L2"

    # --------------------------------------------------------
    # Material properties
    # --------------------------------------------------------

    E = 250e6
    A = 0.5

    medium_set = {
        "type": "elastic_moduli",
        "E": E,
        "nu": 0.3,
        "A": A
    }

    # --------------------------------------------------------
    # Body force due to gravity
    # --------------------------------------------------------

    rho = 1500.0
    g = 10.0

    load_type = {
        "type": "gravity",
        "rho": rho,
        "g": g
    }

    # --------------------------------------------------------
    # Gauss quadrature
    # --------------------------------------------------------

    r, w = GaussPoints(
        dim,
        EleType,
        2
    )

    # --------------------------------------------------------
    # FEM local matrices
    # --------------------------------------------------------

    KLocal, RLocal = CalculateLocalMatrices(
        medium_set,
        dofs_per_node,
        EleNodes,
        EleType,
        load_type,
        r,
        w,
        xCap
    )

    # --------------------------------------------------------
    # Analytical local stiffness matrix
    # --------------------------------------------------------

    Le = 1.0

    K_exact = (
        E * A / Le
        * np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])
    )

    # --------------------------------------------------------
    # Analytical local body-force vector
    # --------------------------------------------------------

    R_exact = (
        rho * g * A * Le / 2.0
        * np.array([
            [1.0],
            [1.0]
        ])
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    assert np.allclose(
        KLocal,
        K_exact
    )

    assert np.allclose(
        RLocal,
        R_exact
    )


# ============================================================
# Test global assembly for one L2 element
# ============================================================

def test_assemble_l2_element():

    # --------------------------------------------------------
    # One two-node element
    # --------------------------------------------------------

    EleNodes = np.array([
        1,
        2
    ])

    dofs_per_node = 1

    # Node 1 prescribed
    # Node 2 free

    GlobalID = np.array([
        [-1],
        [ 1]
    ])

    # --------------------------------------------------------
    # Local stiffness matrix
    # --------------------------------------------------------

    KLocal = np.array([
        [ 10.0, -10.0],
        [-10.0,  10.0]
    ])

    # --------------------------------------------------------
    # Local load vector
    # --------------------------------------------------------

    RLocal = np.array([
        [2.0],
        [3.0]
    ])

    # --------------------------------------------------------
    # Initialize global system
    # --------------------------------------------------------

    K_FF = np.zeros(
        (1, 1)
    )

    K_FP = np.zeros(
        (1, 1)
    )

    K_PP = np.zeros(
        (1, 1)
    )

    R_F = np.zeros(
        (1, 1)
    )

    R_P = np.zeros(
        (1, 1)
    )

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    K_FF, K_FP, K_PP, R_F, R_P = Assemble(
        dofs_per_node,
        EleNodes,
        GlobalID,
        KLocal,
        RLocal,
        K_FF,
        K_FP,
        K_PP,
        R_F,
        R_P
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    assert np.allclose(
        K_FF,
        [[10.0]]
    )

    assert np.allclose(
        K_FP,
        [[-10.0]]
    )

    assert np.allclose(
        K_PP,
        [[10.0]]
    )

    assert np.allclose(
        R_F,
        [[3.0]]
    )

    assert np.allclose(
        R_P,
        [[2.0]]
    )

def test_global_matrices_two_element_rod():

    from src.physics_models.elasticity_kernel import CalculateGlobalMatrices

    # --------------------------------------------------------
    # Two-element rod
    #
    # node 1 ---- node 2 ---- node 3
    # x = 0        0.5          1
    #
    # Node 1 prescribed
    # --------------------------------------------------------

    Connectivity = np.array([
        [1, 2],
        [2, 3]
    ])

    Coord = np.array([
        [0.0],
        [0.5],
        [1.0]
    ])

    GlobalID = np.array([
        [-1],
        [ 1],
        [ 2]
    ])

    medium_set = {
        "type": "elastic_moduli",
        "E": 1.0,
        "nu": 0.3,
        "A": 1.0
    }

    load_type = {
        "type": "none"
    }

    dim = 1
    dofs_per_node = 1
    EleType = "L2"

    NCons = 1
    Nele = 2
    NEqns = 2
    NGPTS = 2

    K_FF, K_FP, K_PP, R_F, R_P = CalculateGlobalMatrices(
        Connectivity,
        Coord,
        medium_set,
        dim,
        dofs_per_node,
        EleType,
        GlobalID,
        load_type,
        NCons,
        Nele,
        NEqns,
        NGPTS
    )

    # --------------------------------------------------------
    # Expected global matrices
    # --------------------------------------------------------

    K_FF_exact = np.array([
        [ 4.0, -2.0],
        [-2.0,  2.0]
    ])

    K_FP_exact = np.array([
        [-2.0],
        [ 0.0]
    ])

    K_PP_exact = np.array([
        [2.0]
    ])

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    assert np.allclose(
        K_FF.toarray(),
        K_FF_exact
    )

    assert np.allclose(
        K_FP.toarray(),
        K_FP_exact
    )

    assert np.allclose(
        K_PP.toarray(),
        K_PP_exact
    )

    assert np.allclose(
        R_F,
        np.zeros((2, 1))
    )

    assert np.allclose(
        R_P,
        np.zeros((1, 1))
    )

# ============================================================
# Create prescribed displacement vector
# ============================================================

def Create_ConstraintsVector(
    Constraints: np.ndarray,
    Global_ID: np.ndarray
) -> np.ndarray:
    """
    Construct the prescribed displacement vector U_P.

    Each row of Constraints has the form:

        [node_number, dof_number, prescribed_value]

    Global_ID:
        positive value -> free DOF
        negative value -> prescribed DOF
    """

    NCons = Constraints.shape[0]

    U_P = np.zeros(
        (NCons, 1),
        dtype=float
    )

    for i in range(NCons):

        node = int(
            Constraints[i, 0]
        ) - 1

        dof = int(
            Constraints[i, 1]
        ) - 1

        value = float(
            Constraints[i, 2]
        )

        constraint_id = int(
            Global_ID[node, dof]
        )

        U_P[
            abs(constraint_id) - 1,
            0
        ] = value

    return U_P

def test_create_constraints_vector():

    from src.physics_models.elasticity_kernel import (
        Create_ConstraintsVector
    )

    # --------------------------------------------------------
    # Three-node 1D problem
    #
    # Node 1: prescribed u = 0
    # Node 2: free
    # Node 3: prescribed u = 0.1
    # --------------------------------------------------------

    Global_ID = np.array([
        [-1],
        [ 1],
        [-2]
    ])

    Constraints = np.array([
        [1, 1, 0.0],
        [3, 1, 0.1]
    ])

    U_P = Create_ConstraintsVector(
        Constraints,
        Global_ID
    )

    U_P_exact = np.array([
        [0.0],
        [0.1]
    ])

    assert np.allclose(
        U_P,
        U_P_exact
    )

def test_post_processing():

    from src.physics_models.elasticity_kernel import PostProcessing

    Global_ID = np.array([
        [-1],
        [ 1],
        [ 2],
        [-2]
    ])

    U_F = np.array([
        [0.02],
        [0.05]
    ])

    U_P = np.array([
        [0.0],
        [0.10]
    ])

    U = PostProcessing(
        Global_ID,
        U_F,
        U_P
    )

    U_exact = np.array([
        [0.00],
        [0.02],
        [0.05],
        [0.10]
    ])

    assert np.allclose(
        U,
        U_exact
    )

def test_driver_single_l2_rod():

    from src.physics_models.elasticity_driver import Driver_LE

    # --------------------------------------------------------
    # One L2 element
    #
    # node 1 ---------------- node 2
    # x = 0                    x = 1
    #
    # u(0) = 0
    # constant body force b = 1
    # --------------------------------------------------------

    Connectivity = np.array([
        [1, 2]
    ])

    Coord = np.array([
        [0.0],
        [1.0]
    ])

    Constraints = np.array([
        [1, 1, 0.0]
    ])

    medium_set = {
        "type": "elastic_moduli",
        "E": 1.0,
        "nu": 0.3,
        "A": 1.0
    }

    load_type = {
        "type": "constant",
        "b": [1.0]
    }

    dim = 1
    dofs_per_node = 1
    EleType = "L2"

    NCons = 1
    Nele = 1
    NGPTS = 2
    NumNodes = 2

    # --------------------------------------------------------
    # Solve
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
        Nele,
        NGPTS,
        NumNodes
    )

    # --------------------------------------------------------
    # Analytical nodal displacement
    #
    # For E = A = L = b = 1:
    #
    # u(x) = x - x^2 / 2
    #
    # u(0) = 0
    # u(1) = 0.5
    # --------------------------------------------------------

    U_exact = np.array([
        [0.0],
        [0.5]
    ])

    assert np.allclose(
        U,
        U_exact
    )

def test_plane_stress_constitutive_matrix():

    from src.physics_models.elasticity_kernel import (
        Get_ConstitutiveMatrix
    )

    medium_set = {
        "type": "elastic_moduli",
        "E": 200.0,
        "nu": 0.25,
        "analysis_type": "plane_stress"
    }

    C = Get_ConstitutiveMatrix(
        medium_set
    )

    assert C.shape == (3, 3)

    assert np.allclose(
        C,
        C.T
    )


def test_plane_strain_constitutive_matrix():

    from src.physics_models.elasticity_kernel import (
        Get_ConstitutiveMatrix
    )

    medium_set = {
        "type": "elastic_moduli",
        "E": 200.0,
        "nu": 0.25,
        "analysis_type": "plane_strain"
    }

    C = Get_ConstitutiveMatrix(
        medium_set
    )

    assert C.shape == (3, 3)

    assert np.allclose(
        C,
        C.T
    )

def test_q8_local_stiffness_matrix():

    from src.fem.gauss_quadrature import GaussPoints

    # --------------------------------------------------------
    # Q8 rectangle
    #
    # 4 ----- 7 ----- 3
    # |               |
    # 8               6
    # |               |
    # 1 ----- 5 ----- 2
    # --------------------------------------------------------

    xCap = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.1],
        [0.0, 0.1],
        [0.5, 0.0],
        [1.0, 0.05],
        [0.5, 0.1],
        [0.0, 0.05]
    ])

    EleNodes = np.arange(
        1,
        9
    )

    medium_set = {
        "type": "elastic_moduli",
        "E": 200.0,
        "nu": 0.3,
        "analysis_type": "plane_stress",
        "thickness": 1.0
    }

    load_type = {
        "type": "none"
    }

    r, w = GaussPoints(
        dim=2,
        EleType="Q8",
        NGPTS=3
    )

    KLocal, RLocal = CalculateLocalMatrices(
        medium_set,
        2,
        EleNodes,
        "Q8",
        load_type,
        r,
        w,
        xCap
    )


    # 8 nodes x 2 DOFs
    assert KLocal.shape == (
        16,
        16
    )

    assert RLocal.shape == (
        16,
        1
    )


    # Stiffness matrix must be symmetric
    assert np.allclose(
        KLocal,
        KLocal.T,
        atol=1e-10
    )


    # No body force
    assert np.allclose(
        RLocal,
        0.0
    )


    # Values should all be finite
    assert np.all(
        np.isfinite(
            KLocal
        )
    )

def test_q8_rigid_body_modes():

    from src.fem.gauss_quadrature import GaussPoints

    # --------------------------------------------------------
    # Q8 rectangular element
    #
    # 4 ----- 7 ----- 3
    # |               |
    # 8               6
    # |               |
    # 1 ----- 5 ----- 2
    # --------------------------------------------------------

    xCap = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.1],
        [0.0, 0.1],
        [0.5, 0.0],
        [1.0, 0.05],
        [0.5, 0.1],
        [0.0, 0.05]
    ])


    EleNodes = np.arange(
        1,
        9
    )


    medium_set = {
        "type": "elastic_moduli",
        "E": 200.0,
        "nu": 0.3,
        "analysis_type": "plane_stress",
        "thickness": 1.0
    }


    load_type = {
        "type": "none"
    }


    r, w = GaussPoints(
        dim=2,
        EleType="Q8",
        NGPTS=3
    )


    KLocal, _ = CalculateLocalMatrices(
        medium_set,
        2,
        EleNodes,
        "Q8",
        load_type,
        r,
        w,
        xCap
    )


    # ========================================================
    # Rigid translation in x
    # ========================================================

    rigid_x = np.zeros(
        16
    )

    rigid_x[0::2] = 1.0


    # ========================================================
    # Rigid translation in y
    # ========================================================

    rigid_y = np.zeros(
        16
    )

    rigid_y[1::2] = 1.0


    # ========================================================
    # Rigid rotation
    #
    # u = -y
    # v =  x
    # ========================================================

    rigid_rotation = np.zeros(
        16
    )


    for node in range(8):

        x = xCap[node, 0]
        y = xCap[node, 1]

        rigid_rotation[
            2 * node
        ] = -y

        rigid_rotation[
            2 * node + 1
        ] = x


    # ========================================================
    # Internal force must vanish for rigid-body motion
    # ========================================================

    force_x = (
        KLocal
        @ rigid_x
    )

    force_y = (
        KLocal
        @ rigid_y
    )

    force_rotation = (
        KLocal
        @ rigid_rotation
    )


    assert np.allclose(
        force_x,
        0.0,
        atol=1e-9
    )

    assert np.allclose(
        force_y,
        0.0,
        atol=1e-9
    )

    assert np.allclose(
        force_rotation,
        0.0,
        atol=1e-9
    )