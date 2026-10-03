"""
Elasticity Finite Element Kernel

Core finite element routines for linear elasticity.

Current capabilities
--------------------
1D:
    - L2 rod elements
    - Axial deformation
    - Constant or gravitational body forces

2D:
    - Q4 and Q8 elements
    - Plane stress
    - Plane strain
    - Vector body forces

The kernel contains:
    - Material-property handling
    - Constitutive matrix construction
    - Body-force evaluation
    - Local element matrices
    - Global assembly
    - Constraint handling
    - Solution post-processing
"""

import numpy as np

from typing import Dict, Any, Tuple

from scipy.sparse import lil_matrix

from src.fem.gauss_quadrature import GaussPoints
from src.fem.shape_functions import ShapeFunctions


# ============================================================
# Elastic material properties
# ============================================================

def Get_ElasticModuli(
    medium_set: Dict[str, Any]
) -> Tuple[float, float, float, float]:
    """
    Return elastic constants:

        E       Young's modulus
        nu      Poisson's ratio
        lam     First Lame parameter
        mu      Shear modulus

    Supported material descriptions:

        {
            "type": "elastic_moduli",
            "E": ...,
            "nu": ...
        }

        {
            "type": "young_poisson",
            "E": ...,
            "nu": ...
        }

        {
            "type": "Lame_params",
            "lambda": ...,
            "mu": ...
        }
    """

    if "type" not in medium_set:
        raise ValueError(
            "medium_set must contain a material 'type'."
        )

    material_type = str(
        medium_set["type"]
    ).lower()


    # ========================================================
    # Young's modulus and Poisson's ratio
    # ========================================================

    if material_type in (
        "elastic_moduli",
        "young_poisson"
    ):

        if "E" not in medium_set:
            raise ValueError(
                "Young's modulus E is required."
            )

        if "nu" not in medium_set:
            raise ValueError(
                "Poisson's ratio nu is required."
            )

        E = float(
            medium_set["E"]
        )

        nu = float(
            medium_set["nu"]
        )


        if E <= 0.0:
            raise ValueError(
                "Young's modulus E must be positive."
            )

        if not (
            -1.0 < nu < 0.5
        ):
            raise ValueError(
                "Poisson's ratio must satisfy "
                "-1 < nu < 0.5."
            )


        lam = (
            E * nu
            / (
                (1.0 + nu)
                * (1.0 - 2.0 * nu)
            )
        )

        mu = (
            E
            / (
                2.0
                * (1.0 + nu)
            )
        )


        return (
            E,
            nu,
            lam,
            mu
        )


    # ========================================================
    # Lame parameters
    # ========================================================

    elif material_type == "lame_params":

        if "lambda" not in medium_set:
            raise ValueError(
                "Lame parameter lambda is required."
            )

        if "mu" not in medium_set:
            raise ValueError(
                "Lame parameter mu is required."
            )


        lam = float(
            medium_set["lambda"]
        )

        mu = float(
            medium_set["mu"]
        )


        if mu <= 0.0:
            raise ValueError(
                "Shear modulus mu must be positive."
            )

        if (
            3.0 * lam
            + 2.0 * mu
        ) <= 0.0:
            raise ValueError(
                "Invalid Lame parameters."
            )


        E = (
            mu
            * (
                3.0 * lam
                + 2.0 * mu
            )
            / (
                lam
                + mu
            )
        )

        nu = (
            lam
            / (
                2.0
                * (
                    lam
                    + mu
                )
            )
        )


        return (
            E,
            nu,
            lam,
            mu
        )


    # ========================================================
    # Unsupported material
    # ========================================================

    else:

        raise ValueError(
            f"Unsupported material type: "
            f"{medium_set['type']}"
        )


# ============================================================
# 2D constitutive matrix
# ============================================================

def Get_ConstitutiveMatrix(
    medium_set: Dict[str, Any]
) -> np.ndarray:
    """
    Construct the 2D isotropic linear-elastic
    constitutive matrix.

    Supported analysis types:

        plane_stress
        plane_strain
    """

    E, nu, lam, mu = Get_ElasticModuli(
        medium_set
    )


    analysis_type = str(
        medium_set.get(
            "analysis_type",
            ""
        )
    ).lower()


    # ========================================================
    # Plane stress
    # ========================================================

    if analysis_type == "plane_stress":

        factor = (
            E
            / (
                1.0 - nu**2
            )
        )


        C = factor * np.array([
            [
                1.0,
                nu,
                0.0
            ],
            [
                nu,
                1.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.5 * (1.0 - nu)
            ]
        ])


        return C


    # ========================================================
    # Plane strain
    # ========================================================

    elif analysis_type == "plane_strain":

        C = np.array([
            [
                lam + 2.0 * mu,
                lam,
                0.0
            ],
            [
                lam,
                lam + 2.0 * mu,
                0.0
            ],
            [
                0.0,
                0.0,
                mu
            ]
        ])


        return C


    # ========================================================
    # Unsupported analysis type
    # ========================================================

    else:

        raise ValueError(
            "medium_set must specify "
            "'analysis_type' as either "
            "'plane_stress' or 'plane_strain'."
        )


# ============================================================
# Body force
# ============================================================

def Get_BodyForce(
    load_type: Dict[str, Any],
    x: np.ndarray,
    dim: int
) -> np.ndarray:
    """
    Evaluate the body-force vector.

    Supported load descriptions:

        {"type": "none"}

        {
            "type": "constant",
            "b": [...]
        }

        {
            "type": "gravity",
            "rho": ...,
            "g": ...
        }

        {
            "type": "function",
            "b": callable
        }
    """

    if "type" not in load_type:
        raise ValueError(
            "load_type must contain a 'type'."
        )


    force_type = str(
        load_type["type"]
    ).lower()


    # ========================================================
    # No body force
    # ========================================================

    if force_type == "none":

        return np.zeros(
            dim,
            dtype=float
        )


    # ========================================================
    # Constant body force
    # ========================================================

    elif force_type == "constant":

        if "b" not in load_type:
            raise ValueError(
                "Constant body force requires 'b'."
            )


        b = np.asarray(
            load_type["b"],
            dtype=float
        ).reshape(-1)


        if b.size != dim:
            raise ValueError(
                f"Body force must contain "
                f"{dim} component(s)."
            )


        return b


    # ========================================================
    # Gravity
    # ========================================================

    elif force_type == "gravity":

        if "rho" not in load_type:
            raise ValueError(
                "Gravity load requires density rho."
            )

        if "g" not in load_type:
            raise ValueError(
                "Gravity load requires acceleration g."
            )


        rho = float(
            load_type["rho"]
        )

        g = np.asarray(
            load_type["g"],
            dtype=float
        )


        # ----------------------------------------------------
        # 1D gravity
        # ----------------------------------------------------

        if dim == 1:

            if g.size != 1:
                raise ValueError(
                    "1D gravity requires scalar g."
                )

            return np.array([
                rho * float(g.reshape(-1)[0])
            ])


        # ----------------------------------------------------
        # Multi-dimensional gravity
        # ----------------------------------------------------

        g = g.reshape(-1)


        if g.size != dim:
            raise ValueError(
                f"{dim}D gravity requires a "
                f"{dim}-component gravity vector."
            )


        return (
            rho * g
        )


    # ========================================================
    # User-defined function
    # ========================================================

    elif force_type == "function":

        if "b" not in load_type:

            raise ValueError(
                "Functional body force requires callable 'b'."
            )


        body_force_function = load_type["b"]


        if not callable(
            body_force_function
        ):

            raise TypeError(
                "load_type['b'] must be callable."
            )


        coordinates = np.asarray(
            x,
            dtype=float
        ).reshape(-1)


        b = body_force_function(
            *coordinates
        )


        b = np.asarray(
            b,
            dtype=float
        ).reshape(-1)


        if b.size != dim:

            raise ValueError(
                f"Body-force function must return "
                f"{dim} component(s)."
            )


        return b


    # ========================================================
    # Unsupported load
    # ========================================================

    else:

        raise ValueError(
            f"Unsupported body-force type: "
            f"{load_type['type']}"
        )


# ============================================================
# Local element matrices
# ============================================================

def CalculateLocalMatrices(
    medium_set: Dict[str, Any],
    dofs_per_node: int,
    EleNodes: np.ndarray,
    EleType: str,
    load_type: Dict[str, Any],
    r: np.ndarray,
    w: np.ndarray,
    xCap: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the local stiffness matrix and
    local load vector.

    Supported formulations:

        1D elasticity using L2 elements

        2D elasticity using Q4/Q8 elements
    """

    # --------------------------------------------------------
    # Basic element information
    # --------------------------------------------------------

    NodesPerEle = len(
        EleNodes
    )

    dim = xCap.shape[1]

    n_dofs = (
        NodesPerEle
        * dofs_per_node
    )


    # --------------------------------------------------------
    # Initialize local matrices
    # --------------------------------------------------------

    KLocal = np.zeros(
        (
            n_dofs,
            n_dofs
        ),
        dtype=float
    )

    RLocal = np.zeros(
        (
            n_dofs,
            1
        ),
        dtype=float
    )


    # ========================================================
    # 1D elasticity
    # ========================================================

    if dim == 1:

        if dofs_per_node != 1:

            raise ValueError(
                "1D elasticity requires "
                "one DOF per node."
            )


        # ----------------------------------------------------
        # Material properties
        # ----------------------------------------------------

        E, _, _, _ = Get_ElasticModuli(
            medium_set
        )


        # ----------------------------------------------------
        # Cross-sectional area
        # ----------------------------------------------------

        A = float(
            medium_set.get(
                "A",
                1.0
            )
        )


        if A <= 0.0:

            raise ValueError(
                "Cross-sectional area A "
                "must be positive."
            )


        # ----------------------------------------------------
        # Gauss integration
        # ----------------------------------------------------

        for gpt in range(
            len(w)
        ):

            zeta = np.asarray(
                [r[gpt]],
                dtype=float
            )


            # ------------------------------------------------
            # Shape functions
            # ------------------------------------------------

            N, DN = ShapeFunctions(
                EleType,
                zeta
            )


            # ------------------------------------------------
            # Physical coordinate
            # ------------------------------------------------

            x = (
                xCap.T
                @ N.T
            )


            # ------------------------------------------------
            # Jacobian
            # ------------------------------------------------

            J = (
                xCap.T
                @ DN
            )


            detJ = float(
                np.linalg.det(J)
            )


            if detJ <= 0.0:

                raise ValueError(
                    "Element Jacobian "
                    "must be positive."
                )


            # ------------------------------------------------
            # Spatial derivatives
            # ------------------------------------------------

            dNdx = (
                DN
                @ np.linalg.inv(J)
            )


            # ------------------------------------------------
            # Strain-displacement matrix
            # ------------------------------------------------

            B = dNdx.T


            # ------------------------------------------------
            # Local stiffness
            #
            # K_e = integral B^T E B A dx
            # ------------------------------------------------

            KLocal += (
                w[gpt]
                * (
                    B.T
                    @ B
                )
                * E
                * A
                * detJ
            )


            # ------------------------------------------------
            # Body force
            # ------------------------------------------------

            b = Get_BodyForce(
                load_type,
                x,
                dim
            )


            b_scalar = float(
                b[0]
            )


            # ------------------------------------------------
            # Local load
            #
            # R_e = integral N^T b A dx
            # ------------------------------------------------

            RLocal += (
                w[gpt]
                * N.T
                * b_scalar
                * A
                * detJ
            )


        return (
            KLocal,
            RLocal
        )


    # ========================================================
    # 2D elasticity
    # ========================================================

    elif dim == 2:

        if dofs_per_node != 2:

            raise ValueError(
                "2D elasticity requires "
                "two DOFs per node."
            )


        # ----------------------------------------------------
        # Constitutive matrix
        # ----------------------------------------------------

        C = Get_ConstitutiveMatrix(
            medium_set
        )


        # ----------------------------------------------------
        # Out-of-plane thickness
        # ----------------------------------------------------

        thickness = float(
            medium_set.get(
                "thickness",
                1.0
            )
        )


        if thickness <= 0.0:

            raise ValueError(
                "Element thickness "
                "must be positive."
            )


        # ----------------------------------------------------
        # Gauss integration
        # ----------------------------------------------------

        for gpt in range(
            len(w)
        ):

            zeta = np.asarray(
                r[gpt],
                dtype=float
            )


            # ------------------------------------------------
            # Shape functions
            #
            # N:
            #     1 x NodesPerEle
            #
            # DN:
            #     NodesPerEle x 2
            # ------------------------------------------------

            N, DN = ShapeFunctions(
                EleType,
                zeta
            )


            # ------------------------------------------------
            # Physical Gauss-point coordinate
            # ------------------------------------------------

            x = (
                N
                @ xCap
            ).reshape(-1)


            # ------------------------------------------------
            # Jacobian
            #
            # J =
            #
            # [ dx/dxi    dy/dxi  ]
            # [ dx/deta   dy/deta ]
            # ------------------------------------------------

            J = (
                DN.T
                @ xCap
            )


            detJ = float(
                np.linalg.det(J)
            )


            if detJ <= 0.0:

                raise ValueError(
                    "Element Jacobian "
                    "must be positive."
                )


            # ------------------------------------------------
            # Shape derivatives in physical coordinates
            #
            # dNdx[:,0] = dN/dx
            # dNdx[:,1] = dN/dy
            #
            # Because:
            #
            # J = DN.T @ xCap
            #
            # the row-wise gradient transformation is:
            #
            # dNdx = DN @ inv(J).T
            # ------------------------------------------------

            dNdx = (
                DN
                @ np.linalg.inv(J).T
            )


            # ------------------------------------------------
            # Strain-displacement matrix
            #
            # strain =
            #
            # [ epsilon_xx ]
            # [ epsilon_yy ]
            # [ gamma_xy   ]
            # ------------------------------------------------

            B = np.zeros(
                (
                    3,
                    n_dofs
                ),
                dtype=float
            )


            for a in range(
                NodesPerEle
            ):

                dN_dx = dNdx[
                    a,
                    0
                ]

                dN_dy = dNdx[
                    a,
                    1
                ]


                u_dof = (
                    2 * a
                )

                v_dof = (
                    2 * a + 1
                )


                # epsilon_xx
                B[
                    0,
                    u_dof
                ] = dN_dx


                # epsilon_yy
                B[
                    1,
                    v_dof
                ] = dN_dy


                # gamma_xy
                B[
                    2,
                    u_dof
                ] = dN_dy

                B[
                    2,
                    v_dof
                ] = dN_dx


            # ------------------------------------------------
            # Shape-function matrix for vector body forces
            #
            # Nmat =
            #
            # [ N1  0   N2  0  ... ]
            # [ 0   N1  0   N2 ... ]
            # ------------------------------------------------

            Nmat = np.zeros(
                (
                    2,
                    n_dofs
                ),
                dtype=float
            )


            for a in range(
                NodesPerEle
            ):

                Nmat[
                    0,
                    2 * a
                ] = N[0, a]

                Nmat[
                    1,
                    2 * a + 1
                ] = N[0, a]


            # ------------------------------------------------
            # Local stiffness
            #
            # K_e = integral B^T C B t dA
            # ------------------------------------------------

            KLocal += (
                w[gpt]
                * (
                    B.T
                    @ C
                    @ B
                )
                * thickness
                * detJ
            )


            # ------------------------------------------------
            # Body force
            # ------------------------------------------------

            b = Get_BodyForce(
                load_type,
                x,
                dim
            )


            b = np.asarray(
                b,
                dtype=float
            ).reshape(
                2,
                1
            )


            # ------------------------------------------------
            # Local body-force vector
            #
            # R_e = integral N^T b t dA
            # ------------------------------------------------

            RLocal += (
                w[gpt]
                * (
                    Nmat.T
                    @ b
                )
                * thickness
                * detJ
            )


        return (
            KLocal,
            RLocal
        )


    # ========================================================
    # Unsupported dimension
    # ========================================================

    else:

        raise NotImplementedError(
            f"Elasticity is not implemented "
            f"for dimension {dim}."
        )


# ============================================================
# Global assembly
# ============================================================

def Assemble(
    dofs_per_node: int,
    EleNodes: np.ndarray,
    GlobalID: np.ndarray,
    KLocal: np.ndarray,
    RLocal: np.ndarray,
    K_FF,
    K_FP,
    K_PP,
    R_F: np.ndarray,
    R_P: np.ndarray
):
    """
    Assemble one local element into the partitioned
    global system.

    GlobalID convention:

        positive ID:
            free degree of freedom

        negative ID:
            prescribed degree of freedom
    """

    # --------------------------------------------------------
    # Number of nodes in the element
    # --------------------------------------------------------

    NodesPerEle = len(
        EleNodes
    )


    # --------------------------------------------------------
    # Local-to-global equation vector
    # --------------------------------------------------------

    v_vector = np.zeros(
        NodesPerEle * dofs_per_node,
        dtype=int
    )


    local_index = 0


    for local_node in range(
        NodesPerEle
    ):

        global_node = (
            int(
                EleNodes[local_node]
            )
            - 1
        )


        for dof in range(
            dofs_per_node
        ):

            v_vector[
                local_index
            ] = GlobalID[
                global_node,
                dof
            ]

            local_index += 1


    # --------------------------------------------------------
    # Number of local DOFs
    # --------------------------------------------------------

    n_local_dofs = len(
        v_vector
    )


    # ========================================================
    # Assembly loop
    # ========================================================

    for i_local in range(
        n_local_dofs
    ):

        row_id = int(
            v_vector[i_local]
        )


        # ====================================================
        # Free row
        # ====================================================

        if row_id > 0:

            row_F = (
                row_id - 1
            )


            # ------------------------------------------------
            # Load vector
            # ------------------------------------------------

            R_F[
                row_F,
                0
            ] += RLocal[
                i_local,
                0
            ]


            # ------------------------------------------------
            # Stiffness matrix
            # ------------------------------------------------

            for j_local in range(
                n_local_dofs
            ):

                col_id = int(
                    v_vector[j_local]
                )


                # --------------------------------------------
                # Free-free block
                # --------------------------------------------

                if col_id > 0:

                    col_F = (
                        col_id - 1
                    )

                    K_FF[
                        row_F,
                        col_F
                    ] += KLocal[
                        i_local,
                        j_local
                    ]


                # --------------------------------------------
                # Free-prescribed block
                # --------------------------------------------

                else:

                    col_P = (
                        -col_id - 1
                    )

                    K_FP[
                        row_F,
                        col_P
                    ] += KLocal[
                        i_local,
                        j_local
                    ]


        # ====================================================
        # Prescribed row
        # ====================================================

        else:

            row_P = (
                -row_id - 1
            )


            # ------------------------------------------------
            # Load vector
            # ------------------------------------------------

            R_P[
                row_P,
                0
            ] += RLocal[
                i_local,
                0
            ]


            # ------------------------------------------------
            # Prescribed-prescribed block
            # ------------------------------------------------

            for j_local in range(
                n_local_dofs
            ):

                col_id = int(
                    v_vector[j_local]
                )


                if col_id < 0:

                    col_P = (
                        -col_id - 1
                    )

                    K_PP[
                        row_P,
                        col_P
                    ] += KLocal[
                        i_local,
                        j_local
                    ]


    return (
        K_FF,
        K_FP,
        K_PP,
        R_F,
        R_P
    )

# ============================================================
# Global matrices
# ============================================================

def CalculateGlobalMatrices(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    medium_set: Dict[str, Any],
    dim: int,
    dofs_per_node: int,
    EleType: str,
    GlobalID: np.ndarray,
    load_type: Dict[str, Any],
    NCons: int,
    Nele: int,
    NEqns: int,
    NGPTS: int
):
    """
    Assemble the complete partitioned global system.

    Returns:

        K_FF
        K_FP
        K_PP
        R_F
        R_P
    """

    # --------------------------------------------------------
    # Global matrices
    # --------------------------------------------------------

    K_FF = lil_matrix(
        (
            NEqns,
            NEqns
        ),
        dtype=float
    )


    K_FP = lil_matrix(
        (
            NEqns,
            NCons
        ),
        dtype=float
    )


    K_PP = lil_matrix(
        (
            NCons,
            NCons
        ),
        dtype=float
    )


    R_F = np.zeros(
        (
            NEqns,
            1
        ),
        dtype=float
    )


    R_P = np.zeros(
        (
            NCons,
            1
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # Gauss points
    # --------------------------------------------------------

    r, w = GaussPoints(
        dim,
        EleType,
        NGPTS
    )


    # --------------------------------------------------------
    # Element loop
    # --------------------------------------------------------

    for ele in range(
        Nele
    ):

        EleNodes = np.asarray(
            Connectivity[
                ele,
                :
            ],
            dtype=int
        )


        # ----------------------------------------------------
        # Element coordinates
        #
        # Connectivity uses 1-based node numbering.
        # ----------------------------------------------------

        xCap = Coord[
            EleNodes - 1,
            :
        ]


        # ----------------------------------------------------
        # Local matrices
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Global assembly
        # ----------------------------------------------------

        (
            K_FF,
            K_FP,
            K_PP,
            R_F,
            R_P
        ) = Assemble(
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
    # Convert stiffness matrices to CSR
    # --------------------------------------------------------

    K_FF = K_FF.tocsr()
    K_FP = K_FP.tocsr()
    K_PP = K_PP.tocsr()


    return (
        K_FF,
        K_FP,
        K_PP,
        R_F,
        R_P
    )


# ============================================================
# Prescribed displacement vector
# ============================================================

def Create_ConstraintsVector(
    Constraints: np.ndarray,
    GlobalID: np.ndarray
) -> np.ndarray:
    """
    Construct the prescribed displacement vector U_P.

    Each row of Constraints has the form:

        [node, dof, prescribed_value]

    Node and DOF numbering are 1-based.
    """

    NCons = Constraints.shape[0]


    U_P = np.zeros(
        (
            NCons,
            1
        ),
        dtype=float
    )


    for constraint in Constraints:

        node = (
            int(
                constraint[0]
            )
            - 1
        )

        dof = (
            int(
                constraint[1]
            )
            - 1
        )

        value = float(
            constraint[2]
        )


        equation_id = int(
            GlobalID[
                node,
                dof
            ]
        )


        if equation_id >= 0:

            raise ValueError(
                "Constraint refers to a DOF "
                "that is not marked as prescribed "
                "in GlobalID."
            )


        prescribed_index = (
            -equation_id
            - 1
        )


        U_P[
            prescribed_index,
            0
        ] = value


    return U_P


# ============================================================
# Post-processing
# ============================================================

def PostProcessing(
    GlobalID: np.ndarray,
    U_F: np.ndarray,
    U_P: np.ndarray
) -> np.ndarray:
    """
    Reconstruct the complete nodal displacement matrix
    from the free and prescribed displacement vectors.

    Output shape:

        NumNodes x dofs_per_node
    """

    NumNodes = GlobalID.shape[0]

    dofs_per_node = GlobalID.shape[1]


    U = np.zeros(
        (
            NumNodes,
            dofs_per_node
        ),
        dtype=float
    )


    U_F = np.asarray(
        U_F,
        dtype=float
    ).reshape(-1)


    U_P = np.asarray(
        U_P,
        dtype=float
    ).reshape(-1)


    # --------------------------------------------------------
    # Reconstruct nodal displacement field
    # --------------------------------------------------------

    for node in range(
        NumNodes
    ):

        for dof in range(
            dofs_per_node
        ):

            equation_id = int(
                GlobalID[
                    node,
                    dof
                ]
            )


            # ------------------------------------------------
            # Free DOF
            # ------------------------------------------------

            if equation_id > 0:

                U[
                    node,
                    dof
                ] = U_F[
                    equation_id - 1
                ]


            # ------------------------------------------------
            # Prescribed DOF
            # ------------------------------------------------

            else:

                U[
                    node,
                    dof
                ] = U_P[
                    -equation_id - 1
                ]


    return U