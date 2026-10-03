"""
Driver for linearized elasticity finite element problems.
"""

import numpy as np

from scipy.sparse.linalg import spsolve

from src.fem.create_id import create_id_matrix

from src.physics_models.elasticity_kernel import (
    CalculateGlobalMatrices,
    Create_ConstraintsVector,
    PostProcessing,
)


# ============================================================
# Apply nodal loads
# ============================================================

def Apply_NodalLoads(
    NodalLoads,
    GlobalID,
    R_F,
    R_P,
):
    """
    Add externally applied nodal forces to the global
    load vectors.

    Parameters
    ----------
    NodalLoads : np.ndarray or None
        Rows have the form

            [node, dof, value]

        Node and DOF numbering are 1-based.

    GlobalID : np.ndarray
        Global equation-number matrix.

    R_F : np.ndarray
        Global load vector for free degrees of freedom.

    R_P : np.ndarray
        Global load vector for prescribed degrees of freedom.

    Returns
    -------
    R_F : np.ndarray
        Updated free-DOF load vector.

    R_P : np.ndarray
        Updated prescribed-DOF load vector.
    """

    # --------------------------------------------------------
    # No external nodal loads
    # --------------------------------------------------------

    if NodalLoads is None:

        return (
            R_F,
            R_P,
        )


    # --------------------------------------------------------
    # Convert input
    # --------------------------------------------------------

    NodalLoads = np.asarray(
        NodalLoads,
        dtype=float,
    )


    if NodalLoads.size == 0:

        return (
            R_F,
            R_P,
        )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        NodalLoads.ndim != 2
        or NodalLoads.shape[1] != 3
    ):

        raise ValueError(
            "NodalLoads must have shape (n, 3) "
            "with rows [node, dof, value]."
        )


    NumNodes = GlobalID.shape[0]

    dofs_per_node = GlobalID.shape[1]


    # --------------------------------------------------------
    # Apply each nodal load
    # --------------------------------------------------------

    for load in NodalLoads:

        node = int(
            load[0]
        )

        dof = int(
            load[1]
        )

        value = float(
            load[2]
        )


        # ----------------------------------------------------
        # Validate node
        # ----------------------------------------------------

        if (
            node < 1
            or node > NumNodes
        ):

            raise ValueError(
                "Nodal load contains an invalid node number."
            )


        # ----------------------------------------------------
        # Validate DOF
        # ----------------------------------------------------

        if (
            dof < 1
            or dof > dofs_per_node
        ):

            raise ValueError(
                "Nodal load contains an invalid DOF number."
            )


        # ----------------------------------------------------
        # Obtain global equation number
        # ----------------------------------------------------

        equation_id = int(
            GlobalID[
                node - 1,
                dof - 1,
            ]
        )


        # ----------------------------------------------------
        # Free DOF
        # ----------------------------------------------------

        if equation_id > 0:

            R_F[
                equation_id - 1,
                0,
            ] += value


        # ----------------------------------------------------
        # Prescribed DOF
        # ----------------------------------------------------

        else:

            prescribed_id = (
                -equation_id
            )

            R_P[
                prescribed_id - 1,
                0,
            ] += value


    return (
        R_F,
        R_P,
    )


# ============================================================
# Linear elasticity driver
# ============================================================

def Driver_LE(
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
    NumNodes,
    NodalLoads=None,
):
    """
    Solve a linearized elasticity finite element problem.

    Parameters
    ----------
    Connectivity : np.ndarray
        Element connectivity matrix.

    Constraints : np.ndarray
        Prescribed displacement conditions with rows

            [node, dof, value]

    Coord : np.ndarray
        Nodal coordinates.

    medium_set : dict
        Material properties.

    dim : int
        Spatial dimension.

    dofs_per_node : int
        Number of displacement DOFs per node.

    EleType : str
        Finite element type.

    load_type : dict
        Volumetric/body-force definition.

    NCons : int
        Number of prescribed constraints.

    Nele : int
        Number of elements.

    NGPTS : int
        Number of Gaussian points per coordinate direction.

    NumNodes : int
        Number of mesh nodes.

    NodalLoads : np.ndarray or None
        Optional external nodal-force array with rows

            [node, dof, value]

        This is used for concentrated loads or equivalent
        nodal forces obtained from boundary traction
        integration.

    Returns
    -------
    U : np.ndarray
        Nodal displacement matrix.
    """

    # --------------------------------------------------------
    # Equation numbering
    # --------------------------------------------------------

    GlobalID, NEqns = create_id_matrix(
        Constraints,
        dofs_per_node,
        NumNodes,
    )


    # --------------------------------------------------------
    # Prescribed displacement vector
    # --------------------------------------------------------

    U_P = Create_ConstraintsVector(
        Constraints,
        GlobalID,
    )


    # --------------------------------------------------------
    # Assemble stiffness matrices and body-force vectors
    # --------------------------------------------------------

    (
        K_FF,
        K_FP,
        K_PP,
        R_F,
        R_P,
    ) = CalculateGlobalMatrices(
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
        NGPTS,
    )


    # --------------------------------------------------------
    # Add nodal / boundary loads
    # --------------------------------------------------------

    (
        R_F,
        R_P,
    ) = Apply_NodalLoads(
        NodalLoads,
        GlobalID,
        R_F,
        R_P,
    )


    # --------------------------------------------------------
    # Reduced right-hand side
    #
    # K_FF U_F + K_FP U_P = R_F
    #
    # therefore
    #
    # K_FF U_F = R_F - K_FP U_P
    # --------------------------------------------------------

    RHS = (
        R_F
        - K_FP
        @ U_P
    )


    # --------------------------------------------------------
    # Solve free degrees of freedom
    # --------------------------------------------------------

    U_F = spsolve(
        K_FF,
        RHS.reshape(-1),
    )


    # --------------------------------------------------------
    # Reconstruct complete nodal displacement matrix
    # --------------------------------------------------------

    U = PostProcessing(
        GlobalID,
        U_F,
        U_P,
    )


    return U