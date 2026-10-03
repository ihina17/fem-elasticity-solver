# Finite Element Solver for Linear Elasticity

**Ihina Mahajan**  
Materials Science and Engineering  
University of Houston

A finite element solver for one- and two-dimensional linear elasticity problems implemented in Python.

The repository contains reusable finite element routines together with benchmark problems for verifying element behavior, numerical convergence, stress recovery, bending response, and nearly incompressible elasticity.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ihina17/fem-elasticity-solver.git
cd fem-elasticity-solver
```

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Examples

All problem scripts should be run from the repository root.

### Rod under self-weight

```bash
python -m problems.rod_self_weight
```

### Cantilever beam

```bash
python -m problems.cantilever_beam
```

### Thick cylinder

```bash
python -m problems.thick_cylinder
```

### Cook's membrane

```bash
python -m problems.cooks_problem
```

The generated plots are saved automatically in the `figures/` directory.

---

## Running the Tests

Run the complete test suite with:

```bash
python -m pytest tests -v
```

The tests cover the finite element utilities, mesh generators, element matrices, global assembly, constitutive models, boundary loads, analytical verification problems, and complete FEM solutions.

---

## Features

The current implementation includes:

- L2 linear elements for one-dimensional elasticity
- Q4 bilinear quadrilateral elements
- Q8 serendipity quadrilateral elements
- Gaussian quadrature
- Shape-function evaluation
- Sparse global stiffness assembly
- Prescribed displacement boundary conditions
- Body-force loading
- Concentrated nodal forces
- Distributed boundary tractions
- Plane-stress elasticity
- Plane-strain elasticity
- Stress recovery
- Error calculations
- Mesh-refinement studies

---

## Repository Structure

```text
fem-elasticity-solver/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── fem/
│   │   ├── create_id.py
│   │   ├── gauss_quadrature.py
│   │   ├── shape_functions.py
│   │   └── Vector.py
│   │
│   ├── meshes/
│   │   ├── L2.py
│   │   ├── Q4.py
│   │   ├── Q8.py
│   │   ├── annulus_Q4.py
│   │   └── cooks_Q4.py
│   │
│   ├── physics_models/
│   │   ├── elasticity_driver.py
│   │   ├── elasticity_kernel.py
│   │   └── boundary_loads.py
│   │
│   └── error_calculation.py
│
├── problems/
│   ├── rod_self_weight.py
│   ├── cantilever_beam.py
│   ├── thick_cylinder.py
│   └── cooks_problem.py
│
├── tests/
│
└── figures/
```

---

# Finite Element Formulation

For small-strain linear elasticity, the governing equilibrium equation is

\[
\nabla \cdot \boldsymbol{\sigma}
+
\mathbf{b}
=
0.
\]

The infinitesimal strain tensor is

\[
\boldsymbol{\varepsilon}
=
\frac{1}{2}
\left(
\nabla \mathbf{u}
+
\nabla \mathbf{u}^{T}
\right),
\]

with constitutive relation

\[
\boldsymbol{\sigma}
=
\mathbb{C}
:
\boldsymbol{\varepsilon}.
\]

After spatial discretization, the finite element equations take the form

\[
\mathbf{K}\mathbf{U}
=
\mathbf{F},
\]

where the element stiffness matrix is

\[
\mathbf{K}^{e}
=
\int_{\Omega_e}
\mathbf{B}^{T}
\mathbf{C}
\mathbf{B}
\,d\Omega.
\]

The assembled global system is stored and solved using SciPy sparse linear algebra.

---

# Benchmark Problems

## 1. Rod Under Self-Weight

The first verification problem is a one-dimensional elastic rod subjected to gravitational loading.

The analytical displacement is

\[
u(x)
=
\frac{\rho g}{E}
\left(
Lx-\frac{x^2}{2}
\right),
\]

and the analytical axial stress is

\[
\sigma(x)
=
\rho g(L-x).
\]

The problem verifies the one-dimensional L2 element and the implementation of distributed body forces.

### Convergence

| Elements | L2 error | H1 error |
|---:|---:|---:|
| 4 | 2.1850e-02 | 2.0412e-01 |
| 8 | 5.4625e-03 | 1.0206e-01 |
| 16 | 1.3656e-03 | 5.1031e-02 |
| 32 | 3.4141e-04 | 2.5516e-02 |

The displacement error converges at approximately second order, while the gradient error converges at approximately first order.

<p align="center">
  <img src="figures/rod_self_weight_convergence.png" width="600">
</p>

---

## 2. Cantilever Beam

A two-dimensional cantilever beam subjected to gravity is used to compare Q4 and Q8 elements in a bending-dominated problem.

The parameters are

```text
Length          = 1.0
Height          = 0.1
Young's modulus = 200 MPa
Poisson ratio   = 0
Density         = 1000 kg/m^3
Gravity         = 10 m/s^2
```

The numerical tip displacement is compared with the Euler-Bernoulli beam solution

\[
u_y(L)
=
-\frac{qL^4}{8EI}.
\]

### Normalized tip displacement

| Nx | Q4, 1 layer | Q4, 4 layers | Q8, 1 layer |
|---:|---:|---:|---:|
| 2 | 0.0807 | 0.0807 | 0.9471 |
| 5 | 0.3400 | 0.3401 | 1.0029 |
| 10 | 0.6733 | 0.6737 | 1.0064 |
| 20 | 0.8956 | 0.8963 | 1.0066 |
| 40 | 0.9764 | 0.9772 | 1.0067 |

The Q8 element captures the bending response much more efficiently than the Q4 element.

<p align="center">
  <img src="figures/cantilever_convergence.png" width="600">
</p>

---

## 3. Thick Cylinder

The third problem considers a displacement-controlled thick elastic cylinder modeled using plane-stress Q4 elements.

The analytical radial displacement has the form

\[
u_r(r)
=
Ar+\frac{B}{r}.
\]

The boundary conditions are

\[
u_r(r_i)=0,
\qquad
u_r(r_o)=u_0.
\]

The material and geometry parameters are

```text
Inner radius       = 0.1
Outer radius       = 1.0
Outer displacement = 0.05
Young's modulus    = 100000
Poisson ratio      = 0.3
```

For the representative \(8\times64\) mesh,

```text
Relative nodal error   = 9.4049e-04

FEM strain energy      = 1.13826435e+03
Analytical energy      = 1.13943323e+03
```

### Mesh refinement

| Nr | Ntheta | Relative L2 error | Energy error |
|---:|---:|---:|---:|
| 2 | 16 | 1.6201e-02 | 2.2652e-02 |
| 4 | 32 | 6.2987e-03 | 4.9455e-03 |
| 8 | 64 | 2.1338e-03 | 1.0258e-03 |
| 16 | 128 | 6.2748e-04 | 2.1964e-04 |

The FEM displacement approaches the analytical solution under mesh refinement, and the total strain energy converges toward the analytical value.

<table>
<tr>
<td align="center">
<img src="figures/thick_cylinder_radial_displacement.png" width="430">
</td>
<td align="center">
<img src="figures/thick_cylinder_convergence.png" width="430">
</td>
</tr>
</table>

---

## 4. Cook's Membrane

Cook's membrane is a classical benchmark for distorted meshes and nearly incompressible elasticity.

The left boundary is fixed, while a total upward shear force

\[
V=100
\]

is distributed consistently over the right boundary.

The problem is modeled using plane strain with

```text
Young's modulus = 100
Poisson ratio   = 0.4999
```

### Mesh refinement

| Mesh | Vertical tip displacement |
|---:|---:|
| 2 x 2 | 5.08496 |
| 4 x 4 | 5.20732 |
| 8 x 8 | 5.35029 |
| 16 x 16 | 5.77859 |
| 32 x 32 | 7.08263 |

### Effect of Poisson's Ratio

For a \(32\times32\) Q4 mesh:

| Poisson ratio | Vertical tip displacement |
|---:|---:|
| 0.0000 | 24.1844 |
| 0.1000 | 24.1963 |
| 0.2500 | 23.2843 |
| 0.4000 | 21.1003 |
| 0.4500 | 19.9417 |
| 0.4900 | 17.9879 |
| 0.4999 | 7.08263 |

As the material approaches incompressibility, the standard displacement-based Q4 formulation becomes artificially stiff. The strong reduction in tip displacement near

\[
\nu \rightarrow 0.5
\]

demonstrates volumetric locking.

The implementation also evaluates the averaged stress trace

\[
\operatorname{tr}(\mathbf{T})
=
\sigma_{xx}
+
\sigma_{yy}
+
\sigma_{zz}
\]

along the line \(BC\).

<table>
<tr>
<td align="center">
<img src="figures/cooks_deformed.png" width="430">
</td>
<td align="center">
<img src="figures/cooks_poisson_study.png" width="430">
</td>
</tr>
</table>

---

## Numerical Tools

This project uses Python with NumPy for numerical operations, SciPy for sparse matrix assembly and linear solution, Matplotlib for visualization, and Pytest for verification.

---

## Notes

The project was developed as a finite element implementation and verification study for linear elasticity. The benchmark problems were selected to exercise different aspects of the solver, including body-force loading, bending behavior, analytical verification, mesh convergence, distributed boundary traction, stress recovery, and volumetric locking.