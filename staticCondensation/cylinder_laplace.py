from __future__ import division
from dolfin import *
from mshr import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import math
import ufl

from petsc4py import PETSc

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
#mesh = UnitSquareMesh(10, 10)
# cylinder mesh
cylinder_o = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 2, 2)
cylinder_i = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 1, 1)

geometry = cylinder_o - cylinder_i

mesh = generate_mesh(geometry, 10)
n = FacetNormal(mesh)
V = FunctionSpace(mesh, 'CG', 2)


# Mark boundary subdomians
left = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[2], side) && on_boundary", side=3.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(('0', '0', '0'), element=V.ufl_element())
r = Expression(('0', '0', '0'), element=V.ufl_element())

bc1 = DirichletBC(V, Constant(0.0), left)
bc2 = DirichletBC(V, Constant(1.0),right)

bcs = [bc1, bc2]

# Define functions
du = TrialFunction(V)           # Incremental displacement
v = TestFunction(V)             # Test function
u = Function(V)                 # Displacement from previous iteration

Pi = 0.5 * dot(grad(u), grad(u)) * dx

# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(dPi, u, du)


# Solve variational problem
problem = NonlinearVariationalProblem(dPi, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-08
prm['newton_solver']['relative_tolerance'] = 1E-12
prm['newton_solver']['maximum_iterations'] = 20
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['lu_solver']['symmetric'] = True
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 200

solver.solve()


file = File("laplace.pvd")
file << u
