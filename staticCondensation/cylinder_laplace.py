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


def my_cross(a,b):
       return as_vector(( a[1]*b[2]-a[2]*b[1],  a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0] ))

def build_nullspace(V):
    x = Function(V).vector()
    nullspace_basis = [x.copy() for i in range(6)]
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0)
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2)
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0)
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2)
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1)
    for x in nullspace_basis:
        x.apply("insert")
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()
    return basis

class ProblemWithNullSpace(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)
        null_space = build_nullspace(V)
        null_space.orthogonalize(b)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

class SolverWithNullSpace(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver("gmres"), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        null_space = build_nullspace(V)
        as_backend_type(A).set_nullspace(null_space)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("ksp_max_it", 1000)
        PETScOptions.set("ksp_gmres_restart", 200)
        PETScOptions.set("pc_type", "asm")
        PETScOptions.set("sub_pc_type", "ilu")
        PETScOptions.set("sub_pc_factor_levels", 10)

        self.linear_solver().set_from_options()

class CustomProblem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver("gmres"), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("ksp_max_it", 1000)
        PETScOptions.set("ksp_gmres_restart", 200)
        PETScOptions.set("pc_type", "asm")
        PETScOptions.set("sub_pc_type", "lu")
        self.linear_solver().set_from_options()

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# mesh
cylinder_o = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 2, 2)
cylinder_i = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 1, 1)
geometry = cylinder_o - cylinder_i
mesh = generate_mesh(geometry, 10)
n = FacetNormal(mesh)
info(mesh)

# function space
V = FunctionSpace(mesh, 'CG', 2)
VV = VectorFunctionSpace(mesh, 'CG', 2)
dv = TrialFunction(V)
w = TestFunction(V)
phi = Function(V)

# mark boundary subdomians
bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
top = CompiledSubDomain("near(x[2], side) && on_boundary", side=3.0)
inner = CompiledSubDomain("near(sqrt(x[0]*x[0]+x[1]*x[1]), side) && on_boundary", side=1.0)
outer = CompiledSubDomain("near(sqrt(x[0]*x[0]+x[1]*x[1]), side) && on_boundary", side=2.0)

# write mesh
mesh_file = File("mesh.xml")
mesh_file << mesh

# boundary conditions
bc_top = DirichletBC(V, Constant(1.0), top)
bc_bottom = DirichletBC(V, Constant(0.0), bottom)
bc_inner = DirichletBC(V, Constant(0.0), inner)
bc_outer = DirichletBC(V, Constant(1.0), outer)
bcs_1 = [bc_top, bc_bottom]
bcs_2 = [bc_inner, bc_outer]

# variational problem
Pi = 0.5 * dot(grad(phi), grad(phi)) * dx
dPi = derivative(Pi, phi, w)
J = derivative(dPi, phi, dv)

# define variational problem for phi_1 and phi_2
problem_1 = CustomProblem(J, dPi, bcs_1)
problem_2 = CustomProblem(J, dPi, bcs_2)
solver = CustomSolver()

# solve and write phi_1
solver.solve(problem_1, phi.vector())
fFile = HDF5File(MPI.comm_world,"phi1.h5","w")
fFile.write(phi, "phi1")
fFile.close()

# solve and write phi_2
solver.solve(problem_2, phi.vector())
fFile = HDF5File(MPI.comm_world,"phi2.h5","w")
fFile.write(phi, "phi2")
fFile.close()

exit()
