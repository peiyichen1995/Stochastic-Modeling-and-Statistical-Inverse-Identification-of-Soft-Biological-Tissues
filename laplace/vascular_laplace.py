from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import math
import ufl

# build the solver
from petsc4py import PETSc


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


class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)
        # null_space = build_nullspace(V)
        # null_space.orthogonalize(b)

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
        # null_space = build_nullspace(V)
        # as_backend_type(A).set_nullspace(null_space)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("ksp_max_it", 1000)
        PETScOptions.set("ksp_gmres_restart", 200)
        PETScOptions.set("pc_type", "asm")
        PETScOptions.set("sub_pc_type", "lu")
        #PETScOptions.set("sub_pc_factor_levels", 10)

        self.linear_solver().set_from_options()


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and mesh faces
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
File("mesh/vascular.pvd").write(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("mf.xdmf") as infile:
    infile.read(mvc, "face_id")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
File("mesh/vascular_facets.pvd").write(mf)

ds = Measure("ds", domain=mesh, subdomain_data=mf)
n = FacetNormal(mesh)
# Define function spaces
V = FunctionSpace(mesh, 'CG', 2)

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u1  = Function(V)
u2  = Function(V)
# Mark boundary subdomians
# bc1 = DirichletBC(V.sub(0), Constant(0), mf, 7)
# bc2 = DirichletBC(V.sub(0), Constant(0), mf, 8)

bc1 = DirichletBC(V, Constant(0.0), mf, 7)
bc2 = DirichletBC(V, Constant(1.0), mf, 8)

# DirichletBC
bcs = [bc1, bc2]

# Total potential energy
Pi = 0.5 * dot(grad(u1), grad(u1)) * dx


# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi = derivative(Pi, u1, v)

# Compute Jacobian of F
J = derivative(dPi, u1, du)


# Solve variational problem
problem = Problem(J, dPi, bcs)
solver = CustomSolver()
solver.solve(problem, u1.vector())

# Write `f` to a file:
fFile = HDF5File(MPI.comm_world,"u1.h5","w")
fFile.write(u1,"/f")
fFile.close()

# Read the contents of the file back into a new function, `f2`:
u_read = Function(V)
fFile = HDF5File(MPI.comm_world,"u1.h5","r")
fFile.read(u_read,"/f")
fFile.close()
print(assemble(((u1-u_read)**2)*dx))


#####################
bc3 = DirichletBC(V, Constant(0.0), mf, 1)
bc4 = DirichletBC(V, Constant(1.0), mf, 5)

bcs = [bc3, bc4]

# Total potential energy
Pi = 0.5 * dot(grad(u2), grad(u2)) * dx


# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi = derivative(Pi, u2, v)

# Compute Jacobian of F
J = derivative(dPi, u2, du)


# Solve variational problem
problem = Problem(J, dPi, bcs)
solver = CustomSolver()
solver.solve(problem, u2.vector())

# Write `f` to a file:
fFile = HDF5File(MPI.comm_world,"u2.h5","w")
fFile.write(u2,"/f")
fFile.close()

# Read the contents of the file back into a new function, `f2`:
u_read = Function(V)
fFile = HDF5File(MPI.comm_world,"u2.h5","r")
fFile.read(u_read,"/f")
fFile.close()
print(assemble(((u2-u_read)**2)*dx))
