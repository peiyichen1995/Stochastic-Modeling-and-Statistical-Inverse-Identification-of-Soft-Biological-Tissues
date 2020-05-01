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
        null_space = build_nullspace(V)
        null_space.orthogonalize(b)

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


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# Create mesh and define function space
#mesh = UnitSquareMesh(10, 10)
# cylinder mesh
cylinder_o = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 2, 2)
cylinder_i = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 1, 1)

geometry = cylinder_o - cylinder_i

mesh = generate_mesh(geometry, 10)
n = FacetNormal(mesh)
V = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)
V_read = FunctionSpace(mesh, 'CG', 2)

# Read the contents of the file back into a new function, `f2`:
u_read = Function(V_read)
fFile = HDF5File(MPI.comm_world,"f.h5","r")
fFile.read(u_read,"/f")
fFile.close()

u_grad = project(grad(u_read), V)

u_array = u_grad.vector().get_local()

max_u = u_array.max()
u_array /= max_u
u_grad.vector()[:] = u_array
# u_grad.vector().set_local(u_array)  # alternative
e3 = u_grad

e2 = cross(e3, e3)

a = sqrt(0.5) * e2
A = outer(a,a)


# Mark boundary subdomians
left = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[2], side) && on_boundary", side=3.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(('0', '0', '0'), element=V.ufl_element())
r = Expression(('0', '0', '0'), element=V.ufl_element())

P = Constant(0.1)
c_fixed = Expression(('0.0', '0.0', '0.0'), element=V.ufl_element())
point = CompiledSubDomain(
    "near(x[0], side1) && near(x[1], side2) && near(x[2], side3)", side1=1.5, side2=1.5, side3=5)
# bcp = DirichletBC(V, c_fixed, point, method="pointwise")
# bcl = DirichletBC(V, c, left)
# bc2 = DirichletBC(V.sub(2), Constant(0),right)
# bcs = [bcl, bc2]
bcs = []

# Define functions
du = TrialFunction(V)            # Incremental displacement
v = TestFunction(V)             # Test function
u = Function(V)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)                   # Right Cauchy-Green tensor
A_1 = as_vector([sqrt(0.5), sqrt(0.5), 0])
M_1 = outer(A_1, A_1)
J4_1 = tr(C*A)
A_2 = as_vector([-sqrt(0.5), sqrt(0.5), 0])
M_2 = outer(A_2, A_2)
J4_2 = tr(C*A)

# Body forces
T = Constant((0.0, 0.0, 0.0))  # Traction force on the boundary
# Body force per unit volume
B = Expression(('0.0', '0.0', '0.0'), element=V.ufl_element())

# Invariants of deformation tensors
I1 = tr(C)
I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I3 = det(C)
eta1 = 141
eta2 = 160
eta3 = 3100
delta = 2*eta1 + 4*eta2 + 2*eta3

e1 = 0.005
e2 = 10

k1 = 100000
k2 = 0.04

# compressible Mooney-Rivlin model
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))
# penalty
psi_P = e1*(pow(I3, e2)+pow(I3, -e2)-2)
# tissue
# psi_ti_1 = k1/2/k2*(exp(pow(conditional(gt(J4_1,1),conditional(gt(J4_1,2),J4_1-1,2*pow(J4_1-1,2)-pow(J4_1-1,3)),0),2)*k2)-1)
psi_ti_1 = k1*(exp(k2*conditional(gt(J4_1, 1), pow((J4_1-1), 2), 0))-1)/k2/2
psi_ti_2 = k1*(exp(k2*conditional(gt(J4_2, 1), pow((J4_2-1), 2), 0))-1)/k2/2

psi = psi_MR + psi_P + psi_ti_1 + psi_ti_2

# FacetFunction("size_t", mesh)
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundaries.set_all(0)


class Inner(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (-1.1, 1.1)) and between(x[0], (-1.1, 1.1)) and between(x[2], (-0.1, 6)))


Inner = Inner()
Inner.mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds - dot(-P*n, u)*ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(dPi, u, du)

# Solve variational problem
problem = Problem(J, dPi, bcs)
solver = CustomSolver()
solver.solve(problem, u.vector())

PK2 = 2.0*diff(psi, C)
PK2Project = project(PK2, VVV)

file = File("cylinder.pvd")
file << u

file = XDMFFile("cylinder.xdmf")
file.write(PK2Project, 0)
