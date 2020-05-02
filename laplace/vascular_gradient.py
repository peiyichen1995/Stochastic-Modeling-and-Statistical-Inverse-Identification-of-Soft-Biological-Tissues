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
with XDMFFile("../mesh.xdmf") as infile:
    infile.read(mesh)
File("../mesh/vascular.pvd").write(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("../mf.xdmf") as infile:
    infile.read(mvc, "face_id")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
File("../mesh/vascular_facets.pvd").write(mf)

ds = Measure("ds", domain=mesh, subdomain_data=mf)
n = FacetNormal(mesh)
# Define function spaces
V = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)

#read function from file
V_g = FunctionSpace(mesh, 'CG', 2)


# Read the contents of the file back into a new function, `f2`:
u1_read = Function(V_g)
fFile = HDF5File(MPI.comm_world,"u1.h5","r")
fFile.read(u1_read,"/f")
fFile.close()
e3 = grad(u1_read)

# Read the contents of the file back into a new function, `f2`:
u2_read = Function(V_g)
fFile = HDF5File(MPI.comm_world,"u2.h5","r")
fFile.read(u2_read,"/f")
fFile.close()
e1 = grad(u2_read)

# e2 from e1 and e3
e2 = my_cross(e3, e1)

# e1Project = project(e1, V, solver_type="mumps")
# file = XDMFFile("e1.xdmf")
# file.write(e1Project,0)
#
# e2Project = project(e2, V, solver_type="mumps")
# file = XDMFFile("e2.xdmf")
# file.write(e2Project,0)
#
# e3Project = project(e3, V, solver_type="mumps")
# file = XDMFFile("e3.xdmf")
# file.write(e3Project,0)


# normalize the gradient field
e1 = sqrt(inner(e1, e1))
e2 = sqrt(inner(e2, e2))
e3 = sqrt(inner(e3, e3))

a1 = sqrt(3)/2*e1 + 1/2*e2
a2 = sqrt(3)/2*e1 - 1/2*e2

##################################
# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Mark boundary subdomians
bc1 = DirichletBC(V.sub(0), Constant(0), mf, 7)
bc2 = DirichletBC(V.sub(0), Constant(0), mf, 8)

#bc2 = DirichletBC(V, Constant((0.0, 0.0, 0.0)), mf, 8)
# no DirichletBC
bcs = []

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)                   # Right Cauchy-Green tensor
M_1 = outer(a1, a1)
J4_1 = tr(C*M_1)
M_2 = outer(a2, a2)
J4_2 = tr(C*M_2)

# Body forces
P  = Constant(10)  # Traction force on the boundary
B  = Expression(('0.0', '0.0', '0.0'), element = V.ufl_element())  # Body force per unit volume

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
psi_P = e1*(pow(I3,e2)+pow(I3,-e2)-2)
# tissue
# psi_ti_1 = k1/2/k2*(exp(pow(conditional(gt(J4_1,1),conditional(gt(J4_1,2),J4_1-1,2*pow(J4_1-1,2)-pow(J4_1-1,3)),0),2)*k2)-1)
psi_ti_1 = k1*(exp(k2*conditional(gt(J4_1,1),pow((J4_1-1),2),0))-1)/k2/2
psi_ti_2 = k1*(exp(k2*conditional(gt(J4_2,1),pow((J4_2-1),2),0))-1)/k2/2

psi = psi_MR + psi_P + psi_ti_1 + psi_ti_2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(-P*n, u)*ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(dPi, u, du)

# Solve variational problem
problem = Problem(J, dPi, bcs)
solver = CustomSolver()
solver.solve(problem, u.vector())

PK2 = 2.0*diff(psi,C)
PK2Project = project(PK2, VVV)

file = File("vascularIterativeSolver.pvd")
file << u

file = XDMFFile("PK2TensorVascularIterativeSolver.xdmf")
file.write(PK2Project,0)
