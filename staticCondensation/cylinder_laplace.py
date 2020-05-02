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
info(mesh)
n = FacetNormal(mesh)
V = FunctionSpace(mesh, 'CG', 2)
V_cyl = VectorFunctionSpace(mesh, 'CG', 2)

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

# file = File("laplace.pvd")
# file << u

# Write `f` to a file:
fFile = HDF5File(MPI.comm_world,"f2.h5","w")
fFile.write(u, "f")
fFile.close()

mesh_file = File("mesh.xml")
mesh_file << mesh

exit()

u_grad = grad(u)

# normalize the gradient field
u_grad = sqrt(inner(u_grad, u_grad))

print(type(u_grad))

e2 = u_grad
a = sqrt(0.5) * e2
A = outer(a,a)

#############################################
print("solving...")


VVV = TensorFunctionSpace(mesh, 'DG', 1)
# Define functions
du = TrialFunction(V_cyl)            # Incremental displacement
v = TestFunction(V_cyl)             # Test function
u = Function(V_cyl)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
# C = variable(F.T*F)                   # Right Cauchy-Green tensor
C = F.T*F
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

psi = psi_MR  # + psi_P + psi_ti_1 + psi_ti_2

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
