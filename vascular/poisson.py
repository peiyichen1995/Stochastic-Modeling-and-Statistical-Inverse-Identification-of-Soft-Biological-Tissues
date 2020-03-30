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
V = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Mark boundary subdomians
#bc1 = DirichletBC(V, Constant((0.1, 0.0, 0.0)), mf, 7)
#bc2 = DirichletBC(V, Constant((0.0, 0.0, 0.0)), mf, 8)
bcs = []

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)                   # Right Cauchy-Green tensor
A_1 = as_vector([sqrt(0.5),sqrt(0.5),0])
M_1 = outer(A_1, A_1)
J4_1 = tr(C*M_1)
A_2 = as_vector([-sqrt(0.5),sqrt(0.5),0])
M_2 = outer(A_2, A_2)
J4_2 = tr(C*M_2)

# Body forces
P  = Constant(100.0)  # Traction force on the boundary
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

PK2 = 2.0*diff(psi,C)
PK2Project = project(PK2, VVV)

file = File("vascularPressure.pvd")
file << u

file = XDMFFile("PK2TensorVascularPressure.xdmf")
file.write(PK2Project,0)
