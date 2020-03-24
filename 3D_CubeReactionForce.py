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

# Create mesh and define function space
#mesh = UnitSquareMesh(10, 10)
N = 5
mesh = UnitCubeMesh(24,N,N)
V = VectorFunctionSpace(mesh, 'CG',1)

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
A_1 = as_vector([1,0,0])
M_1 = outer(A_1, A_1)
J4_1 = tr(C*M_1)
A_2 = as_vector([1,0,0])
M_2 = outer(A_2, A_2)
J4_2 = tr(C*M_2)

# Body forces
T  = Constant((0.0, 0.0, 0.0))  # Traction force on the boundary
B  = Expression(('0.0', '0.0', '0.0'), element = V.ufl_element())  # Body force per unit volume

# Invariants of deformation tensors
I1 = tr(C)
I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I3 = det(C)

eta1 = 141
eta2 = 160
eta3 = 3100
delta = 2*eta1 + 4*eta2 + 2*eta3

#e1 = 0.005
e1 = 10
e2 = 10

k1 = 10
k2 = 0.04

# compressible Mooney-Rivlin model
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))
# penalty
psi_P = e1*(pow(I3,e2)+pow(I3,-e2)-2)
# tissue
psi_ti_1 = k1/2/k2*(exp(pow(conditional(gt(J4_1,1),conditional(gt(J4_1,2),J4_1-1,2*pow(J4_1-1,2)-pow(J4_1-1,3)),0),2)*k2)-1)
psi_ti_2 = k1*(exp(k2*conditional(gt(J4_2,1),pow((J4_2-1),2),0))-1)/k2/2

psi = psi_MR + psi_P + psi_ti_1 + psi_ti_2
# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)


# Mark boundary subdomians
#bottom =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
#top = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
#back =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
#front = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)
#left =  CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0)
#right = CompiledSubDomain("near(x[2], side) && on_boundary", side = 1.0)
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)



# Define Dirichlet boundary (x = 0 or x = 1)
#c = Expression(('0.1', '0', '0'), element = V.ufl_element())
#c2 = Expression(('0.0', '0', '0'), element = V.ufl_element())

#bc_t = DirichletBC(V, c, top)
#bc_b = DirichletBC(V, c2, bottom)
#bc_f = DirichletBC(V, c2, front)
#bc_ba = DirichletBC(V, c2, back)
#bc_l = DirichletBC(V, c2, left)
#bc_r = DirichletBC(V, c2, right)
#bcs = [bc_l, bc_r, bc_f, bc_ba, bc_t, bc_b]

ReactionT = []
DisplacementT = []
volume = []
for i in range(10):
    DisplacementT.append(0.05*i)
    c = Expression(('A', '0', '0'), A = -0.05*i, element = V.ufl_element())
    r = Expression(('0', '0', '0'), element = V.ufl_element())
    bcl = DirichletBC(V.sub(0), Constant(-0.05*i), left)
    bcr = DirichletBC(V, r, right)
    bcs = [bcl, bcr]
    # Solve variational problem
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['maximum_iterations'] = 100
    prm['newton_solver']['relaxation_parameter'] = 1.0
    solver.solve()
    # Integrate the forces
    ndim = mesh.geometry().dim()
    zero_v=Constant((0.,)*ndim)
    f=zero_v
    coords = V.tabulate_dof_coordinates() #mesh.coordinates()
    x_dofs = V.sub(0).dofmap().dofs()
    nodes = []
    for index in range(0, len(coords), 3):
        nodes.append([coords[index,0], coords[index,1], coords[index,2]])
    RF = derivative(Pi, u, v)
    f_int = assemble(RF)
    bcl.apply(f_int)
    Fx = []
    totalFx = 0
    for i in x_dofs:
        Fx.append(f_int[i])
        totalFx = totalFx + f_int[i]
    ReactionT.append(totalFx)
    volume_after0 = assemble(det(Identity(3) + grad(u))*dx)
    volume.append(volume_after0)

print("Voulme")
print(volume)
#
# ReactionC = []
# DisplacementC = []
# for i in range(10):
#     DisplacementC.append(0.05*i)
#     c = Expression(('A', '0', '0'), A = -0.05*i, element = V.ufl_element())
#     r = Expression(('0', '0', '0'), element = V.ufl_element())
#     bcl = DirichletBC(V.sub(0), Constant(0.05*i), left)
#     bcr = DirichletBC(V, r, right)
#     bcs = [bcl, bcr]
#     # Solve variational problem
#     problem = NonlinearVariationalProblem(F, u, bcs, J)
#     solver  = NonlinearVariationalSolver(problem)
#     prm = solver.parameters
#     prm['newton_solver']['absolute_tolerance'] = 1E-8
#     prm['newton_solver']['relative_tolerance'] = 1E-7
#     prm['newton_solver']['maximum_iterations'] = 100
#     prm['newton_solver']['relaxation_parameter'] = 1.0
#     solver.solve()
#     # Integrate the forces
#     ndim = mesh.geometry().dim()
#     zero_v=Constant((0.,)*ndim)
#     f=zero_v
#     coords = V.tabulate_dof_coordinates() #mesh.coordinates()
#     x_dofs = V.sub(0).dofmap().dofs()
#     nodes = []
#     for index in range(0, len(coords), 3):
#         nodes.append([coords[index,0], coords[index,1], coords[index,2]])
#     RF = derivative(Pi, u, v)
#     f_int = assemble(RF)
#     bcl.apply(f_int)
#     Fx = []
#     totalFx = 0
#     for i in x_dofs:
#         Fx.append(f_int[i])
#         totalFx = totalFx + f_int[i]
#     ReactionC.append(totalFx)


print("Tension")
print(ReactionT)
print(DisplacementT)

#
# print("Compression")
# print(ReactionC)
# print(DisplacementC)

#plt.plot(Displacement, Reaction)
#plt.ylabel('Force')
#plt.xlabel('Displacement')
#plt.show()


file = File("cubeStretch.pvd")
file << u

# Integrate the forces

#ndim = mesh.geometry().dim()

#zero_v=Constant((0.,)*ndim)
#f=zero_v

#print("Reaction force")

#coords = V.tabulate_dof_coordinates() #mesh.coordinates()
#x_dofs = V.sub(0).dofmap().dofs()
#y_dofs = V.sub(1).dofmap().dofs()
#z_dofs = V.sub(2).dofmap().dofs()

#nodes = []
#for index in range(0, len(coords), 3):
#    nodes.append([coords[index,0], coords[index,1], coords[index,2]])

#RF = derivative(Pi, u, v)
#f_int = assemble(RF)
#bcl.apply(f_int)

#Fx = []
#totalFx = 0
#for i in x_dofs:
#    Fx.append(f_int[i])
#    totalFx = totalFx + f_int[i]
#
#print("Fx")
#print(totalFx)

#Fy = []
#for i in y_dofs:
#    Fy.append(f_int[i])

#Fz = []
#for i in z_dofs:
#    Fz.append(f_int[i])
