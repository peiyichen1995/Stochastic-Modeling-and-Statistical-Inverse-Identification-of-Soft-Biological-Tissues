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


# exponential covariance function
def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi*r*r/2.0/rho/rho)

def solve_covariance_EVP(cov, N, degree=1):
    def setup_FEM(N):
        mesh = UnitCubeMesh(24,N,N)
        V = FunctionSpace(mesh, 'CG', degree)
        u = TrialFunction(V)
        v = TestFunction(V)
        return mesh, V, u, v
    # construct FEM space
    mesh, V, u, v = setup_FEM(N)

    # dof to vertex map
    dof2vert = dof_to_vertex_map(V)
    # coords will be used for interpolation of covariance kernel
    coords = mesh.coordinates()
    # but we need degree of freedom ordering of coordinates
    coords = coords[dof2vert]
    # assemble mass matrix and convert to scipy
    M = assemble(u*v*dx)
    M = M.array()


    # evaluate covariance matrix
    L = coords.shape[0]
    C = np.zeros([L,L])

    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i]-coords[j]))
                C[i,j] = v
                C[j,i] = v

    # solve eigenvalue problem
    A = np.dot(M, np.dot(C, M))

    w, v = dla.eigh(A, b=M)


    return w, v, V, mesh, coords



def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval



w, v, V, mesh, coords = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.1, sigma=1.0), N = 5, degree = 1)

idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(len(w)):
    randomField = randomField + sqrt(w[i]) * v[:,i] * gauss[i]

for i in range(len(w)):
    randomField[i] = norm.cdf(randomField[i])
    randomField[i] = gamma.ppf(randomField[i],1.9)

rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))

file = File("3D_Random.pvd")
file << rF


#exit()

print(randomField.shape)


## start for model


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
#mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, 'CG',1)

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
c = Expression(('0.1', '0', '0'), element = V.ufl_element())
r = Expression(('0.0',
                'scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])',
                'scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])'),
                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3, element = V.ufl_element())

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Kinematics


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
A_1 = as_vector([sqrt(0.5),sqrt(0.5)])
M_1 = outer(A_1, A_1)
J4_1 = tr(C*M_1)

# Body forces
T  = Constant((0.1, 0.0, 0.0))  # Traction force on the boundary
B  = Expression(('0.0', '-0.5', '0.0'), element = V.ufl_element())  # Body force per unit volume

# Invariants of deformation tensors
I1 = tr(C)
I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I3 = det(C)

#eta1 = 141
eta1 = 141*rF
eta2 = 160
eta3 = 3100
delta = 2*eta1 + 4*eta2 + 2*eta3

e1 = 0.005
e2 = 10

k1 = 0.1
k2 = 0.04


# compressible Mooney-Rivlin model
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))
# penalty
psi_P = e1*(pow(I3,e2)+pow(I3,-e2)-2)
# tissue
psi_ti_1 = k1/2/k2*(exp(pow(conditional(gt(J4_1,1),conditional(gt(J4_1,2),J4_1-1,2*pow(J4_1-1,2)-pow(J4_1-1,3)),0),2)*k2)-1)
psi_ti_2 = k1*(exp(k2*conditional(gt(J4_1,1),pow((J4_1-1),2),0))-1)/k2/2

psi = psi_MR
# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem

problem = NonlinearVariationalProblem(F, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()


file = File("displacement_rf_3D.pvd")
file << u
