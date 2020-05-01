from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import scipy.integrate as integrate
import math
import ufl

# exponential covariance function
def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi*r*r/2.0/rho/rho)

def cov_len(rho, sigma=1.0):
    return integrate.quad(lambda r: cov_exp(r, rho), 0, math.inf)

def solve_covariance_EVP(cov, N, degree=1):
    def setup_FEM(N):
        mesh = UnitSquareMesh(N,N)
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

    print("size of M: ")
    print(len(M))

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


    # w, v = spla.eigsh(A, k, M)
    w, v = dla.eigh(A, b=M)

    return w, v, mesh, C, M, V



def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval




w, v, mesh, C, M, V = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.2, sigma=1.0), N = 50, degree = 1)

idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]


print("Truncation error")
e = 0
eig = 0
trCM = np.trace(np.dot(C, M))
while 1 - eig / trCM > 0.1:
    eig = eig + w[e]
    e = e + 1
print(e)
print(1-eig/trCM)


randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(e):
    print(w[i])
    randomField = randomField + sqrt(w[i]) * v[:,i] * gauss[i]

for i in range(len(w)):
    randomField[i] = norm.cdf(randomField[i])
    randomField[i] = gamma.ppf(randomField[i],20,loc=0,scale=0.05)


rF = set_fem_fun(randomField, V)

file = File("2D_Random.pvd")
file << rF

plt.figure()
im = plot(rF)
plt.colorbar(im)
plt.title("randomField")
plt.savefig('randomField.png')
plt.show()


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
VVV = TensorFunctionSpace(mesh, 'DG', 1)

# Mark boundary subdomians
bottom =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
top = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
left =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(('0.1*x[0]*x[0]', '0'), element = V.ufl_element())

bc_t = DirichletBC(V, c, top)
bc_b = DirichletBC(V, c, bottom)
bc_l = DirichletBC(V, c, left)
bc_r = DirichletBC(V, c, right)
bcs = [bc_l, bc_r, bc_t, bc_b]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)
A_1 = as_vector([sqrt(0.5),sqrt(0.5)])
M_1 = outer(A_1, A_1)
J4_1 = tr(C*M_1)

# Invariants of deformation tensors
I1 = tr(C)
I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I3 = det(C)

#eta1 = 141
eta1 = 141*rF

eta2 = 160*rF
eta3 = 3100*rF
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

psi = psi_MR + psi_ti_1
# Total potential energy
Pi = psi*dx

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

file = File("displacement_rf.pvd")
file << u
#file << rF


PK2 = 2.0*diff(psi,C)
PK2Project = project(PK2,VVV)

file = XDMFFile("PK2Tensor.xdmf")
file.write(PK2Project,0)
