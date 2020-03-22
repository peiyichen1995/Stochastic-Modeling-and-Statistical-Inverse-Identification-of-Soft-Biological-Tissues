from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import ufl


# exponential covariance function
def cov_exp(r, rho, sigma2=1.0):
    return sigma2 * np.exp(-r*r/2.0/rho/rho)

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



w, v, V, mesh, coords = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.1, sigma2=1.0), N = 21, degree = 1)

idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(len(w)):
    randomField = randomField + sqrt(w[i]) * v[:,i] * gauss[i]

for i in range(len(w)):
    randomField[i] = norm.cdf(randomField[i])
    randomField[i] = gamma.ppf(randomField[i],1.9,scale=1.5)

rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))


plt.figure()
im = plot(rF)
plt.colorbar(im)
plt.title("Non-aussian Field")
plt.show()
exit()

print(type(np.array([1,2])))
print(type(coords[1]))


print(randomField.shape)

plt.figure()
im = plot(rF)
plt.colorbar(im)
plt.title("randomField")
# plt.savefig('randomField.png')
# plt.show()

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
C = F.T*F                   # Right Cauchy-Green tensor
A_1 = as_vector([sqrt(0.5),sqrt(0.5)])
M_1 = outer(A_1, A_1)
J4_1 = tr(C*M_1)

# Body forces
B_I  = Expression(('-(2*(3401*x[0]*x[0] + 34010*x[0] + 174050))/(5*(x[0] + 5)*(x[0] + 5))', '0'), element = V.ufl_element())  # Body force per unit volume
B_P = Expression(('95367431640625/(2*pow(x[0] + 5,22)) - pow(x[0] + 5,18)/190734863281250 - (x[0]/5 + 1)*((9*pow(x[0] + 5,17))/19073486328125 + 5245208740234375/pow(x[0] + 5,23))', '0'), element = V.ufl_element())  # Body force per unit volume
B_T1  = Expression(('-(exp(((x[0]/5 + 1)*(x[0]/5 + 1)/2 - 1/2)*((x[0]/5 + 1)*(x[0]/5 + 1)/2 - 1/2)/25)*(pow(x[0],9) + 45*pow(x[0],8) + 825*pow(x[0],7) + 7875*pow(x[0],6) + 181875*pow(x[0],5) + 3628125*pow(x[0],4) + 30593750*pow(x[0],3) + 105468750*pow(x[0],2) + 1582031250*x[0] + 7324218750))/305175781250',' -(exp(((x[0]/5 + 1)*(x[0]/5 + 1)/2 - 1/2)*((x[0]/5 + 1)*(x[0]/5 + 1)/2 - 1/2)/25)*(pow(x[0],8) + 40*pow(x[0],7) + 625*pow(x[0],6) + 4750*pow(x[0],5) + 126875*pow(x[0],4) + 2212500*pow(x[0],3) + 13281250*pow(x[0],2) + 23437500*x[0] + 488281250))/61035156250'), element = V.ufl_element())

B_T1_12 = Expression(('-(exp((pow(x[0],2)*(x[0] + 10)*(x[0] + 10))/62500)*(3*pow(x[0],17) + 255*pow(x[0],16) + 9625*pow(x[0],15) + 211875*pow(x[0],14) + 4184375*pow(x[0],13) + 105340625*pow(x[0],12) + 2319312500*pow(x[0],11) + 33763125000*pow(x[0],10) + 391903906250*pow(x[0],9) + 5484972656250*pow(x[0],8) + 72800781250000*pow(x[0],7) + 608167187500000*pow(x[0],6) + 2805677734375000*pow(x[0],5) + 7877373046875000*pow(x[0],4) + 19528125000000000*pow(x[0],3) - 3002929687500000*pow(x[0],2) - 277587890625000000*x[0] - 457763671875000000))/3814697265625000000','-(exp((pow(x[0],2)*(x[0] + 10)*(x[0] + 10))/62500)*(3*pow(x[0],16) + 240*pow(x[0],15) + 8425*pow(x[0],14) + 169750*pow(x[0],13) + 3241875*pow(x[0],12) + 83037500*pow(x[0],11) + 1736937500*pow(x[0],10) + 22586250000*pow(x[0],9) + 242449218750*pow(x[0],8) + 3492843750000*pow(x[0],7) + 43317031250000*pow(x[0],6) + 294453125000000*pow(x[0],5) + 950892578125000*pow(x[0],4) + 2051132812500000*pow(x[0],3) + 5454101562500000*pow(x[0],2) - 15234375000000000*x[0] - 67138671875000000))/762939453125000000'), element = V.ufl_element())  # Body force per unit volume

#B_T1_12 = Expression(('0', '0'), element = V.ufl_element())
B_0 = Expression(('0', '0'), element = V.ufl_element())

B = B_I + conditional(gt(J4_1, 1),conditional(gt(J4_1,2),B_T1,B_T1_12),B_0)
T  = Constant((0.0, 0.0))  # Traction force on the boundary

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

psi = psi_MR + psi_ti_1
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


file = File("displacement_rf.pvd")
file << u
#file << rF

u_e = interpolate(c, V)
error = (u-u_e)**2*dx
L2_err= sqrt(abs(assemble(error)))
print(L2_err)
