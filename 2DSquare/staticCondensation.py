from dolfin import *
import math
import ufl
import numpy

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = Mesh()
with XDMFFile("mesh/mesh.xdmf") as infile:
    infile.read(mesh)
File("mesh/square.pvd").write(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("mesh/mf.xdmf") as infile:
    infile.read(mvc, "face_id")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
File("mesh/square.pvd").write(mf)

ds = Measure("ds", domain=mesh, subdomain_data=mf)
n = FacetNormal(mesh)

# Define function spaces
V = VectorFunctionSpace(mesh, 'CG', 2)

# Define Dirichlet boundary (point)

c = Expression(('0.5', '0.5', '0.5'), element = V.ufl_element())

point =  CompiledSubDomain("near(x[0], side) && near(x[1], side) && near(x[2], side)", side = 0.0)

bcp = DirichletBC(V, c, point, method="pointwise")

p = Constant(100)
bcs = [bcp]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Body force
B = Constant((0.0, 0.0, 0.0))

# Invariants of deformation tensors
I1 = tr(C)
I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I3 = det(C)

# Elasticity parameters
eta1 = 141
eta2 = 160
eta3 = 3100
delta = 2*eta1 + 4*eta2 + 2*eta3

# Stored strain energy density (compressible neo-Hookean model)# compressible Mooney-Rivlin model
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))
psi = psi_MR

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(-p*n, u)*ds(2) - dot(-p*n, u)*ds(4)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

problem = NonlinearVariationalProblem(F, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

# Save solution in VTK format
file = File("SC2DPressure.pvd")
file << u
