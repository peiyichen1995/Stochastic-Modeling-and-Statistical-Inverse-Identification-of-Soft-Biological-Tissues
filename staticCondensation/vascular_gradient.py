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

#read function from file
V_g = FunctionSpace(mesh, 'CG', 2)


# Read the contents of the file back into a new function, `f2`:
u_read = Function(V_g)
fFile = HDF5File(MPI.comm_world,"u1.h5","r")
fFile.read(u_read,"/f")
fFile.close()

u_grad = project(grad(u_read), V)

u_array = u_grad.vector().get_local()

max_u = u_array.max()
u_array /= max_u
u_grad.vector()[:] = u_array
e3 = u_grad

u_read = Function(V_g)
fFile = HDF5File(MPI.comm_world,"u2.h5","r")
fFile.read(u_read,"/f")
fFile.close()

u_grad = project(grad(u_read), V)

u_array = u_grad.vector().get_local()

max_u = u_array.max()
u_array /= max_u
u_grad.vector()[:] = u_array
e1 = u_grad

e2 = cross(e3, e1)
