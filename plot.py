from  dolfin  import *
from  matplotlib  import  pyplot
# parameters["plotting_backend"] = "matplotlib"
mesh2D = UnitSquareMesh(16,16)
mesh3D = UnitCubeMesh(16, 16, 16)
plot(mesh2D)
plot(mesh3D)
pyplot.show()
