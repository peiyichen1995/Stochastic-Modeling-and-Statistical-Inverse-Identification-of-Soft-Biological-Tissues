import sympy
from dolfin import *
# We have to create a "symbol" called x
x = sympy.Symbol('x')
y = sympy.Symbol('Y')
z = sympy.Symbol('z')

ux = 0.1*x*x
uy = 0.1*y*y
uz = 0.1*z*z

du = Identity(3)

print(du[1][1])
