from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import time
import scipy.integrate as integrate
import math


# exponential covariance function
def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi*r*r/2.0/rho/rho)

def cov_len(rho, sigma=1.0):
    return integrate.quad(lambda r: cov_exp(r, rho), 0, math.inf)

def solve_covariance_EVP(cov, N, degree=1):
    """

    """
    def setup_FEM(N):
        mesh = UnitSquareMesh(N,N)
        V = FunctionSpace(mesh, 'CG', degree)
        u = TrialFunction(V)
        v = TestFunction(V)
        return mesh, V, u, v
    # construct FEM space
    mesh, V, u, v = setup_FEM(N)
    newMesh = UnitSquareMesh(N,N)
    newV = FunctionSpace(newMesh, 'CG', degree)

    # dof to vertex map
    dof2vert = dof_to_vertex_map(V)
    newDof2vert = dof_to_vertex_map(newV)
    # coords will be used for interpolation of covariance kernel
    coords = mesh.coordinates()
    newCoords = newMesh.coordinates()
    # but we need degree of freedom ordering of coordinates
    coords = coords[dof2vert]
    newCoords = newCoords[newDof2vert]

    # assemble mass matrix and convert to scipy
    M = assemble(u*v*dx)
    M = M.array()

    print("size of M: ")
    print(len(M))

    # evaluate covariance matrix
    L = coords.shape[0]
    newL = newCoords.shape[0]
    # if True: # vectorised
    #         c0 = np.repeat(coords, L, axis=0)
    #         c1 = np.tile(coords, [L,1])
    #         r = np.abs(np.linalg.norm(c0-c1, axis=1))
    #         C = cov(r)
    #         #C = cov(c0-c1)
    #         C.shape = [L,L]
    # else:   # slow validation
    #     C = np.zeros([L,L])
    #     for i in range(L):
    #         for j in range(L):
    #             if j <= i:
    #                 v = cov(np.linalg.norm(coords[i]-coords[j]))
    #                 C[i,j] = v
    #                 C[j,i] = v
    #
    C = np.zeros([L,L])

    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i]-coords[j]))
                C[i,j] = v
                C[j,i] = v



    # solve eigenvalue problem
    A = np.dot(M, np.dot(C, M))

    print("A shape")
    print(A.shape)
    print("M shape")
    print(M.shape)
    print("C shape")
    print(C.shape)

    # w, v = spla.eigsh(A, k, M)
    w, v = dla.eigh(A, b=M)

    print("finished eigenvalue problem solving...")
    print(time.time()-start_time)
    eig_time = time.time()


    # Initialize function and assign eigenvector
    #u = Function(V)
    #u.vector()[:] = rx

    #print(v[:, 1])
    #print("size of vector")
    #print(v[:, 1].shape)
    #print("coords[1]")
    #print(coords[1])
    #print(np.linalg.norm(coords[1]))
    #print(np.dot(v[:,1],v[:,1]))

    ############### check
    #C_eig = np.zeros([L,L])

    L = newL-1
    newC_eig = np.zeros([L,L])

    newC = np.zeros([L,L])
    #for i in range(L):
    #    for j in range(L):
    #        if j <= i:
    #            # sum it up
    #            v_eig = 0
    #            for k in range(len(w)):
    #                eFunc = set_fem_fun(v[:,k], V)
    #                v_eig = v_eig + w[k] * np.dot(eFunc(coords[i]), eFunc(coords[j]))#cov(np.linalg.norm(coords[i]-coords[j]))
    #            C_eig[i,j] = v_eig
    #            C_eig[j,i] = v_eig



    eFuncs = []
    for i in range (len(w)):
        eFuncs.append(set_fem_fun(v[:,i], V))

    eig_coords = []
    for i in range(len(w)):
        temp = []
        for j in range(L):
            temp.append(eFuncs[i](coords[j]/2+coords[j+1]/2))
        eig_coords.append(temp)



    print("finished set_fem_fun and substitute...")
    print(time.time()-eig_time)
    fun_time = time.time()

    for i in range(len(w)):
        newC_eig = newC_eig + w[i]*np.outer(eig_coords[i],eig_coords[i])

    #temp = np.zeros([L,L])

    #for i in range(L):
    #    for j in range(L):
    #        if j <= i:
    #            # sum it up
    #            v_eig = 0
    #            for k in range(len(w)):
    #                eFunc = eFuncs[k]
    #                v_eig = v_eig + w[k] * eFunc(coords[i]/2+coords[i+1]/2) * eFunc(coords[j]/2+coords[j+1]/2)#cov(np.linalg.norm(coords[i]-coords[j]))
    #            temp[i,j] = v_eig
    #            temp[j,i] = v_eig

    #print("checking for method...")
    #print((np.square(temp - newC_eig)).mean(axis=None))
    #print(temp)
    #print(newC_eig)
    #exit()

    print("finished newC_eig solving...")
    print(time.time()-fun_time)
    eigC_time = time.time()

    for i in range(L):
        for j in range(L):
            if j <= i:
                value = cov(np.linalg.norm(coords[i]/2 + coords[i+1]/2 - coords[j]/2 - coords[j+1]/2))
                newC[i,j] = value
                newC[j,i] = value
                #print(newC[i,j]-newC_eig[i,j])


    print("finished newC solving...")
    print(time.time()-eigC_time)
    C_time = time.time()


    # return eigenpairs
    # print ("===== EVP size =====", A.shape, w.shape, v.shape)
    #v = np.array([z[dof2vert] for z in v.T])
    return w, v, V, C, newC_eig, newC, M



def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval

start_time = time.time()

print("cov len")
print(cov_len(0.1))


w, v, V, C, newC, newC_eig, M = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.2, sigma=1.0), N = 100, degree = 1)

print("check cov")
#print((np.square(C - C_eig)).mean(axis=None))

print((np.square(newC - newC_eig)).mean(axis=None))
print("time")
print(time.time()-start_time)


idx = w.argsort()[::-1]
w = w[idx]
#v = v[:,idx]


print("Truncation error (v = 20)")
e = 20
eig = 0
for i in range(e):
    eig = eig + w[i]
print(1-eig/np.trace(np.dot(C, M)))
exit()

# check mode
firstE = set_fem_fun(v[:,1], V)
print("eigenvalue numbers: ")
print(w.shape)

print("eigenvector")
print(v.shape)

#plt.figure()
#plt.subplot(121)
#im = plot(firstE)
#plt.colorbar(im)
#plt.title("eigenvector")
# plt.savefig('eigenVector.png')
#plt.show()

randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(len(w)):
    randomField = randomField + sqrt(w[i]) * v[:,i] * gauss[i]

rF = set_fem_fun(randomField, V)

file = File("2D_Random.pvd")
file << rF

#plt.figure()
#im = plot(rF)
#plt.colorbar(im)
#plt.title("randomField")
# plt.savefig('randomField.png')
#plt.show()
