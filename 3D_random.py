from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg


# exponential covariance function
def cov_exp(r, rho, sigma2=1.0):
    return sigma2 * np.exp(-r*r/2.0/rho/rho)

def solve_covariance_EVP(cov, N, degree=1):
    """

    """
    def setup_FEM(N):
        mesh = UnitCubeMesh(24,N,N)
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
    print(M.shape)

    exit()

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
    newC = np.zeros([newL,newL])

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
    C_eig = np.zeros([L,L])
    newC_eig = np.zeros([newL,newL])
    for i in range(L):
        for j in range(L):
            if j <= i:
                # sum it up
                v_eig = 0
                for k in range(len(w)):
                    eFunc = set_fem_fun(v[:,k], V)
                    v_eig = v_eig + w[k] * np.dot(eFunc(coords[i]), eFunc(coords[j]))#cov(np.linalg.norm(coords[i]-coords[j]))
                C_eig[i,j] = v_eig
                C_eig[j,i] = v_eig

    L = newL-1

    for i in range(L):
        for j in range(L):
            if j <= i:
                # sum it up
                v_eig = 0
                for k in range(len(w)):
                    eFunc = set_fem_fun(v[:,k], V)
                    v_eig = v_eig + w[k] * np.dot(eFunc(coords[i]/2+coords[i+1]/2), eFunc(coords[j]/2+coords[j+1]/2))#cov(np.linalg.norm(coords[i]-coords[j]))
                newC_eig[i,j] = v_eig
                newC_eig[j,i] = v_eig


    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i]/2+coords[i+1]/2 - coords[j]/2 - coords[j+1]/2))
                newC[i,j] = v
                newC[j,i] = v
                print(newC[i,j]-newC_eig[i,j])

    # return eigenpairs
    # print ("===== EVP size =====", A.shape, w.shape, v.shape)
    #v = np.array([z[dof2vert] for z in v.T])
    return w, v, V, C, C_eig, newC_eig, newC



def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval



w, v, V, C, C_eig, newC, newC_eig = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.1, sigma2=1.0), N = 16, degree = 1)

print("check cov")
print((np.square(C - C_eig)).mean(axis=None))

print((np.square(newC - newC_eig)).mean(axis=None))


exit()

idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

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


plt.figure()
im = plot(rF)
plt.colorbar(im)
plt.title("randomField")
# plt.savefig('randomField.png')
plt.show()
