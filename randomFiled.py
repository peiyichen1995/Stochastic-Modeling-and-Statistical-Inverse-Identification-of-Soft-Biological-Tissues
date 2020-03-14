from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dolfin import * # UnitSquareMesh, FunctionSpace, plot, Function, Expression, interpolate
from functools import partial

# % matplotlib inline






class CoefField:
    """ artificial M-term KL
    """
    def __init__(self, mean, expfield, M, scale=1.0, decay=2.0):
        # type: (float, bool, int, float, float) -> None
        """
         initialise field with given mean.
         field is log-uniform for expfield=True.
         length M of expansion can optionally be fixed in advance.
         usually it is determined by the length of y when evaluating realisations.
        :param mean: mean value of the field
        :param expfield: Switch to go to lognormal field
        :param M: maximal number of terms in realisation
        :param decay: decay of the field
        :return: void
        """
        #                                           # create a Fenics expression of the affine field
        self.a = Expression('C + S*cos(A*pi*F1*x[0]) * cos(A*pi*F2*x[1])', A=2, C=mean, F1=0, F2=0, S=1, degree=5)
        self.mean = mean
        self.expfield = expfield
        self.M = M
        self.scale = scale
        self.decay = decay

    def realisation(self, y, V):
        # type: (List[float], FunctionSpace) -> Function
        """
          evaluate realisation of random field subject to samples y and return interpolation onto FEM space V.
        :param y: list of samples of the RV in [-1, 1]
        :param V: FunctionSpace
        :return: Fenics Function as field realisation
        """

        def indexer(i):
            m1 = np.floor(i/2)
            m2 = np.ceil(i/2)
            return m1, m2
        assert self.M == len(y)                     # strong assumption but convenient
        a = self.a                                  # store affine field Expression

        a.C, a.S = self.mean, 0                     # get mean function as starting point
        # x = interpolate(a, V).vector().array()      # interpolate constant mean on FunctionSpace
        # replace .array() with .get_local()
        x = interpolate(a, V).vector().get_local()
        a.C = 0                                     # set mean back to zero. From now on look only at amp_func
        #                                           # add up stochastic modes
        if self.M > 0:
            #                                       # get mean-length ratio
            for m, ym in enumerate(y):              # loop through sample items
                amp = (m+1) ** (-1 * self.decay)    # algebraic decay in front of the sin*sin
                a.F1, a.F2 = indexer(m+2)           # get running values in Expression
                a.S = self.scale                    # multiply a scaling parameter as well
                #                                   # add current Expression value
                # x += amp * ym * interpolate(a, V).vector().array()
                x += amp * ym * interpolate(a, V).vector().get_local()
        f = Function(V)                             # create empty function on FunctionSpace
        #                                           # set function coefficients to realisation coefficients
        f.vector()[:] = x if not self.expfield else np.exp(x)
        return f









def sample_expectation(rv, N):
    mean = 0
    for i in range(N):
        mean += rv.sample()
    mean *= N**(-1)
    return mean

def sample_variance(rv, N):
    var = 0
    mean = sample_expectation(rv, N)
    for i in range(N):
        var += rv.sample()**2
    var *= N**(-1)
    return var - mean**2






mesh = UnitSquareMesh(25, 25)
fs = FunctionSpace(mesh, 'CG', 1)

mean = 1.0
expfield = True
M = 10
scale = 1.0
decay = 2.0
affine_field = CoefField(mean, not expfield, M, scale, decay)
log_field = CoefField(mean, expfield, M, scale, decay)

def sample_field_affine():
    y = np.random.rand(M)*2 - 1
    # return affine_field.realisation(y, fs).vector().array()
    return affine_field.realisation(y, fs).vector().get_local()
def sample_field_log():
    y = np.random.randn(M)
    # return log_field.realisation(y, fs).vector().array()
    return log_field.realisation(y, fs).vector().get_local()

affine_field.sample = sample_field_affine
log_field.sample = sample_field_log





def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval







mean_affine = set_fem_fun(sample_expectation(affine_field, 100), fs)
mean_log = set_fem_fun(sample_expectation(log_field, 100), fs)

mpl.rcParams['figure.figsize'] = [8, 3]
plt.figure()
plt.subplot(121)
im = plot(mean_affine)
plt.colorbar(im)
plt.title("Mean of affine field")
plt.subplot(122)
im = plot(mean_log)
plt.colorbar(im)
plt.title("Mean of log normal field")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




var_affine = set_fem_fun(sample_variance(affine_field, 100), fs)
var_log = set_fem_fun(sample_variance(log_field, 100), fs)


plt.figure()
plt.subplot(121)
im = plot(var_affine)
plt.colorbar(im)
plt.title("Variance of affine field")
plt.subplot(122)
im = plot(var_log)
plt.colorbar(im)
plt.title("Variance of log normal field")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
