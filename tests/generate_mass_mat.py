import dolfin
import dolfin_navier_scipy.dolfin_to_sparrays as dts

from scipy.io import mmwrite

N = 25


mesh = dolfin.UnitSquareMesh(N, N)
V = dolfin.FunctionSpace(mesh, 'CG', 1)
u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

mass = dolfin.assemble(v*u*dolfin.dx)
massmat = dts.mat_dolfin2sparse(mass)

matstring = 'testdata/massmat_square_CG1_N{0}'.format(N)

mmwrite(matstring, massmat)
