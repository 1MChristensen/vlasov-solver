import solver
import numpy as np

# Parameters
Lx = 4*np.pi
ve_max = 4.5
T = 40

Nx = [5,10,15,20,25,30,35,40,45,50]

nv = 32
v = numpy.linspace(-ve_max, ve_max, nv)

for nx in Nx:
    x = numpy.linspace(0, Lx, nx)
