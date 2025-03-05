import solver
import numpy as np
# Parameters
Lx = 1
Lv = 1
nx = 10
nv = 10
nt = 10
T = 1

t = np.linspace(0, T, nt)
dt = t[1] - t[0]

grid = np.zeros((nx, nv))

x = np.linspace(0, Lx, nx)
v = np.linspace(0, Lv, nv)


# Loop over all time steps
for ts in t:
    # First advect in x for dt/2
    solver.advect_x(grid, x, v, dt)
    # Second solve poisson/ampere equation for dt/2
    solver.solve_poisson(grid, x, v, dt)
    # Thirdly advect in v for dt
    solver.advect_v(grid, x, v, dt)
    # Finally advect in x again for dt/2
    solver.advect_x(grid, x, v, dt)