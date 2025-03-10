import solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Parameters
Lx = 1
v_max = 1
nx = 50
nv = 50
T = 10

CFL = 0.8
E_max = 1
v_max = 1

x = np.linspace(0, Lx, nx)
v = np.linspace(-v_max, v_max, nv)

dx = x[1] - x[0]
dv = v[1] - v[0]

q = m = 1

dt = min(dx/v_max, dv/E_max)
nt = int(round(T / dt))
print(f"dt = {dt}, nt = {nt}")
t = np.linspace(0, T, nt)

grid = np.zeros((nx, nv))

E = np.ones(len(x))

# Loop over all time steps
for ts in tqdm(t):
    # First advect in x for dt/2
    grid = solver.advect_x(grid, x, v, dt)
    # Second solve poisson/ampere equation for dt/2
    #solver.solve_poisson(grid, x, v, dt)
    # Thirdly advect in v for dt
    solver.advect_v(grid, E, x, v, dt)
    # Finally advect in x again for dt/2
    solver.advect_x(grid, x, v, dt)


