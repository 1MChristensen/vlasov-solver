import solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Parameters
Lx = 1
v_max = 1
nx = 10
nv = 20
T = 3

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
t = np.linspace(0, T, nt)
print(f"dt = {dt}, nt = {nt}")

f_e = np.zeros((nv, nx))
# Uniform neutralising background ions
f_i = np.ones((nv, nx))

E = np.ones(len(x))

# Loop over all time steps
for ts in tqdm(t):
    # First advect in x for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

    # Second solve poisson/ampere equation for dt/2
    solver.solve_poisson(f_e, f_i, x, v, dt/2)

    # Thirdly advect in v for dt
    f_e = solver.advect_v(f_e, E, x, v, dt/2)

    # Finally advect in x again for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)
