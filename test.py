import numpy as np
import matplotlib.pyplot as plt
import solver
from tqdm import tqdm 

nx = 30
nv = 30

x = np.linspace(-1, 1, nx)
v = np.linspace(-1, 1, nv)

dx = x[1] - x[0]
dv = v[1] - v[0]

v_max = 1
E_max = 1

dx = x[1] - x[0]

dt = min(dx/v_max, dv/E_max)
T = 10
nt = int(round(T / dt))

j = len(v) // 2
j = 0

grid = np.where(np.abs(x) < 1.0 / 3.0, 1.0, 0.0)
grid = np.tile(grid, (nv, 1))

plt.plot(x, grid[j], label='initial')

for i in tqdm(range(15)):
    grid = solver.advect_x(grid, x, v, dt)


plt.plot(x, grid[j], label='final')
plt.legend()
plt.show()

grid = np.where(np.abs(v) < 1.0 / 3.0, 1.0, 0.0)
grid = np.tile(grid, (nv, 1))
grid = grid.T


E = -np.ones(len(x))

plt.plot(x, grid[:,j], label='initial')

for i in tqdm(range(15)):
    grid = solver.advect_v(grid, E, x, v, dt)

plt.plot(v, grid[:,j], label='final')
plt.legend()
plt.show()

