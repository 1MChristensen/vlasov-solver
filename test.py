import numpy as np
import matplotlib.pyplot as plt
import solver
from tqdm import tqdm 

nx = 50
nv = 50

x = np.linspace(-1, 1, nx, endpoint=False)
v = np.linspace(-1, 1, nv, endpoint=False)

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

plot = False
if plot:
    plt.plot(x, grid[j], label='initial')

for i in tqdm(range(15)):
    grid = solver.advect_x(grid, x, v, dt)

if plot:
    plt.plot(x, grid[j], label='final')
    plt.legend()
    plt.show()

grid = np.sin(2*np.pi*v)
grid = np.tile(grid, (nx, 1))
grid = grid.T


E = np.ones(len(x))

plot = False 
if plot:
    plt.plot(v, grid[:,j], label='initial')

for i in tqdm(range(1)):
    grid = solver.advect_v(grid, E, x, v, dt)

if plot:
    plt.plot(v, grid[:,j], label='final')
    plt.legend()
    plt.show()


grid1 = np.cos(2*np.pi*x)/(v[-1]-v[0])
grid1 = np.tile(grid1, (nv, 1))
grid2 = np.ones_like(grid1)/np.trapz(np.ones_like(grid1), v, axis=0)

plot = True
if plot:
    plt.plot(x, grid1[j]*(v[-1]-v[0]), label='initial')

E = solver.solve_poisson(grid1, grid2, x, v, dt)

if plot:
    plt.plot(x, E, label='E')
    plt.legend()
    plt.show()