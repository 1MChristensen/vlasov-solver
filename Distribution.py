import solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


Lx = 2*np.pi/0.3
ve_max = 8

nx = 32
x = np.linspace(0, Lx, nx)
dx = x[1] - x[0]

nv = 32
v = np.linspace(-ve_max, ve_max, nv)
dv = v[1] - v[0]

T = 30

def initial(x,v):
    n_p = 0.9
    nb = 0.2
    vb = 4.5
    vt = 0.5
    alpha = 0.04
    k = 0.3
    return (1+alpha*np.cos(k*x))*((n_p/np.sqrt(2*np.pi))*np.exp(-v**2/2) + nb/np.sqrt(2*np.pi)*np.exp(-((v-vb)/vt)**2/2))

q = -1
m = 1

E_max = 0.6
dt = min(dx/ve_max, dv/E_max)
dt = dx/ve_max
nt = int(round(T / dt))
t = np.linspace(0, T, nt)

f_e = initial(x[None,:], v[:,None])

f_i = np.ones((nv, nx))/ np.trapz(np.ones((nv, nx)), v, axis=0)

max_abs_E = []

X, V = np.meshgrid(x, v)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, V, f_e, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('v')
ax.set_zlabel('f_e')
plt.show()

for ts in tqdm(t):
    # First advect in x for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

    # Second solve poisson/ampere equation for dt/2
    E = solver.solve_poisson(f_e, f_i, x, v, dt/2)

    # Thirdly advect in v for dt
    f_e = solver.advect_v(f_e, q/m * E, x, v, dt/2)

    # Finally advect in x again for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

X, V = np.meshgrid(x, v)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, V, f_e, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('v')
ax.set_zlabel('f_e')
plt.show()