import solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Parameters
Lx = 4*np.pi
ve_max = 4.5
nx = 32
nv = 32
T = 50

CFL = 0.9

x = np.linspace(0, Lx, nx)
v = np.linspace(-ve_max, ve_max, nv)

dx = x[1] - x[0]
dv = v[1] - v[0]

q = -1
m = 1

E_max = 2
dt = min(dx/ve_max, dv/E_max)
dt = dx/ve_max
nt = int(round(T / dt))
t = np.linspace(0, T, nt)

print(f"dt = {dt}, nt = {nt}")

def initial(x,v):
    alpha = 0.01; k = 0.5
    return (1+alpha*np.cos(k*x))*np.exp(-v**2/2)/np.sqrt(2*np.pi)

f_e = initial(x[None,:], v[:,None])

# Uniform neutralising background ions
f_i = np.ones((nv, nx))/ np.trapz(np.ones((nv, nx)), v, axis=0)

E = np.ones(len(x))

E_1 = []

# Loop over all time steps
for ts in tqdm(t):
    # First advect in x for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

    # Second solve poisson/ampere equation for dt/2
    E = solver.solve_poisson(f_e, f_i, x, v, dt/2)

    # Thirdly advect in v for dt
    f_e = solver.advect_v(f_e, q/m * E, x, v, dt/2)

    # Finally advect in x again for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

    E_k = np.fft.fft(E)
    freqs = np.fft.fftfreq(len(E), d=dx)
    pos_mask = freqs > 0
    
    fundamental_idx = np.argmax(np.abs(E_k[pos_mask])) + 1
    fundamental_freq = freqs[fundamental_idx]
    
    E_1.append(np.abs(E_k[fundamental_idx]))
    

print(E_1)
plt.plot(t, E_1)
plt.yscale('log')
plt.show()