import solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.signal import find_peaks

# Parameters
Lx = 4*np.pi
ve_max = 4.5
nx = 32
nv = 32
T = 10 

x = np.linspace(0, Lx, nx, endpoint=False)
v = np.linspace(-ve_max, ve_max, nv, endpoint=False)

dx = x[1] - x[0]
dv = v[1] - v[0]

q = -1
m = 1

def initial(x,v):
    alpha = 0.01; k = 0.5
    return (1+alpha*np.cos(k*x))*np.exp(-v**2/2)/np.sqrt(2*np.pi)

E_max = 0.020887900662348195
dt = min(dx/ve_max, dv/E_max)
nt = int(round(T / dt))
t = np.linspace(0, T, nt)

print(f"dt = {dt}, nt = {nt}")

f_e = initial(x[None,:], v[:,None])

# Uniform neutralising background ions
f_i = np.ones((nv, nx))/ np.trapz(np.ones((nv, nx)), v, axis=0)

E_1 = []

plot = False
if plot:
    X, V = np.meshgrid(x, v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, V, f_e, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('f_e')
    plt.show()


# Loop over all time steps
for ts in t:
    # First advect in x for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

    # Second solve poisson/ampere equation for dt/2
    E = solver.solve_poisson(f_e, f_i, x, v, dt/2)

    # Thirdly advect in v for dt
    f_e = solver.advect_v(f_e, q/m * E, x, v, dt)

    # Finally advect in x again for dt/2
    f_e = solver.advect_x(f_e, x, v, dt/2)

    E_k = np.fft.fft(E)
    
    E_1.append(np.abs(E_k[1])/nx/4)

    #energy_x = np.trapz(f_e, v, axis=0)
    #energy = np.trapz(energy_x, x)
    #print(energy)

if plot:
    X, V = np.meshgrid(x, v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, V, f_e, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('f_e')
    plt.show()

plot = True

log_E = np.log(E_1)
x_peaks = find_peaks(log_E)[0]

if plot:
    plt.plot(t, E_1)
    plt.plot(t[x_peaks], np.exp(log_E[x_peaks]), "x")
    plt.xlabel('Time')
    plt.ylabel('$E_1$')
    plt.yscale('log')
    plt.show()

# Find the gradients of the line that fits the peaks
peak_times = t[x_peaks]
peak_values = log_E[x_peaks]

# Fit a line to the peaks
coeffs = np.polyfit(peak_times, peak_values, 1)
gradient = coeffs[0]

print(f"Gradient of the line that fits the peaks: {-gradient}")
print(f"Percentage difference: {np.abs(0.153359-(-gradient))/0.153359*100}%")
