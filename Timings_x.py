import solver
import numpy as np
import timeit
import matplotlib.pyplot as plt

# Parameters
Lx = 4*np.pi
ve_max = 4.5
T = 40
q = -1
m = 1

# Number of x grid points
Nx = [2,4,6,8,10,12,14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# Velocity grid
nv = 32
v = np.linspace(-ve_max, ve_max, nv)
dv = v[1] - v[0]

# Initial distribution
def initial(x,v):
    alpha = 0.01; k = 0.5
    return (1+alpha*np.cos(k*x))*np.exp(-v**2/2)/np.sqrt(2*np.pi)


times = []
for nx in Nx:
    # Position grid
    x = np.linspace(0, Lx, nx)
    dx = x[1] - x[0]

    # Time
    E_max = 0.020887900662348195
    dt = min(dx/ve_max, dv/E_max)
    dt = dx/ve_max
    nt = int(round(T / dt))
    t = np.linspace(0, T, nt)

    # Electron distribution
    f_e = initial(x[None,:], v[:,None])

    # Uniform neutralising background ions
    f_i = np.ones((nv, nx))/ np.trapz(np.ones((nv, nx)), v, axis=0)

    def f(f_e=f_e, f_i=f_i, x=x, v=v, dt=dt, q=q, m=m, t=t):
        for ts in t:
            # First advect in x for dt/2
            f_e = solver.advect_x(f_e, x, v, dt/2)

            # Second solve poisson/ampere equation for dt/2
            E = solver.solve_poisson(f_e, f_i, x, v, dt/2)

            # Thirdly advect in v for dt
            f_e = solver.advect_v(f_e, q/m * E, x, v, dt)

            # Finally advect in x again for dt/2
            f_e = solver.advect_x(f_e, x, v, dt/2)

    time = timeit.timeit(f, number=1)
    times.append(time)
    print(f"nx = {nx}, time = {time}")

plt.plot(Nx, times)
plt.xlabel('$N_x$')
plt.ylabel('Time (s)')
plt.savefig('Timings_x.pdf')
plt.show()