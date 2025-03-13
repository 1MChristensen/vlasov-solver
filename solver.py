import numpy as np
import numpy.fft as fft

def advect_x(grid, x, v, dt):
    
    # First flatten the grid - this is row major so the x values are looped over within the v values
    flat = grid.flatten()

    # Assuming that nx is even, we split this into two matrices for an upwind scheme
    # Negative velocities
    D_neg = np.diag(np.ones(len(x)), k=0) - np.diag(np.ones(len(x)-1), k=-1)
    D_neg[0, 0] = 1; D_neg[0, -1] = -1
    v_neg = np.diag(v[:len(v)//2])
    A_neg = np.kron(v_neg, D_neg)

    # Positive velocities
    D_pos = np.diag(np.ones(len(x)-1), k=1) - np.diag(np.ones(len(x)), k=0)
    D_pos[-1, -1] = -1; D_pos[-1, 0] = 1
    v_pos = np.diag(v[len(v)//2:])
    A_pos = np.kron(v_pos, D_pos)

    # Combine the two matrices
    #print(A_neg.shape, A_pos.shape)
    A = np.block([[A_neg, np.zeros(A_neg.shape)], [np.zeros(A_pos.shape), A_pos]])
    #print(A_neg.shape)

    I = np.eye(len(x)*len(v))
    

    #print(I.shape, A.shape, flat.shape)
    dx = x[1] - x[0]
    flat = (I + dt*A/dx)@flat

    # Now reshape the grid
    grid = flat.reshape((len(v), len(x)))

    return grid 


def advect_v(grid, E, x, v, dt):
    # First flatten the grid - this is row major so the x values are looped over within the v values
    flat = grid.flatten()

    # Upwinding scheme for E
    A = np.zeros((len(v)*len(x), len(v)*len(x)))

    for i in range(1,len(v)-1):
        for j in range(len(x)):
            if E[j] > 0:
                A[i*len(x)+j, i*len(x)+j] = -E[j]
                A[i*len(x)+j, (i+1)*len(x)+j] = E[j]
            else:
                A[i*len(x)+j, i*len(x)+j] = E[j]
                A[i*len(x)+j, (i-1)*len(x)+j] = -E[j]

    # Periodic boundary conditions
    for j in range(len(x)):
        if E[j] <= 0:
            # Flow across the boundary
            A[j,j] = E[j]
            A[j, len(x)*(len(v) - 1) + j] = -E[j]
        else:
            # No flow across the boundary
            A[j,j] = -E[j]
            A[j, j + len(x)] = E[j]

        if E[j] > 0:
            # Flow across the boundary
            A[len(x)*(len(v) - 1) + j, len(x)*(len(v) - 1) + j] = -E[j]
            A[len(x)*(len(v) - 1) + j, j] = E[j]
        else:
            # No flow across the boundary
            A[len(x)*(len(v) - 1) + j, len(x)*(len(v) - 1) + j] = E[j]
            A[len(x)*(len(v) - 1) + j, len(x)*(len(v) - 2) + j] = -E[j]

    #print(A)
    I = np.eye(len(x)*len(v))

    dv = v[1] - v[0]

    flat = (I + dt*A/dv)@flat

    # Now reshape the grid
    grid = flat.reshape((len(v), len(x)))

    return grid


def solve_poisson(grid1, grid2, x, v, dt):

    # For each x evaluate the integral of f_e and f_i
    rho_e = np.trapz(grid1, v, axis=0)

    # For uniform ions this should be 1
    rho_i = np.trapz(grid2, v, axis=0)

    net_rho = rho_i - rho_e

    dx = x[1] - x[0]

    rho_k = fft.fft(net_rho)
    kx = fft.fftfreq(len(x), d=dx) #* 2 * np.pi

    nonzero_k = kx != 0
    phi_k = np.zeros_like(rho_k, dtype=complex)
    phi_k[nonzero_k] = -rho_k[nonzero_k] / (kx[nonzero_k]**2)

    E_k = -1j * kx * phi_k

    E = np.fft.ifft(E_k).real

    return -E 