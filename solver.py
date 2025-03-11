import numpy as np

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
                A[i*len(x)+j, (i-1)*len(x)+j] = E[j]
            else:
                A[i*len(x)+j, i*len(x)+j] = E[j]
                A[i*len(x)+j, (i+1)*len(x)+j] = -E[j]

    # Periodic boundary conditions
    for j in range(len(x)):
        A[j,j] = E[j]
        A[j, len(v)*(len(x) - 1) + j] = -E[j]

        A[len(v)*(len(x) - 1) + j, len(v)*(len(x) - 1) + j] = E[j]
        A[len(v)*(len(x) - 1) + j, j] = -E[j]

    #print(A)
    I = np.eye(len(x)*len(v))

    dv = v[1] - v[0]

    flat = (I + dt*A/dv)@flat

    # Now reshape the grid
    grid = flat.reshape((len(v), len(x)))

    return grid


def solve_poisson(grid1, grid2, x, v, dt):

    # For each x evaluate the integral of f_e and f_i

    rho_e = np.zeros(len(x))
    rho_i = np.zeros(len(x))

    for i in range(len(x)):
        rho_e[i] = np.sum(grid1[:, i])  
        rho_i[i] = np.sum(grid2[:, i])

    dv = v[1] - v[0]

    rho_e = rho_e*dv
    rho_i = rho_i*dv

    # Solve poissons equation
    E = np.zeros(len(x))

    dx = x[1] - x[0]

    for i in range(1, len(x)-1):
        E[i] = E[i] + dx*(rho_e[i] - rho_i[i])

    return