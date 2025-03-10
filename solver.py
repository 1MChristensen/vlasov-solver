import numpy as np

def advect_x(grid, x, v, dt):
    
    # First flatten the grid - this is row major so the x values are looped over within the v values
    flat = grid.flatten()

    # Assuming that nx is even, we split this into two matrices for an upwind scheme
    # Negative velocities
    D_neg = np.diag(np.ones(len(x)), k=0) - np.diag(np.ones(len(x)-1), k=-1)
    v_neg = np.diag(v[:len(v)//2])
    A_neg = np.kron(v_neg, D_neg)

    # Positive velocities
    D_pos = np.diag(np.ones(len(x)-1), k=1) - np.diag(np.ones(len(x)), k=0)
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

    for i in range(len(v)):
        for j in range(len(x)):
            if E[j] > 0:
                A[i*len(x)+j, i*len(x)+j] = -E[j]
                A[i*len(x)+j, i*len(x)+j-1] = E[j]
            else:
                A[i*len(x)+j, i*len(x)+j] = E[j]
                A[i*len(x)+j, i*len(x)+j+1] = -E[j]
    
    I = np.eye(len(x)*len(v))

    dv = v[1] - v[0]

    flat = (I + dt*A/dv)@flat

    # Now reshape the grid
    grid = flat.reshape((len(v), len(x)))

    return grid

def solve_poisson(grid, x, v, dt):
    return