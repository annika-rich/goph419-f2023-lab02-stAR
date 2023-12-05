# Utility function file for lab02
# Author: Annika Richardson
# Some parts of the code implementation use examples dones in class by Brandon Karchewski and are adapted from Chapra & Clough (2022)

import numpy as np

def gauss_iter_solver(A, b, x0 = None, tol = 1e-8, alg = 'seidel'):
    """ This function implements the iterative Gauss-Seidel Approach to solve linear systems of the form Ax = b.

    Parameters
    ----------
    A: array-like, shape(n, n)
       coefficient matrix of linear system

    b: array-like, shape (n, m) where m >= 1
       right-hand side of linear system

    x0: (optional) array-like, shape (n,) or (n,m)
        initial guesses to solve system

    tol: (optional) float
         stopping criterion

    alg: (optional) string flag
         lists algorithm, the two acceptable inputs are seidel or jacobi

    Returns
    -------
    numpy.array, shape (n,m)
        The solution vector

    Raises
    ------
    TypeError: 
        Checks that the alg flag is a string and contains either 'seidel' or 'jacobi'.

    ValueError:
        If coefficient matrix A is not 2D and square
        If rhs vector b is not 1D or 2D, or has a different number of rows than A
        If the initial guess x0 is not 1D or 2D, has a different shape than b, or has a different number of rows than A and b

    RuntimeWarning:
        If the system does not converge by 100 iterations to a specified error tolerance.
    """
    # check that coefficient matrix and constant vector are valid inputs
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # check that the coefficient matrix is square
    n = len(A)
    ndim = len(A.shape)
    if ndim != 2:
        raise ValueError(f"A has {ndim} dimensions"
                         + ", should be 2")
    if A.shape[1] != n:
        raise ValueError(f"A has {n} rows and {A.shape[1]} cols"
                         + ", should be square")

    # check that the rhs vector is 1D or 2D
    ndimb = len(b.shape)
    if ndimb not in [1, 2]:
        raise ValueError(f"b has {ndimb} dimensions"
                         + ", should be 1D or 2D")
    # check that number of rhs rows matches number of rows in A
    if len(b) != n:
        raise ValueError(f"A has {n} rows, b has {len(b)} values"
                         + ", dimensions incompatible")
     # if b is 1D convert b to a 2D column vector
    if b.ndim == 1:
        b = np.reshape(b, (n, 1))
                        
    # check if the alg flag is either siedel or jacobi (case insensitive and ignores trailing/leading whitespaces)
    alg = alg.strip().lower()
    if alg not in ['seidel', 'jacobi']:
        raise TypeError(f"The algorithm flag ({alg}) contains a string other than 'seidel' or 'jacobi'")

    # check and initialise x0
    if x0 is None:
        # make column vector x0 using column vector b
        x0 = np.zeros_like(b)
    else:
        # make sure x0 is an np.array
        x0 = np.array(x0, dtype=float)
        if x0.ndim == 1:
            # reshape x0 to match dimensions of b and convert to column vector
            x0 = np.reshape(x0, (n,1))
        if x0.ndim not in [1,2]:
            raise ValueError(f"x0 has {x0.ndim}, should be 1D or 2D")
        # make sure x0 has the same number of rows as A and b
        if len(x0) != n:
            raise ValueError(f"x0 has {x0.shape[0]} rows, A and b have {n} rows, dimemsions incompatible")
    
    # set number of maximum iterations (solution must converge before this number is reached)
    maxit = 100
    # approximate relative error variable, ensures that loop will execute at least once
    eps_a = 2 * tol
    # set up A_d with main diagonal entries of coefficient matrix A and zeros elsewhere
    A_d = np.diag(np.diag(A))
    # inverse of A_d (note: this is computationally inexpensive because it only involves scalar inversion of each diagonal entry)
    A_d_inv = np.linalg.inv(A_d)
    # Determine normalized matrix A^*
    A_ = A - A_d
    A_star = A_d_inv @ A_
    # Determine normalized matrix B^*
    B_star = A_d_inv @ b
    
    # set iteration counter
    itr = 1
    while np.max(eps_a) > tol and itr < maxit:
        if alg == 'jacobi':
            x_old = np.array(x0)
            x0 = B_star - (A_star @ x_old)
        elif alg == 'seidel':
            x_old = np.array(x0)
            for i, j in enumerate(A):
                x0[i,:] = B_star[i:(i+1),:] - A_star[i:(i+1),:] @ x0
        # calculate error at each iteration
        num = x0 - x_old
        eps_a = np.linalg.norm(num) / np.linalg.norm(x0)
        itr += 1
        # system must converge over 100 iterations, if it does not, a runtime warning is raised
        if itr >= maxit:
            raise RuntimeWarning(f"The system did not converge over {itr} iterations. The approximate error ({np.max(eps_a)}) is greater than ({tol}) the specified error tolerance.")

    return x0


def spline_function(xd, yd, order = 3):
    """Function that generates a spline function for two given vectors of x and y data.

    Parameters
    ----------
    xd: array-Like of floats
        independent variables

    yd: array-like of floats
        dependent variables
        must be same shape as xd

    order: int
           possible values of polynomial order are 1st, 2nd, and 3rd order (i.e., 1, 2, 3)
           default order is 3

    Returns
    -------
    function 
        takes 1 parameter (float or array-like of floats)
        returns interpolated y-values

    Raises
    ------
    ValuError
        if shape xd is not equal to yd
        if xd has repeated values (number of independent variables is not equal to the number of dependent variables)
    """
    # make sure xd and yd are arrays
    xd = np.array(xd, dtype = float)
    yd = np.array(yd, dtype = float)

    # check that xd and yd have the same length
    if (m := len(xd)) != (n := len(yd)):
        raise ValueError(f"The length of xd ({m}) is not equal to the lengtht of yd ({n}), xd and yd must be the same shape")

    # check for repeated values of xd
    unique = np.unique(xd)
    if (u := len(unique)) != m:
        raise ValueError(f"The number of independent variables ({u}) is not equal to the number of dependent variable ({n}).\nThe array xd has repeated values.")

    # check that xd values are in increasing order
    xd = np.sort(xd, axis = 0)
    # double check they x-values sorted properly
    if (all (xd[i] >= xd[i+1] for i in range(m-1))):
        raise ValueError(f"The values of xd:\n{xd}\nMust be in increasing order.")
    
    if order not in [1, 2, 3]:
        raise ValueError(f"The order given ({order}) is not 1, 2, or 3")

    # deetermine differences for xd and yd
    diff_x = np.diff(xd)
    diff_y = np.diff(yd)

    # determine first order divided difference
    div_dif1 = diff_y / diff_x

    if order == 1:
        def spline_1(x):
            """Linear spline function.

            Inputs
            ------
            x: float or array-like of floats

            Returns
            -------
            Interpolated value of y (or y-values)
            """
            # determine spline function coefficients a and b
            a = yd[:-1]
            b = div_dif1[:-1]
            # determine spline function between known data points in xd
            for xi in x:
                # find indices to determine where xd is larger than xi to interpolate along interval between points
                i = np.array([np.nonzero(xd >= xi)[0][0] - 1 for xi in x])
                i = np.where(i < 0, 0, i)
                # calculate spline functions
                y = a[i-1] + b[i-1] * (x - xd[i-1])
            return y
        return spline_1

    elif order == 2:
        def spline_2(x):
            """Quadratic Spline function.

            Inputs
            ------
            x: float or array-like of floats

            Returns
            -------
            Interpolated value of y (or y-values)
            """
            # set up linear system of equations to solve for a, b, c unknowns
            # set up RHS
            N = m - 1
            rhs = np.zeros(N)
            rhs[1:] = np.diff(div_dif1, axis = 0)
            # set up coefficient matrix
            A = np.zeros((N, N))
            # values of first and last rows of A
            A[0,0:2] = [1, -1]
            A[1:,:-1] += np.diag(diff_x[:-1])
            A[1:,1:] += np.diag(diff_x[1:])
            # determine coefficients
            c = np.linalg.solve(A, rhs)
            # c = gauss_iter_solver(A, rhs)
            b = diff_y - (c * diff_x)
            a = yd[:-1]
            # calculate spline functions
            for xi in x:
                # determine indexing intervals where spline function will interpolate between points
                i = np.array([np.nonzero(xd >= xi)[0][0] - 1 for xi in x])
                i = np.where(i < 0, 0, i)
                # spline function over at index i
                y = a[i] + b[i] * (x - xd[i]) + c[i] * (x - xd[i]) ** 2
            return y
        return spline_2

    elif order == 3:
        def spline_3(x):
            """Cubic Spline function.

            Inputs
            ------
            x: float or array-like of floats

            Returns
            -------
            Interpolated value of y (or y-values)
            """
            # set up linear system of equations to solve for unknowns
            N = m
            div_dif2 = np.diff(div_dif1)
            rhs = np.zeros(N)
            rhs[1:-1] = 3 * div_dif2
            # set up coefficient matrix
            A = np.zeros((N, N))
            A[1, 0] = diff_x[0]
            A[-2, -1] = diff_x[-1]
            A[0,:3] = [-diff_x[1], (diff_x[0] + diff_x[1]), -diff_x[-2]]
            A[-1,-3:] = [-diff_x[-1], (diff_x[-1]+diff_x[-2]), -diff_x[-2]]
            A[1:-1,:-2] += np.diag(diff_x[:-1])
            A[1:-1,1:-1] += np.diag(2 * (diff_x[:-1] + diff_x[1:]))
            A[1:-1,2:] += np.diag(diff_x[1:])
            # calculate coefficients
            c = np.linalg.solve(A, rhs)
            #c = gauss_iter_solver(A, rhs)
            d = np.diff(c) / (diff_x * 3)
            b = div_dif1 - diff_x * (c[:-1] + c[1:] * 2) / 3
            # get indexes for spline function interpolation
            i = np.array([np.nonzero(xd >= xi)[0][0] - 1 for xi in x])
            i = np.where(i < 0, 0, i)
            y = np.array([(yd[i] + b[i] * (xi - xd[i]) + c[i] * (xi - xd[i]) ** 2 + d[i] * (xi - xd[i]) ** 3) for i, xi in zip(i, x)])
            return y
        return spline_3



