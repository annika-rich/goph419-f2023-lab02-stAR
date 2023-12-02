# Unit test for gauss_iter_solver function
from lab02.linalg_interp import gauss_iter_solver
import numpy as np

def test_gauss_iter_solver():
    # set coefficient matrix A and RHS vector b
    A = np.array([
        [9, 2, 4],
        [2, 7, 1],
        [3, 2, 6],
    ])
    b = np.array([10, 1, 2])
    guess = np.array([1, 1, 1])

    # test jacobi approach with no initial guess
    sol_jac = gauss_iter_solver(A, b, x0 = None, alg = 'jacobi')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (jacobi) solution:\n {sol_jac}.\nThe numpy solution (np.linalg.solve): {sol_np}\n")

    # test jacobi with initial guess
    sol_jac = gauss_iter_solver(A, b, x0 = guess, alg = 'jacobi')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (jacobi) solution (with initial guess):\n {sol_jac}.\nThe numpy solution (np.linalg.solve): {sol_np}\n")

    A = np.array([
        [12, 2, -4],
        [0, 1, 1],
        [2, 2, 6],
    ])
    b = np.array([6, 1, 3])
    # test siedel approach with no initial guess
    sol_seid = gauss_iter_solver(A, b, x0 = None, alg = 'seidel')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (seidel) solution:\n {sol_seid}.\nThe numpy solution (np.linalg.solve): {sol_np}\n")

    # test seidel approach with initial guess
    sol_sied = gauss_iter_solver(A, b, x0 = guess, alg = 'seidel')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (seidel) solution (with initial guess):\n {sol_seid}.\nThe numpy solution (np.linalg.solve): {sol_np}\n")

    # test a RHS vector where the result is A^-1
    A = np.array([
        [1, 1, -2],
        [4, 11, 7],
        [3, 2, 4],
    ])
    b = np.eye(len(A))
    x = gauss_iter_solver(A, b)
    sol_x = np.linalg.solve(A, b)
    print(f"The gauss-siedel (seidel) solution:\n {x}.\nThe numpy solution (np.linalg.solve):\n {sol_x}\n")
    # check if A*A^-1 = I
    I = A @ x
    print(f"[A][A^-1] =\n {np.round(I, decimals = 0)}")


if __name__ == "__main__":
    test_gauss_iter_solver()