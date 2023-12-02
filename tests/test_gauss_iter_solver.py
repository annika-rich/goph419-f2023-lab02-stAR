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
    # test jacobi approach with no initial guess
    sol_jac = gauss_iter_solver(A, b, x0 = None, alg = 'jacobi')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (jacobi) solution:\n {sol_jac}.\nThe numpy solution (np.linalg.solve): {sol_np}")

    # jacobi with initial guess
    guess = np.array([1, 1, 1])
    # test jacobi approach with no initial guess
    sol_jac = gauss_iter_solver(A, b, x0 = guess, alg = 'jacobi')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (jacobi) solution (with initial guess):\n {sol_jac}.\nThe numpy solution (np.linalg.solve): {sol_np}")

    # test siedel approach with no initial guess
    sol_sei = gauss_iter_solver(A, b, x0 = None, alg = 'seidel')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (seidel) solution:\n {sol_jac}.\nThe numpy solution (np.linalg.solve): {sol_np}")

    # test seidel approach with no initial guess
    sol_sied = gauss_iter_solver(A, b, x0 = guess, alg = 'seidel')
    sol_np = np.linalg.solve(A,b)
    print(f"The gauss-siedel (seidel) solution (with initial guess):\n {sol_jac}.\nThe numpy solution (np.linalg.solve): {sol_np}")




if __name__ == "__main__":
    test_gauss_iter_solver()