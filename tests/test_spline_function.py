# unit test for spline_function

from lab02.linalg_interp import spline_function
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def test_spline_function():
    # test to check whether spline function returns values as expected from linear, quadratic, and cubic funcitons
    # generate x values
    x = np.linspace(-20, 20, 20)

    # create linear, quadratic, and cubic functions
    linear = 30 * x
    quadratic = 24 * (x ** 2) + 5 * x + 12
    cubic = (x ** 3) + 7 * (x ** 2) - 4 * x + 2

    # interpolate y values using spline function
    f1 = spline_function(x, linear, order = 1)
    y1 = f1(x)

    f2 = spline_function(x, quadratic, order = 2)
    y2 = f2(x)

    f3 = spline_function(x, cubic, order = 3)
    y3 = f3(x)

    # plot functions and spline interpolations
    plt.plot(x, linear, 'ro', label = 'linear function')
    plt.plot(x, y1, 'r', label = 'linear interpolation')

    plt.plot(x, quadratic, 'gx', label = 'quadratic function')
    plt.plot(x, y2, 'g', label = 'quadratic interpolation')

    plt.plot(x, cubic, 'mD', label = 'cubic function')
    plt.plot(x, y3, 'm', label = 'cubic interpolation')

    plt.title('Spline Function Test for Linear, Quadratic, and Cubic Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.savefig('tests/spline_test1')
    plt.close('all')

    # check spline_function(order = 3) against scipy.interpolate.UnivariateSpline() function
    xd = np.linspace(1, 30, 50)
    yd = np.exp(xd)

    f_scipy = UnivariateSpline(xd, yd, k = 3, s = 0, ext = 'raise')
    y_scipy = f_scipy(xd)

    f_spline = spline_function(xd, yd, order = 3)
    y_spline = f_spline(xd)

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,12))

    axs[0].plot(xd, yd, 'ko', label = 'data')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].plot(xd, yd, 'ko', label = 'data')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    axs[0].plot(xd, y_scipy, 'b', label = 'scipy interpolation')
    axs[1].plot(xd, y_spline, 'm', label = 'GOPH419 spline_function')

    fig.suptitle('Scipy Univariate Spline Function vs GOPH419 Spline Function')
    fig.set_label('y')
    axs[0].legend()
    axs[1].legend()

    plt.savefig('tests/spline_test2')


if __name__ == "__main__":
    test_spline_function()