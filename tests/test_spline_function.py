# unit test for spline_function

from lab02.linalg_interp import spline_function
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def test_spline_function():
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

if __name__ == "__main__":
    test_spline_function()