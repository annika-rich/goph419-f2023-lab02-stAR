# Driver script for lab02
# Author: Annika Richardson

import numpy as np 
import matplotlib.pyplot as plt
from lab02.linalg_interp import spline_function

def main():
    # load water data
    water_data = np.loadtxt('data/water_density_vs_temp_usgs.txt')
    # set up x and y values
    xw = water_data[:, 0]
    yw = water_data[:, 1]
    # use linspace function to get 100 evenly spaced temperature values
    x_water = np.linspace(np.min(xw), np.max(xw), 100)

    # interpolate water density values using 1st order spline function
    f_w1 = spline_function(xw, yw, order = 1)
    y_water1 = f_w1(x_water)

    # set up subplots
    fig, axs = plt.subplots(nrows= 2, ncols = 3, figsize = (16, 12))
    fig.suptitle('Spline Interpolation for Water and Air Density Data')

    # plot first order water density interpolation
    axs[0][0].plot(x_water, y_water1, 'b--', label = 'linear spline')
    axs[0][0].plot(xw, yw, 'ko', label = 'discrete data')
    axs[0][0].set_ylabel('Water density (g/cm^3)')
    axs[0][0].set_xlabel('Temperature (C)')
    axs[0][0].set_title('First Order')
    axs[0][0].legend()

    # interpolate water density values using 2nd order spline function
    f_w2 = spline_function(xw, yw, order = 2)
    y_water2 = f_w2(x_water)

    # plot second order water density interpolation
    axs[0][1].plot(x_water, y_water2, 'b--', label = 'quadratic spline')
    axs[0][1].plot(xw, yw, 'ko', label = 'discrete data')
    axs[0][1].set_ylabel('Water density (g/cm^3)')
    axs[0][1].set_xlabel('Temperature (C)')
    axs[0][1].set_title('Second Order')
    axs[0][1].legend()

    # interpolate water density values using 3rd order spline function
    f_w3 = spline_function(xw, yw, order = 3)
    y_water3 = f_w3(x_water)

    # plot second order water density interpolation
    axs[0][2].plot(x_water, y_water3, 'b--', label = 'cubic spline')
    axs[0][2].plot(xw, yw, 'ko', label = 'discrete data')
    axs[0][2].set_ylabel('Water density (g/cm^3)')
    axs[0][2].set_xlabel('Temperature (C)')
    axs[0][2].set_title('Third Order')
    axs[0][2].legend()

    # load air density data
    air_density_data = np.loadtxt('data/air_density_vs_temp_eng_toolbox.txt')
    # set up x and y values
    xa = air_density_data[:, 0]
    ya = air_density_data[:, 1]
    # use linspace funciton to get 100 evenly spaces temperature values
    x_air = np.linspace(np.min(xa), np.max(xa), 100)

     # interpolate air density values using 1st order spline function
    f_a1 = spline_function(xa, ya, order = 1)
    y_air1 = f_a1(x_air)

    # plot first order air density interpolation
    axs[1][0].plot(x_air, y_air1, 'k--', label = 'linear spline')
    axs[1][0].plot(xa, ya, 'ro', label = 'discrete data')
    axs[1][0].set_ylabel('Air density (kg/m^3)')
    axs[1][0].set_xlabel('Temperature (C)')
    axs[1][0].set_title('First Order')
    axs[1][0].legend()

    # interpolate air density values using 2nd order spline function
    f_a2 = spline_function(xa, ya, order = 2)
    y_air2 = f_a2(x_air)

    # plot second order air density interpolation
    axs[1][1].plot(x_air, y_air2, 'k--', label = 'quadratic spline')
    axs[1][1].plot(xa, ya, 'ro', label = 'discrete data')
    axs[1][1].set_ylabel('Air density (kg/m^3)')
    axs[1][1].set_xlabel('Temperature (C)')
    axs[1][1].set_title('Second Order')
    axs[1][1].legend()

    # interpolate air density values using 3rdorder spline function
    f_a3 = spline_function(xa, ya, order = 3)
    y_air3 = f_a3(x_air)

    # plot second order air density interpolation
    axs[1][2].plot(x_air, y_air3, 'k--', label = 'cubic spline')
    axs[1][2].plot(xa, ya, 'ro', label = 'discrete data')
    axs[1][2].set_ylabel('Air density (kg/m^3)')
    axs[1][2].set_xlabel('Temperature (C)')
    axs[1][2].set_title('Third Order')
    axs[1][2].legend()

    plt.savefig('figures/density vs temperature graphs')


    
if __name__ == "__main__":
    main()