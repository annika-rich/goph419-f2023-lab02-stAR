# Driver script for lab02
# Author: Annika Richardson

import numpy as np 

def main():
    water_data = np.loadtxt('water_density_vs_temp_usgs.txt')
    air_density_data = np.loadtxt('air_density_vs_temp_eng_toolbox.txt')

if __name__ == "__main__":
    main()