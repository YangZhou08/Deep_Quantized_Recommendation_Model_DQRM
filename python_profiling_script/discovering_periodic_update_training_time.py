import numpy as np 
import matplotlib.pyplot as plt 

cpu_periodic_update_200 = [58.26, 58.67, 58.49, 59.05, 58.73, 58.66, 61.11, 58.41, 58.39] 
cpu_periodic_update_500 = [50.88, 50.48, 51.01, 51.35, 50.33, 50.70, 51.65,  50.23, 50.44] 

cpu_200_mean = np.mean(cpu_periodic_update_200) 
print("Iteration period {}, training time {}".format(200, cpu_200_mean)) 
cpu_500_mean = np.mean(cpu_periodic_update_500) 
print("Iteration period {}, training time {}".format(500, cpu_500_mean)) 
