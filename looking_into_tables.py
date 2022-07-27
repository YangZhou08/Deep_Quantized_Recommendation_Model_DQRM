import numpy as np 
from matplotlib.cbook import ls_mapper
import matplotlib 
'''
matplotlib.use('Qt5Agg') 
''' 
import matplotlib.pyplot as plt 
import sys 

path = "/rscratch/data/dlrm_criteo/" 
table_num = 0 
file_name_one = "table" + str(table_num) + "epoch" + "0.txt" 
file_name_two = "table" + str(table_num) + "epoch" + "1.txt" 

list_one = [] 
file_path = path + file_name_two 
file = open(file_path, "r") 
lines = file.readlines() 
for line in lines: 
    line_seg = line.split(", ") 
    for word_with_value in line_seg: 
        list_one.append(float(word_with_value)) 
print(len(list_one)) 

min = np.min(list_one) 
max = np.max(list_one) 

plt.hist(list_one, bins = 16) 
plt.vlines(min, 0, 8500, color = 'red', linestyles = 'dashed') 
plt.vlines(max, 0, 8500, color = 'red', linestyles = 'dashed') 
plt.title("Table {} min: {} max: {}".format(table_num, min, max)) 
plt.savefig("hist.png") 
