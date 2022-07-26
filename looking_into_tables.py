from matplotlib.cbook import ls_mapper
import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 

path = "/rscratch/data/dlrm_criteo/" 
table_num = 0 
file_name_one = "table" + str(table_num) + "epoch" + "0.txt" 
file_name_two = "table" + str(table_num) + "epoch" + "1.txt" 

list_one = [] 
file_path = path + file_name_one 
file = open(file_path, "r") 
lines = file.readlines() 
for line in lines: 
    line_seg = line.split(", ") 
    for word_with_value in line_seg: 
        list_one.append(float(word_with_value)) 
print(len(list_one)) 

plt.hist(list_one) 
plt.show() 
