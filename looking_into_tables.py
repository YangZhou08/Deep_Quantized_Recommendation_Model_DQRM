import numpy as np 
from matplotlib.cbook import ls_mapper
import matplotlib 
'''
matplotlib.use('Qt5Agg') 
''' 
import matplotlib.pyplot as plt 
import sys 
import argparse 

parser = argparse.ArgumentParser(description = "investigating distribution") 
parser.add_argument("--table-num", type = int, default = 0) 
args = parser.parse_args() 
path = "/rscratch/data/dlrm_criteo/" 
table_num = args.table_num 
file_name_one = "table" + str(table_num) + "epoch" + "0.txt" 
file_name_two = "table" + str(table_num) + "epoch" + "1.txt" 

file_names = [file_name_one, file_name_two] 

fig, axes = plt.subplots(1, 2) 

n_l = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572] 

list_one = [] 
for table_num in range(0, 26): 
    file_name_one = "table" + str(table_num) + "epoch" + "0.txt" 
    file_name_two = "table" + str(table_num) + "epoch" + "1.txt" 

    file_names = [file_name_one, file_name_two] 
    
    for i, file_name in enumerate(file_names): 
        file_path = path + file_name 
        file = open(file_path, "r") 
        lines = file.readlines() 
        for line in lines: 
            line_seg = line.split(", ") 
            for word_with_value in line_seg: 
                list_one.append(float(word_with_value)) 
        print(len(list_one)) 

        min = np.min(list_one) 
        max = np.max(list_one) 

        axes[i].hist(list_one, bins = 100) 
        axes[i].axvline(-np.sqrt(1/n_l[table_num]), color = 'yellow', lw = 2) 
        axes[i].axvline(np.sqrt(1/n_l[table_num]), color = 'yellow', lw = 2) 
        axes[i].axvline(min, color = 'red', lw = 2) 
        axes[i].axvline(max, color = 'red', lw = 2) 
        '''
        axes[i].set_title("Table {}\nmin {:.2f} max {:.2f}".format(table_num, min, max)) 
        ''' 
        axes[i].set_title("Table {}\nmin {:.4f} max {:.4f}".format(table_num, min, max)) 

        list_one = [] 

    print("printing table {} distribution".format(table_num)) 
    plt.savefig("hist" + str(table_num) + ".png") 

    axes[0].clear() 
    axes[1].clear() 

'''
plt.hist(list_one, bins = 100) 
plt.vlines(min, 0, 8500, color = 'red', linestyles = 'dashed') 
plt.vlines(max, 0, 8500, color = 'red', linestyles = 'dashed') 
plt.vlines(min, 0, 1100, color = 'red', linestyles = 'dashed') 
plt.vlines(max, 0, 1100, color = 'red', linestyles = 'dashed') 
plt.title("Table {} min: {} max: {}".format(table_num, min, max)) 
''' 
