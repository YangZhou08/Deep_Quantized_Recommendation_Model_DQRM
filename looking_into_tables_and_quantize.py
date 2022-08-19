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

log_file_name = "documenting_dist.txt" 

def finding_scale_and_params(table_num, 
                             num_bits, 
                             min, 
                             max): 
    n = 2 ** (num_bits - 1) - 1 
    return np.clip(np.max(min, max), min = 1e-8) / n 

n_l = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572] 
thr_r = [0.2, 0.4, 0.6, 0.8, 1.0] 
colors = ['red', 'green', 'yellow', 'purple', 'black', 'navy'] 

list_one = [] 
chicanes = [] 
for table_num in range(0, 26): 
    file_name = "table" + str(table_num) + "epoch" + "1.txt" 

    file_path = path + file_name 
    file = open(file_path, "r") 
    lines = file.readlines() 
    for line in lines: 
        line_seg = line.split(", ") 
        for word_with_value in line_seg: 
            list_one.append(float(word_with_value)) 
    print("table", table_num) 
    print(len(list_one)) 
    '''
    y, x, _ = plt.hist(list_one, bins = 1000) 
    y_max = np.max(y) 
    chicanes.append(np.sqrt(1/n_l[table_num])) 
    for ratio in thr_r: 
        chicanes.append(np.sqrt(1/n_l[table_num]) * (1 + ratio)) 

    for i, l in enumerate(chicanes): 
        plt.vlines(-l, ymin = 0, ymax = y_max, color = colors[i]) 
        plt.vlines(l, ymin = 0, ymax = y_max, color = colors[i]) 
    if len(list_one) > 1e6: 
        plt.xlim(-0.1, 0.1) 
    
    plt.title("table {}".format(table_num)) 

    logger_path = path + log_file_name 
    logger = open(logger_path, "a") 
    logger.write("table {}\n".format(table_num)) 
    logger.write(str(x)) 
    logger.write("\n") 
    logger.write(str(y)) 
    logger.write("\n") 
    logger.close() 

    list_one = [] 
    chicanes = [] 
    plt.savefig("hist" + str(table_num) + "_.png") 
    plt.clf() 
    ''' 
    print("min: {}, max: {}, mean: {}, standard deviation: {}".format(np.min(list_one), np.max(list_one), np.mean(list_one), np.std(list_one))) 
    print() 
