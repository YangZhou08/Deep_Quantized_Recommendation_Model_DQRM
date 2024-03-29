import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
single_node_cpu_tt = [19.66, 19.22, 19.25, 19.23, 19.56, 19.31, 19.59, 19.44, 19.27, 19.42] 
single_node_cpu_meantt = np.mean(single_node_cpu_tt) 
multiple4_node_cpu_tt = [163.07, 165.88, 164.78, 163.48, 173.90, 170.52, 171.87, 170.03, 167.47, 168.33, 169.45, 167.88, 170.88, 168.98, 174.75, 173.33, 171.97, 177.93, 173.03, 171.54, 171.66, 170.17, 167.40, 169.88, 167.98, 168.73, 177.08, 174.68, 175.14, 173.61, 170.39, 172.46, 169.43, 170.57, 173.24] 
multiple4_node_cpu_meantt = np.mean(multiple4_node_cpu_tt) 
cpu_tt = [single_node_cpu_meantt, multiple4_node_cpu_meantt] 

print(cpu_tt) 

gpu_tt = [5, 20] 

# Set position of bar on X axis
br1 = np.arange(len(cpu_tt)) 
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
fig2, ax = plt.subplots(1, 2) 
ax[0].bar(["single node", "four nodes DP"], cpu_tt, color ='tab:blue',
        edgecolor ='grey') 
ax[1].bar(["single GPU", "four GPUs DP"], gpu_tt, color = 'tab:green', 
        edgecolor = 'grey')
'''
plt.bar(br2, gpu_tt, color ='g', width = barWidth,
        edgecolor ='grey', label ='Single node / GPU') 
''' 
# plt.bar(br3, CSE, color ='b', width = barWidth,
#         edgecolor ='grey', label ='CSE')
 
# Adding Xticks
# ax[0].set_xlabel('Different training settings', fontweight ='bold', fontsize = 15)
ax[0].set_ylabel('Training iteration time in ms/iter (averaged from 10 samples)', fontweight ='bold', fontsize = 15)
ax[0].set_title("Terabyte Dataset DDP Single process versus Multiple processes (CPU)\nnp = 4, ppn = 2", fontweight = 'bold', fontsize = 15) 
# ax[1].set_xlabel('Different training settings', fontweight ='bold', fontsize = 15)
# ax[1].set_ylabel('Training iteration time (averaged from 10 samples)', fontweight ='bold', fontsize = 15)
ax[1].set_ylabel('Training iteration time in ms/iter (averaged from 10 samples)', fontweight ='bold', fontsize = 15)
ax[1].set_title("Kaggle Dataset DDP Single GPU versus Multiple GPUs (GPU)\n4 GPUs", fontweight = 'bold', fontsize = 15) 
# plt.xticks([r + barWidth for r in range(len(cpu_tt))],
#         ['GPU platform', 'CPU platform'])
 
plt.show() 