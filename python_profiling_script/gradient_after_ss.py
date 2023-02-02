import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font) 

sections_performed = ["Embedding table gradient", "MLP layer gradient"] 
category_colors = ["#E6B9B8", "#B9CDE5"] 
y_pos = [0, 0.25] 
t_settings = ["Kaggle", "Terabyte"] 

percentage_performance = [[0.05, 0.95], [0.45, 0.55]] 

perc_per_hlp = np.array(percentage_performance) 
perc_per_cum = perc_per_hlp.cumsum(axis = 1) 
print(perc_per_hlp) 
print(perc_per_hlp.cumsum(axis = 1)) 

fig, ax = plt.subplots() 
for i, (section, color) in enumerate(zip(sections_performed, category_colors)): 
    widths = perc_per_hlp[:, i] 
    starts = perc_per_cum[:, i] - widths 
    ax.barh(y_pos, width = widths, left = starts, height = 0.2, label = section, color = color) 

    xcenters = starts + widths / 2 

    text_color = "black" 

    for y, (x, c) in enumerate(zip(xcenters, [str(int(i * 100)) + "%" for i in widths])): 
        if widths[y] >= 0.03: 
            ax.text(x, y_pos[y], str(c), ha='center', va='center',
                    color=text_color) 

ax.invert_yaxis() 
ax.set_ylim(-0.15, 0.4) 
ax.set_yticks(y_pos) 
ax.set_yticklabels(t_settings) 
ax.set_xlim(0, 1) 
ticks = [i for i in np.arange(0, 1, 0.1)] 
ticks.append(1.0) 
ax.set_xticks(ticks) 
ticks_x = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"] 
ax.set_xticklabels(ticks_x) 

ax.legend(ncol = 2, loc = 'lower left', fontsize = 'small') 
ax.xaxis.grid(True, color = 'grey') 
plt.show() 
