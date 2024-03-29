# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font) 
 
# # Data
# r = [0,1,2,3,4]
# raw_data = {'greenBars': [20, 1.5, 7, 10, 5], 'orangeBars': [5, 15, 5, 10, 15],'blueBars': [2, 15, 18, 5, 10]}
# df = pd.DataFrame(raw_data)
 
# # From raw value to percentage
# totals = [i+j+k for i,j,k in zip(df['greenBars'], df['orangeBars'], df['blueBars'])]
# greenBars = [i / j * 100 for i,j in zip(df['greenBars'], totals)]
# orangeBars = [i / j * 100 for i,j in zip(df['orangeBars'], totals)]
# blueBars = [i / j * 100 for i,j in zip(df['blueBars'], totals)]
 
# # plot
# barWidth = 0.55 
# names = ('A','B','C','D','E')
# # Create green Bars
# plt.barh(r, greenBars, color='#b5ffb9', edgecolor='white', height=barWidth) 
# # Create orange Bars
# plt.barh(r, orangeBars, left=greenBars, color='#f9bc86', edgecolor='white', height=barWidth) 
# # Create blue Bars
# plt.barh(r, blueBars, left=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', height=barWidth) 
 
# # Custom x axis
# plt.xticks(r, names)
# plt.xlabel("group")
 
# # Show graphic
# plt.show() 

# sections_performed = ["gradient communication", "single-process backward", "DLRM forward", "weight step", "others"] 
sections_performed = ["DQRM forward", "single node backward", "sparsed gradient communication", "weight update", "others"] 
# category_colors = plt.get_cmap('RdYlGn')(
#         np.linspace(0.15, 0.85, len(sections_performed))) 
# category_colors = ["tab:blue", "darkorange", "darkgrey", "gold", "tab:green"] 
category_colors = ["darkgrey", "gold", "tab:blue", "darkorange", "tab:green"] 
y_pos = [0, 0.5] 
devices = ["CPU", "GPU"] 
# percentage_performance = [[0.2804, 0.0283, 0.0524, 0.5914], [0.5751, 0.0826, 0.1259, 0.0536]] 
percentage_performance = [[0.0524, 0.0283, 0.2804, 0.5914], [0.1261, 0.0817, 0.5758, 0.0529]] 
percentage_performance[0].append(1.0 - np.sum(percentage_performance[0])) 
percentage_performance[1].append(1.0 - np.sum(percentage_performance[1])) 

perc_per_hlp = np.array(percentage_performance) 
perc_per_cum = perc_per_hlp.cumsum(axis = 1) 
print(perc_per_hlp) 
print(perc_per_hlp.cumsum(axis = 1)) 

fig, ax = plt.subplots() 
for i, (section, color) in enumerate(zip(sections_performed, category_colors)): 
    widths = perc_per_hlp[:, i] 
    starts = perc_per_cum[:, i] - widths 
    ax.barh(y_pos, width = widths, left = starts, height = 0.3, label = section, color = color) 

    xcenters = starts + widths / 2

    # r, g, b, _ = color
    # text_color = 'white' if r * g * b < 0.5 else 'darkgrey' 
    text_color = "black" 

    for y, (x, c) in enumerate(zip(xcenters, ['%.1f'%(i * 100) + "%" for i in widths])): 
        if widths[y] >= 0.03: 
            ax.text(x, y_pos[y], str(c), ha='center', va='center',
                    color=text_color) 

ax.invert_yaxis() 
ax.set_ylim(-0.3, 0.8) 
ax.set_yticks(y_pos) 
ax.set_yticklabels(devices) 
ax.set_xlim(0, 1) 
ticks = [i for i in np.arange(0, 1, 0.1)] 
ticks.append(1.0) 
ax.set_xticks(ticks) 
ticks_x = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"] 
ax.set_xticklabels(ticks_x) 

ax.legend(ncol = len(sections_performed) - 1, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small') 
ax.xaxis.grid(True, color = 'grey') 

plt.show() 
