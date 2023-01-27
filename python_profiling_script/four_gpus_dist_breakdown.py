# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
 
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

sections_performed = ["gradient communication", "single-process backward", "DLRM forward", "weight step", "others"] 
# category_colors = plt.get_cmap('RdYlGn')(
#         np.linspace(0.15, 0.85, len(sections_performed))) 
category_colors = ["tab:blue", "darkorange", "darkgrey", "gold", "tab:green"] 
devices = ["CPU", "GPU"] 
percentage_performance = [[0.2804, 0.0283, 0.0524, 0.5914], [0.5751, 0.0826, 0.1259, 0.0536]] 
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
    ax.barh(devices, width = widths, left = starts, height = 0.5, label = section, color = color) 

    xcenters = starts + widths / 2

    # r, g, b, _ = color
    # text_color = 'white' if r * g * b < 0.5 else 'darkgrey' 
    text_color = "white" 

    for y, (x, c) in enumerate(zip(xcenters, ['%.1f'%(i * 100) + "%" for i in widths])): 
        ax.text(x, y, str(c), ha='center', va='center',
                color=text_color) 

ax.invert_yaxis() 
ax.set_xlim(0, 1) 
ticks = [i for i in np.arange(0, 1, 0.1)] 
ticks.append(1.0) 
ax.set_xticks(ticks) 

ax.legend(ncol=len(sections_performed), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small') 

plt.show() 
