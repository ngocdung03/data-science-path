## Matplotlib styles: FiveThirtyEight case study
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('ggplot')
plt.plot([2, 4, 6], [10, 15, 5])
plt.show()

style.use('default')
plt.plot([2, 4, 6], [10, 15, 5])
plt.show()

colors = ['red', 'orange', 'purple', 'green']
plt.bar(['a', 'b', 'c', 'd'], [2, 4, 7, 3],
        color=colors)
plt.show()
# Example of red, white wine
import pandas as pd
red_wine = pd.read_csv('winequality-red.csv', sep=';')
red_corr = red_wine.corr()['quality'][:-1]  #see the correlation values between quality and the other columns 

white_wine = pd.read_csv('winequality-white.csv', sep=';')
white_corr = white_wine.corr()['quality'][:-1]

# Plot
style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(9, 5))
# ax.barh(white_corr.index, white_corr, left=2, height=0.5)
# ax.barh(red_corr.index, red_corr, left=-0.1, height=0.5)
# Remove the grid, x- and y- tick labels
ax.grid(b=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
# Add y-tick labels using Axes.text() and specify the coordinates manually
x_coords = {'Alcohol': 0.82, 'Sulphates': 0.77, 'pH': 0.91,
            'Density': 0.80, 'Total Sulfur Dioxide': 0.59,
            'Free Sulfur Dioxide': 0.6, 'Chlorides': 0.77,
            'Residual Sugar': 0.67, 'Citric Acid': 0.76,
            'Volatile Acidity': 0.67, 'Fixed Acidity': 0.71}
y_coord = 9.8
for y_label, x_coord in x_coords.items():
    ax.text(x_coord, y_coord, y_label)
    y_coord -= 1
# Add two vertical lines to visually separate the labels from the bar plots
ax.axvline(0.5, ymin=0.1, ymax=0.9, color='grey', alpha=0.1, linewidth=0.1)
ax.axvline(1.45, ymin=0.1, ymax=0.9, color='grey', alpha=0.1, linewidth=0.1)
# Add x-tick labels and a horizontal line under each bar plot
ax.axhline(-1, color='grey', linewidth=1, alpha=0.5,
          xmin=0.01, xmax=0.32)
ax.text(-0.7, -1.7, '-0.5'+ ' '*31 + '+0.5',
        color='grey', alpha=0.5)  #adding 31 space chars between
ax.axhline(-1, color='grey', linewidth=1, alpha=0.5,
           xmin=0.67, xmax=0.98)
ax.text(1.43, -1.7, '-0.5'+ ' '*31 + '+0.5',
        color='grey', alpha=0.5)
# Add a horizontal line and a title above each plot
ax.axhline(11, color='grey', linewidth=1, alpha=0.5, xmin=0.01, xmax=0.32)
ax.text(-0.33, 11.2, 'RED WINE', weight='bold')

ax.axhline(11, color='grey', linewidth=1, alpha=0.5, xmin=0.67, xmax=0.98)
ax.text(1.75, 11.2, 'WHITE WINE', weight='bold')
# Generating signature bar
ax.text(-0.7, -2.9,
        '©DATAQUEST' + ' '*94 + 'Source: P. Cortez et al.',
        color = '#f0f0f0', backgroundcolor = '#4d4d4d',
        size=12)
# Add title and subtitle
ax.text(-0.7, 13.5, 'Wine Quality Most Strongly Correlated With Alcohol Level', size=17, weight='bold')
ax.text(-0.7, 12.7, 'Correlation values between wine quality and wine properties (alcohol, pH, etc.)')
# Map colors with positive/negative values
positive_white = white_corr >= 0
color_map_white = positive_white.map({True:'#33A1C9',
                                      False:'#ffae42'}
                                    )
ax.barh(white_corr.index, white_corr, color=color_map_white,
         height=0.5, left=2)  # A pandas Series is also an array, which means we can pass color_map_white to the color parameter.

positive_red = red_corr >=0
color_map_red = positive_red.map({True:'#33A1C9',
                                  False: '#ffae42'}
                                )
ax.barh(red_corr.index, red_corr, color=color_map_red,
        height=0.5, left=-0.1)

plt.show()
