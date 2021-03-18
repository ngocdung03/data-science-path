## Storytelling data visualization
import pandas as pd
import matplotlib.pyplot as plt

death_toll = pd.read_csv("covid_avg_deaths.csv")

# Drawing a grid chart
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(6,8))
(6,8))
axes = [ax1, ax2, ax3, ax4]
for ax in axes:
    ax.plot(death_toll['Month'], death_toll['New_deaths'], color='#b00b1e', alpha=0.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(bottom=False, left=False)
    for loc in ['left','right','top','bottom']:
        ax.spines[loc].set_visible(False)
# Impose plots of each period - continue effect
ax1.plot(death_toll['Month'][:3], death_toll['New_deaths'][:3],
         color='#b00b1e', linewidth=2.5)
ax2.plot(death_toll['Month'][2:6], death_toll['New_deaths'][2:6],
         color='#b00b1e', linewidth=2.5)
ax3.plot(death_toll['Month'][5:10], death_toll['New_deaths'][5:10],
         color='#b00b1e', linewidth=2.5)
ax4.plot(death_toll['Month'][9:12], death_toll['New_deaths'][9:12],
         color='#b00b1e', linewidth=2.5)
# Adding data label
ax1.text(0.5, -80, '0', alpha=0.5)
ax1.text(3.5, 2000, '1,844', alpha=0.5)
ax1.text(11.5, 2400, '2,247', alpha=0.5)

ax1.text(1.1, -300, 'Jan - Mar', color='#b00b1e',
         weight='bold', rotation=3)
ax2.text(3.7, 800, 'Mar - Jun', color='#b00b1e',
         weight='bold')
ax3.text(7.1, 500, 'Jun - Oct', color='#b00b1e',
         weight='bold')
ax4.text(10.5, 600, 'Oct - Dec', color='#b00b1e',
         weight='bold', rotation=45)
# Adding title and subtitle
ax1.text(0.5, 3500, 'The virus kills 851 people each day', size=14, weight='bold')
ax1.text(0.5, 3150, 'Average number of daily deaths per month in the US', size=12)
# Progress bars
for ax, xmax, death in zip(axes, xmax_vals, deaths):
    ax.axhline(y=1600, xmin=0.5, xmax=0.8,
               linewidth=6, color='#b00b1e',
               alpha=0.1)
    ax.axhline(y=1600, xmin=0.5, xmax=xmax,
               linewidth=6, color='#b00b1e')
    ax.text(7.5, 1850, format(death, ','), color='#b00b1e', weight='bold')
plt.show()

plt.show()