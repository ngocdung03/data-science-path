## Storytelling data visualization
import pandas as pd
import matplotlib.pyplot as plt

death_toll = pd.read_csv("covid_avg_deaths.csv")

# Drawing a grid chart
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(6,8))
(6,8))
axes = [ax1, ax2, ax3, ax4]
for ax in axes:
    ax.plot(death_toll['Month'], death_toll['New_deaths'])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(bottom=False, left=False)
    for loc in ['left','right','top','bottom']:
        ax.spines[loc].set_visible(False)

plt.show()