## Design for an audience
import pandas as pd
import matplotlib.pyplot as plt
top20_deathtoll = pd.read_csv("top20_deathtoll.csv")
#top20_deathtoll.plot.barh(x='Country_Other', y = 'Total_Deaths')
plt.barh(top20_deathtoll['Country_Other'],
         top20_deathtoll['Total_Deaths'])
plt.show()

# Apply OO to create horizontal bar plot
fig, ax = plt.subplots(figsize=(4.5, 6))  #change the proportions as width of 4.5inches and height of 6 inches
ax.barh(top20_deathtoll['Country_Other'],
        top20_deathtoll['Total_Deaths'],
        height = 0.45, color = '#b00b1e')               # remove redundant data-ink

for location in ['left', 'right', 'top', 'bottom']:    # Remove 4 axes/spines
    ax.spines[location].set_visible(False)

ax.set_xticks([0, 150000, 300000])   # remove redundant data-ink
ax.set_xticklabels(['0', '150,000', '300,000']) 
ax.xaxis.tick_top()                 #Move the tick labels to the top
ax.tick_params(top=False, left=False)# Remove top and left ticks
ax.tick_params(axis='x', colors='grey')
# Add the title
ax.text(x=-80000, y=23.5, s='The Death Toll Worldwide Is 1.5M+',
        size=17, weight='bold')
# Add the subtitle
ax.text(x=-80000, y=22.5, s='Top 20 countries by death toll (December 2020)',
        size=12)
# Left-align the y-tick labels
ax.set_yticklabels([])
country_names = top20_deathtoll['Country_Other']
for i, country in zip(range(0, 21), country_names):
    ax.text(x=-80000, y=i-0.15, s=country)
# Add a vertical line
ax.axvline(x=150000, ymin=0.045, c='grey', alpha=0.5)

plt.show()

