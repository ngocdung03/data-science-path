## Frequency distribution
import pandas as pd

pd.options.display.max_rows = 200
pd.options.display.max_columns = 50
wnba = pd.read_csv('wnba.csv')
wnba.shape
print(wnba)

# Frequency table
wnba = pd.read_csv('wnba.csv')
freq_distro_pos = wnba['Pos'].value_counts()
freq_distro_height = wnba['Height'].value_counts()

# Freq table, sort by index
wnba = pd.read_csv('wnba.csv')
age_ascending = wnba['Age'].value_counts().sort_index()
age_descending = wnba['Age'].value_counts().sort_index(ascending=False)

# Freq table sort manually
def make_pts_ordinal(row):
    if row['PTS'] <= 20:
        return 'very few points'
    if (20 < row['PTS'] <=  80):
        return 'few points'
    if (80 < row['PTS'] <=  150):
        return 'many, but below average'
    if (150 < row['PTS'] <= 300):
        return 'average number of points'
    if (300 < row['PTS'] <=  450):
        return 'more than average'
    else:
        return 'much more than average'
    
wnba['PTS_ordinal_scale'] = wnba.apply(make_pts_ordinal, axis = 1)
pts_ordinal_desc = wnba['PTS_ordinal_scale'].value_counts().iloc[[4, 3, 0, 2, 1, 5]]

# Find percentile
wnba = pd.read_csv('wnba.csv')
from scipy.stats import percentileofscore
a = 17
percentileofscore(wnba['Games Played'], score = a, kind='weak')
percentile_rank_half_less = 16.083916083916083
percentage_half_more = 100 - percentile_rank_half_less

# Freq table with 10 interval, percentages, descending
grouped_freq_table = (wnba['PTS'].value_counts(bins=10, normalize=True)*100).sort_index(ascending=False)

# Freq table with manual class intervals
wnba = pd.read_csv('wnba.csv')
intervals = pd.interval_range(start=0, end=600, freq=60)
gr_freq_table_10 = pd.Series([0,0,0,0,0,0,0,0,0,0], index = intervals)
for value in wnba['PTS']:
    for interval in intervals:
        if value in interval:
            gr_freq_table_10.loc[interval] +=1
            break

## Visualizing frequency distribution
def make_experience_ordinal(row):
    if row['Experience'] == 0:
        return 'Rookie'
    if (1 <= row['Experience'] <=  3):
        return 'Little experience'
    if (4 <= row['Experience'] <=  5):
        return 'Experienced'
    if (5 <= row['Experience'] <= 10):
        return 'Very experienced'
    if (row['Experience'] >  10):
        return 'Veteran'
    
wnba['Exp_ordinal'] = wnba.apply(make_experience_ordinal, axis = 1)
# Bar plot - horizontal bar plot
freq_table = wnba['Exp_ordinal'].value_counts().iloc[[3, 0, 2, 1, 4]]
freq_table.plot.bar(rot = 45)
freq_table.plot.barh(title = "Number of players in WNBA by level of experience")

# Pie chart
wnba['Exp_ordinal'].value_counts().plot.pie(figsize=(6,6), autopct = '%.2f%%',
                                           title='Percentage of players in WNBA by level of experience')
plt.ylabel('')

# Histogram
wnba['PTS'].plot.hist()  # not wnba['PTS'].value_counts().plot.hist()