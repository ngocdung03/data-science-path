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
wnba['Games Played'].plot.hist(range=(1,32), bins=8, title='The distribution of players by games played')
plt.xlabel('Games played')

## Comparing frequency distributions
rookies = wnba[wnba['Exp_ordinal'] == 'Rookie']
little_xp = wnba[wnba['Exp_ordinal'] == 'Little experience']
experienced = wnba[wnba['Exp_ordinal'] == 'Experienced']
very_xp = wnba[wnba['Exp_ordinal'] == 'Very experienced']
veterans =  wnba[wnba['Exp_ordinal'] == 'Veteran']

rookie_distro = rookies['Pos'].value_counts()
little_xp_distro = little_xp['Pos'].value_counts()
experienced_distro = experienced['Pos'].value_counts()
very_xp_distro = very_xp['Pos'].value_counts()
veteran_distro = veterans['Pos'].value_counts()

# Grouped bar plot
import seaborn as sns
sns.countplot(x = 'Exp_ordinal', hue = 'Pos', data = wnba,
             order=['Rookie', 'Little experience', 'Experienced', 'Very experienced', 'Veteran'],
             hue_order=['C', 'F', 'F/C', 'G', 'G/F'])

# 2nd grouped bar plot
wnba['age_mean_relative'] = wnba['Age'].apply(lambda x: 'old' if x >= 27 else 'young')
wnba['min_mean_relative'] = wnba['MIN'].apply(lambda x: 'average or above' if x >= 497 else
                                           'below average')
sns.countplot(x='age_mean_relative', hue='min_mean_relative', data=wnba)

# Superimposed step histograms
import matplotlib.pyplot as plt
wnba[wnba.Age >= 27]['MIN'].plot.hist(histtype = 'step', label = 'Old', legend = True)
wnba[wnba.Age < 27]['MIN'].plot.hist(histtype = 'step', label = 'Young', legend = True)
# Adding the vertical line for average number
plt.axvline(497, label='Average')
plt.legend()
plt.show()

# Kernel density plot
wnba[wnba.Age >= 27]['MIN'].plot.kde(label = 'Old', legend = True)
wnba[wnba.Age < 27]['MIN'].plot.kde(label = 'Young', legend = True)
plt.axvline(497, label='Average')
plt.legend()  #if not, axvline's legend will not display
plt.show()

# Strip plot
sns.stripplot(x = 'Pos', y = 'Weight', data = wnba, jitter = True)

# Box plot
sns.boxplot(x = 'Pos', y = 'Weight', data = wnba)
plt.show()

# Consider the quartiles of the Games Played variable:
print(wnba['Games Played'].describe())
iqr = 29-22
lower_bound = 22 - 1.5*iqr
upper_bound = 29 + 1.5*iqr
outliers_low = sum(wnba['Games Played']<lower_bound)
outliers_high = sum(wnba['Games Played']>upper_bound)

sns.boxplot(wnba['Games Played'])
plt.show()