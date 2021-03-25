## Sampling
# Data: https://www.kaggle.com/jinxbe/wnba-player-stats-2017
# Measure sampling error
import pandas as pd
wnba = pd.read_csv('wnba.csv')
wnba.head()
wnba.tail()
wnba.shape
parameter = wnba['Games Played'].max()
sample = wnba.sample(30, random_state=1)
statistic = sample['Games Played'].max()
sampling_error = parameter - statistic

# visualize the discrepancy between a parameter and its corresponding statistics in the case of simple random sampling.
import pandas as pd
import matplotlib.pyplot as plt

wnba = pd.read_csv('wnba.csv')
pop_mean = wnba['PTS'].mean()
list = []
for i in range(0, 100):
    sample = wnba['PTS'].sample(10, random_state=i)
    mean = sample.mean()
    list.append(mean)
plt.scatter(range(1,101), list)
plt.axhline(pop_mean)
plt.show()

# Stratified sampling
wnba['points_per_game'] = wnba['PTS']/wnba['Games Played']
positions = wnba['Pos'].unique()

points_per_position = {}
for position in positions:
    sample = wnba[wnba['Pos']==position].sample(10,random_state = 0)
    mean = sample['points_per_game'].mean()
    points_per_position[position] = mean
    
position_most_points = max(points_per_position, key=points_per_position.get)
### Solution ###
stratum_G = wnba[wnba.Pos == 'G']
stratum_F = wnba[wnba.Pos == 'F']
stratum_C = wnba[wnba.Pos == 'C']
stratum_GF = wnba[wnba.Pos == 'G/F']
stratum_FC = wnba[wnba.Pos == 'F/C']

points_per_position = {}
for stratum, position in [(stratum_G, 'G'), (stratum_F, 'F'), (stratum_C, 'C'),
                (stratum_GF, 'G/F'), (stratum_FC, 'F/C')]:
    
    sample = stratum['Pts_per_game'].sample(10, random_state = 0) # simple random sampling on each stratum
    points_per_position[position] = sample.mean()
    
position_most_points = max(points_per_position, key = points_per_position.get)
######

# Stratified sampling based on the proportions of games played
stratum_1 = wnba[wnba['Games Played']<=12]
stratum_2 = wnba[(wnba['Games Played']>12) & (wnba['Games Played']<=22)] #must include ()
stratum_3 = wnba[wnba['Games Played'] >22]
mean_point = []
for i in range(100):
    sample1 = stratum_1.sample(1, random_state=i)
    sample2 = stratum_2.sample(2, random_state=i)
    sample3 = stratum_3.sample(7, random_state=i)
    sample = pd.concat([sample1, sample2, sample3])   #must include []
    mean_point.append(sample['PTS'].mean())
  
plt.scatter(range(1,101), mean_point)
plt.axhline(wnba.PTS.mean())
plt.show() 

# It makes more sense to stratify the data by number of minutes played, rather than by number of games played
wnba['MIN'].value_counts(bins = 3, normalize = True)

# Simulate a cluster sampling
team_clusters = pd.Series(wnba['Team'].unique()).sample(4, random_state=0)
sample = pd.DataFrame()
for team in team_clusters:
    sample_i = wnba[wnba['Team']==team]
    sample = pd.concat([sample, sample_i])      #solution: sample = sample.append(sample_i)
sampling_error_height = wnba['Height'].mean() - sample['Height'].mean()
sampling_error_age = wnba['Age'].mean() - sample['Age'].mean()
sampling_error_BMI = wnba['BMI'].mean() - sample['BMI'].mean()
sampling_error_points = wnba['PTS'].mean() - sample['PTS'].mean()

## Variables in statistics
variables = {'Name': 'qualitative', 'Team': 'qualitative', 'Pos': 'qualitative', 'Height': 'quantitative', 'BMI': 'quantitative',
             'Birth_Place': 'qualitative', 'Birthdate': 'quantitative', 'Age': 'quantitative', 'College': 'qualitative', 'Experience': 'quantitative',
             'Games Played': 'quantitative', 'MIN': 'quantitative', 'FGM': 'quantitative', 'FGA': 'quantitative',
             '3PA': 'quantitative', 'FTM': 'quantitative', 'FTA': 'quantitative', 'FT%': 'quantitative', 'OREB': 'quantitative', 'DREB': 'quantitative',
             'REB': 'quantitative', 'AST': 'quantitative', 'PTS': 'quantitative'}
# Nominal variables
#wnba.columns
nominal_scale = sorted(['Name', 'Team', 'Pos', 'Birth_Place', 'College'])  # sort() changes the list directly and doesn't return any value, while sorted() doesn't change the list and returns the sorted list.
interval = sorted(['Birthdate', 'Weight_deviation'])
ratio = sorted(['Height', 'Weight', 'BMI', 'Age', 'Experience', 'Games Played', 'MIN', 'FGM', 'FGA', 'FG%', '15:00', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB',
       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS', 'DD2', 'TD3'])
ratio_interval_only = {'Height':'continuous', 'Weight': 'continuous', 'BMI': 'continuous', 'Age': 'continuous', 'Games Played': 'discrete', 'MIN': 'continuous', 'FGM': 'discrete', 'FGA': 'discrete', 'FG%': 'continuous', '3PA': 'discrete', '3P%': 'continuous', 'FTM': 'discrete', 'FTA': 'discrete', 'FT%': 'continuous',
                       'OREB': 'discrete', 'DREB': 'discrete', 'REB': 'discrete', 'AST': 'discrete', 'STL': 'discrete', 'BLK': 'discrete', 'TO': 'discrete',
                       'PTS': 'discrete', 'DD2': 'discrete', 'TD3': 'discrete', 'Weight_deviation': 'continuous'}
                       