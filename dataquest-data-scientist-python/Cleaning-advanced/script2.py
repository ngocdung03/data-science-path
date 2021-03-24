## List Comprehensions and Lambda Functions
world_cup_str = """
[
    {
        "team_1": "France",
        "team_2": "Croatia",
        "game_type": "Final",
        "score" : [4, 2]
    },
    {
        "team_1": "Belgium",
        "team_2": "England",
        "game_type": "3rd/4th Playoff",
        "score" : [2, 0]
    }
]
"""
import json
world_cup_obj = json.loads(world_cup_str)

file = open('hn_2014.json')
hn = json.load(file)        #without s
# Find out how many objects are in the list, and the type of the first object (which will almost always be the type of every object in the list in JSON data)

# Remove the createdAtI key from every story in dataset
def del_key(dict_, key):
    # create a copy so we don't
    # modify the original dict
    modified_dict = dict_.copy()
    del modified_dict[key]
    return modified_dict
hn_clean = []
for dict in hn:
    cleaned_dict = del_key(dict, 'createdAtI')
    hn_clean.append(cleaned_dict)

# Use list comprehension
hn_clean = [del_key(d, 'createdAtI') for d in hn]

# Create a new list containing just the URLs from each story.
urls = [dict['url'] for dict in hn_clean]

# Count how many stories have more than 1,000 points
thousand_points = [dict for dict in hn_clean if dict['points']>1000]
num_thousand_points = len(thousand_points)

# Find the story that has the greatest number of comments
def key_function(dict):
    return dict['numComments']
most_comments = max(hn_clean, key=key_function)

# Sort the hn_clean JSON list by the number of points (dictionary key points) from highest to lowest. Return a list of the five post titles (dictionary key title) that have the most points 
hn_sorted_points = sorted(hn_clean, key=lambda d: d['points'], reverse=True)

top_5_titles = [dict['title'] for dict in hn_sorted_points][:5]

# Create a Pandas dataframe version of the hn_clean JSON list
import pandas as pd
hn_df = pd.DataFrame(hn_clean)

# Check type and length of each list in tags column if consistent
tags = hn_df['tags']
print(tags.dtype)  #object
tags_types = tags.apply(type)
type_counts = tags_types.value_counts(dropna=False)
print(type_counts)
tags_types = tags.apply(len)  #list
type_lengths = tags_types.value_counts(dropna=False)
print(type_lengths)    #most 3, over 2000 4
# Let's use a boolean mask to look at the items where the list has four items
four_tags = tags[tags.apply(len)==4]

# Use ternary operator and lambda function: where the item is a list with length four, return the last item. In all other cases, return None.
cleaned_tags = tags.apply(lambda tag: tag[-1] if len(tag)==4 else None)
hn_df['tags'] = cleaned_tags
