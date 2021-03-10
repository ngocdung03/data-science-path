# import the reader function from the csv module
from csv import reader

# use the python built-in function open()
# to open the children.csv file
opened_file = open('artworks.csv')

# use csv.reader() to parse the data from
# the opened file
read_file = reader(opened_file)

# use list() to convert the read file
# into a list of lists format
moma = list(read_file)

# remove the first row of the data, which
# contains the column names
moma = moma[1:]

# Write your code here
#Data analysis basics
#artworks_clean.csv

#Create a frequency table for the values in the Gender (row index 5) column.
gender_freq = {}
for row in moma:
    gender = row[5]
    if gender not in gender_freq:
        gender_freq[gender] = 1
    else:
        gender_freq[gender] += 1

#Loop over each key-value pair in the dictionary. Display a line of output in the format shown above summarizing each pair.
for gender, frequency in gender_freq.items():
    output = "There are {f:,.2f} artworks by {g} artists".format(g=gender, f=frequency)
    print(output)