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
