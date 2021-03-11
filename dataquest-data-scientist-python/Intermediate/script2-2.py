# Date and time
# Importing file
from csv import reader
opened_file = open('potus_visitors_2015.csv')
read_file = reader(opened_file)
potus = list(read_file)
potus = potus[1:
]

# Convert date string into date time objects
# date_string = '12/18/15 16:30'
# date,time = date_string.split()

# hr, mn = time.split(':')
# mnth, day, yr = date.split('/')

# hr = int(hr)
# mn = int(mn)
# mnth = int(mnth)
# day = int(day)
# yr = int(yr)

# date_dt = dt.datetime(yr, mnth, day, hr, mn)

# Convert all of the date values in the appt_start_date column to datetime objects:
#This date value indicates clearly that the format is month/day/year, and additionally confirms that the time is in 24-hour format.
print(potus[-1][2])

date_format = "%m/%d/%y %H:%M"
for row in potus:
    start_date = row[2]
    start_date = dt.datetime.strptime(start_date, date_format)
    row[2] = start_date

# Create a formatted frequency table for visitor per month
visitors_per_month = {}
for row in potus:
    date = row[2]
    date = date.strftime("%B, %Y")
    if date not in visitors_per_month:
        visitors_per_month[date] = 1
    else:
        visitors_per_month[date] += 1

# convert app_start_date column to time objects
appt_times = []
for row in potus:
    date = row[2]
    time = date.time
    appt_times.append(time)

# Because we have already converted the app_start_date column to datetime objects, it's easy for us to convert them to time objects.
appt_times = []
for row in potus:
    date = row[2]
    time = date.time()
    appt_times.append(time)
    
appt_times

# calculate the length of a meeting:
for row in potus:
    end_date = row[3]
    end_date = dt.datetime.strptime(end_date, "%m/%d/%y %H:%M")
    row[3] = end_date
    
appt_lengths = {}
for row in potus:
    start = row[2]
    end = row[3]
    length = end - start
    if length not in appt_lengths:
        appt_lengths[length] = 1
    else:
        appt_lengths[length] += 1

min_length = min(appt_lengths)
max_length = max(appt_lengths)