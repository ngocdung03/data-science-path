## Regex basics
import pandas as pd
hn = pd.read_csv("hacker_news.csv")

# Count how many time 'Python' or 'python' are mentioned in title of the dataset
import re

titles = hn["title"].tolist()  #all the tiltles from the dataset
python_mentions = 0
pattern = '[Pp]ython'
for title in titles:
    if re.search(pattern, title):
        python_mentions += 1

# We should avoid using loops in pandas, and that vectorized methods are often faster and require less code.
pattern = '[Pp]ython'
titles = hn['title']
python_mentions = titles.str.contains(pattern).sum()

# See the titles that contain [Rr]uby
pattern = '[Rr]uby'
ruby_titles = titles[titles.str.contains(pattern)]

# Applying quantifiers
email_bool = titles.str.contains('e-?mail')
email_count = email_bool.sum()
email_titles = titles[email_bool]

# Find single-word tags without special characters
pattern = r'\[\w+\]'
tag_titles = titles[titles.str.contains(pattern)]
tag_count = titles.str.contains(pattern).sum()    #tag_titles.shape[0]

# Extract only single word insided square brackets
pattern = r"\[(\w+)\]"
tag_freq = titles.str.extract(pattern, expand=False).value_counts()

# Check if the regex pattern is as expected and use negative character classes
def first_10_matches(pattern):
    """
    Return the first 10 story titles that match
    the provided regular expression
    """
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10
first_10_matches('[Jj]ava[^Ss]')   # The regex shouldn't match where 'Java' is followed by the letter 'S' or 's'.
java_titles = titles[titles.str.contains(r'[Jj]ava[^Ss]')]

# The negative set had the side-effect of removing any titles where Java occurs at the end of the string. This is because the negative set [^Ss] must match one character. Instances at the end of a string aren't followed by any characters, so there is no match.
pattern = r'\b[Jj]ava\b'
first_10_matches(pattern)
java_titles = titles[titles.str.contains(pattern)]

# Number of times that a tag occurs at the start/end of a title
beginning_count = titles.str.contains(r'^\[\w+\]').sum()
ending_count = titles.str.contains(r'\[\w+\]$').sum()

# Find if email is mentioned in titles
import re

email_tests = pd.Series(['email', 'Email', 'e Mail', 'e mail', 'E-mail',
              'e-mail', 'eMail', 'E-Mail', 'EMAIL', 'emails', 'Emails',
              'E-Mails'])

#not r'e.*mail'
#r'e.?mail'  'Source Mailbox'

pattern = r'\be.?mails?\b'       #Solution: r"\be[\-\s]?mails?\b"
email_tests.str.contains(pattern,flags=re.I)
email_mentions = titles.str.contains(pattern,flags=re.I).sum()

## Advanced regular expressions
sql_counts = titles.str.contains(r'sql', flags=re.I).sum()

# Extract the mentions of different SQL flavors into a new column and clean those duplicates by making them all lowercase
hn_sql = hn[hn['title'].str.contains(r"\w+SQL", flags=re.I)].copy()
hn_sql['flavor'] = hn['title'].str.extract(r"(\w+SQL)", flags=re.I)
hn_sql['flavor'] = hn_sql['flavor'].str.lower()
sql_pivot = hn_sql.pivot_table(index='flavor', values='num_comments')

# Capture the version number after the word "Python," and then build a frequency table
# Match Python or python, followed by a space, followed by one or more digit characters or periods.
pattern = r'[Pp]ython ([\d.]+)'        #r'[Pp]ython ?([\d.]+)' not counted?
py_versions = titles.str.extract(pattern, expand=False)
py_versions_freq = dict(py_versions.value_counts())

# Use a negative set to prevent matches for the + character and the . character.
pattern = r"\b[Cc]\b[^.+]"
first_ten = first_10_matches(pattern)

