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

# Keep excluding matches that are followed by . or +, but still match cases where "C" falls at the end of the sentence. Exclude matches that have the word 'Series' immediately preceding them.
pattern = r'(?<!Series\s)\b[Cc]\b(?![.+]|\.$)'       #?why |\.$
c_mentions = titles.str.contains(pattern).sum()

# Match cases of repeated words
# define a word as a series of one or more word characters preceded and followed by a boundary anchor
# define repeated words as the same word repeated twice, separated by a single whitespace character.
pattern = r'\b(\w+)\s\1\b'                  #wrong r'(\b\w+\b)\s\1'
repeated_words = titles[titles.str.contains(pattern)]

# Replace substring, ignore case 
email_variations = pd.Series(['email', 'Email', 'e Mail',
                        'e mail', 'E-mail', 'e-mail',
                        'eMail', 'E-Mail', 'EMAIL'])
#wrong r'\be[-?\s?]mail\b'
pattern = r'\be[-\s]?mail'     # ?why not r'\be[-\s]?mail\b'      
email_uniform = email_variations.str.replace(pattern, "email", flags=re.I)
titles_clean = titles.str.replace(pattern, "email", flags=re.I)

# Extract URL
test_urls = pd.Series([
 'https://www.amazon.com/Technology-Ventures-Enterprise-Thomas-Byers/dp/0073523429',
 'http://www.interactivedynamicvideo.com/',
 'http://www.nytimes.com/2007/11/07/movies/07stein.html?_r=0',
 'http://evonomics.com/advertising-cannot-maintain-internet-heres-solution/',
 'HTTPS://github.com/keppel/pinn',
 'Http://phys.org/news/2015-09-scale-solar-youve.html',
 'https://iot.seeed.cc',
 'http://www.bfilipek.com/2016/04/custom-deleters-for-c-smart-pointers.html',
 'http://beta.crowdfireapp.com/?beta=agnipath',
 'https://www.valid.ly?param',
 'http://css-cursor.techstream.org'
])
pattern = r'(?<=://)([\w\.\-]*)'       #wrong r'(?<=://)(.*)(?=[/?\b])'
test_urls_clean = test_urls.str.extract(pattern, expand=False)
domains = hn['url'].str.extract(pattern, expand=False)
top_domains = domains.value_counts().head(5)
### Solution ###
pattern = r"https?://([\w\-\.]+)"

test_urls_clean = test_urls.str.extract(pattern, flags=re.I, expand=False)
domains = hn['url'].str.extract(pattern, flags=re.I, expand=False)
top_domains = domains.value_counts().head(5)
######

# Multiple capture groups
# `test_urls` is available from the previous screen
# wrong r'(.+)://([\w\.\-]+)[/\b](.*)' 
# why wrong?  pattern = r'(.+)://([\w\.\-]+)/?(.*)'              
pattern = r'(https?)://([\w\.\-]+)/?(.*)'
test_url_parts = test_urls.str.extract(pattern, flags = re.I)
url_parts = hn['url'].str.extract(pattern, flags = re.I)

# Named capture groups.
pattern = r"(?P<protocol>https?)://(?P<domain>[\w\.\-]+)/?(?P<path>.*)"
url_parts = hn['url'].str.extract(pattern, flags=re.I)
