## Working with strings in Pandas
happiness2015 = pd.read_csv("World_Happiness_2015.csv")
world_dev = pd.read_csv("World_dev.csv")
col_renaming = {'SourceOfMostRecentIncomeAndExpenditureData': 'IESurvey'}
merged = pd.merge(happiness2015, world_dev, how='left', left_on='Country', right_on='ShortName')
merged = merged.rename(col_renaming, axis=1)   #!!remember to assign back to the orginal dataframe

# Extract the unit of currency without the leading nationality
def extract_last_word(element):
    element = str(element)
    element = element.split()
    return element[-1]    #return last word
merged['Currency Apply'] = merged['CurrencyUnit'].apply(extract_last_word)
merged['Currency Apply'].head(5)

# Chaining string methods: split the CurrencyUnit column into a list of words and select just the last word
merged['Currency Vectorized'] = merged['CurrencyUnit'].str.split().str.get(-1)
merged['Currency Vectorized'].head(5)

# Different from apply(function), vectorized methods do not treat NaN as a string then returned a length of 3 for each NaN value
lengths = merged['CurrencyUnit'].str.len()
value_counts = lengths.value_counts(dropna=False) #If value_counts contains NaNs, it means the Series.str.len() method excluded them and didn't treat them as strings.

# Practice with regex
pattern = r"[Nn]ational accounts"   # either "national accounts" or "National accounts" should produce a match.
national_accounts = merged['SpecialNotes'].str.contains(pattern, na=False)  # if don't include na=False, error will arise with missing values
merged_national_accounts = merged[national_accounts]
merged_national_accounts.head(5)

pattern =r"([1-2][0-9]{3})"
years = merged['SpecialNotes'].str.extract(pattern)
# return the results as a dataframe
years = merged['SpecialNotes'].str.extract(pattern, expand=True)

# .str.extractall() and named capturing group
pattern = r"(?P<Years>[1-2][0-9]{3})"
years = merged['IESurvey'].str.extractall(pattern)
value_counts = years['Years'].value_counts()

# Deal with "2018/29" pattern
# extract just the years from the IESurvey column. Then, we'll reformat the second year so that it contains all four digits of the year, not just the last tw
pattern = r"(?P<First_Year>[1-2][0-9]{3})/?(?P<Second_Year>[0-9]{2})?"  # we didn't enclose /? in parantheses so that the resulting dataframe will only contain a First_Year and Second_Year column.
years = merged['IESurvey'].str.extractall(pattern)
first_two_year = years['First_Year'].str[0:2]   #vectorized slicing to extract the first two numbers 
years['Second_Year'] = first_two_year + years['Second_Year']

# Practice: plotting for Happiness Score by Income group
pattern = r"( income)?(:)?"
merged['IncomeGroup'] = merged['IncomeGroup'].str.replace(pattern,"").str.upper().str.strip()
                                             
pv_incomes = merged.pivot_table(index='IncomeGroup', values='Happiness Score')
pv_incomes.plot(kind='bar', rot=30, ylim=(0,10))
plt.show()
