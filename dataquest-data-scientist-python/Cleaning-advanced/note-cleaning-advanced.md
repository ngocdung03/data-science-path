##### Regex basics
- `re` module: contains a number of different functions and classes for working with regular expressions. https://docs.python.org/3/library/re.html#module-re
- re.search(regex pattern, string we want to search that pattern for) 
- *Set []* allows us to specify two or more characters that can match in a single character's position
```py
[msb]end
# the set contains one of m,s or b
# the regex will match the strrings mend, send, and bend
```
- One of the neat things about boolean masks is that you can use the Series.sum() method to sum all the values in the boolean mask
    - Firstly creating a pandas object containing our strings
    ```py
    eg_list = ["Julie's favorite color is green.",
           "Keli's favorite color is Blue.",
           "Craig's favorite colors are blue and red."]

    eg_series = pd.Series(eg_list)
    pattern = "[Bb]lue"
    pattern_contained = eg_series.str.contains(pattern)
    ```
- *Quantifier {}* specify how many of the previous character our pattern requires
    - [Quantifier-ex1.jpg] [Quantifier-ex2.jpg]
- *Character classes* allows us to match certain groups of characters
    - [Character-class.jpg] [Common-character-class.jpg]
- A backslash followed by certain characters represents an escape sequence 
- Recommend using raw strings for every regex you write, rather than remember which sequences are escape sequences and using raw strings selectively: `print(r'hello\b world')` = `print('hello\\b world')`
- When an "r" or "R" prefix is present, a character following a backslash is included in the string without change, and all backslashes are left in the string: https://stackoverflow.com/questions/2241600/python-regex-r-prefix
- *Capture groups ()* allow us to specify one or more groups within our match that we can access separately: `(\[\w+\])`
```py
pattern = r"\[(\w+)\]"
tag_5_matches = tag_5.str.extract(pattern, expand=False)   # expand=False - returns a series
print(tag_5_matches)
# get just the text inside square brackets eg. pdf
```
- It can be helpful to create a function that returns the first few matching strings:
```py
def first_10_matches(pattern):
    """
    Return the first 10 story titles that match
    the provided regular expression
    """
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10
```
- RegExr online tool to learn, build, & test Regular Expressions: https://regexr.com/
- *Negative character classes [^]* are character classes that match every character except a character class
    - [Negative-char-classes.jpg]
- *Word boundary anchor \b* matches the position between a word character and a non-word character, or a word character and the start/end of a string: `pattern_2 = r"\bJava\b"` - Matches somthing that isn't a character.\ 
    - Beginning anchor: `^abc`. ^ character is used both as a beginning anchor and to indicate a negative set, depending on whether the character preceding it is a [ or not.
    - Ending anchor: `abc$` 
- *flags* specify that our regular expression should ignore case.
    - List of all available flags: https://docs.python.org/3/library/re.html#re.A
    - re.IGNORECASE - alias re.I
    ```py
    import re
    email_tests.str.contains(r"email",flags=re.I)
    ```

##### Advanced regular expressions
- *Lookarounds* let us define a character or sequence of characters that either must or must not come before or after our regex match.
    - [Lookaround.jpg]
    - can include any other regular expression component, eg. cases that are followed by exactly five characters: `run_test_cases(r"Green(?=.{5})")`
-  We'll use the re module rather than pandas since it tells us the exact text that matches, which will help us understand how the lookaround is working
- Sometimes programming languages won't implement support for all lookarounds (notably, lookbehinds are not in the official JavaScript specification). As an example, to get full support in the RegExr tool, you'll need to set it to use the PCRE regex engine.
- *Backreferences*: we don't know ahead of time what letters might be repeated, we need a way to specify a capture group and then to repeat it: 
```py
(Hello)(Goodbye)\2\1
# \2 refers to the contents of group 2
# Will match 'HelloGoodbyeGoodbyeHello'
```
    - same two word characters in a row: (\w)\1
- re.sub() function: replace substrings, similar to str.replace() method:
```py
re.sub(pattern, repl, string, flags=0)
repl: text to substitute the match
```
- Multiple capture groups: `(.+)\s(.+)`
- Named capture groups ?P<name>: `(?P<date>.+)\s(?P<time>.+)`

##### List Comprehensions and Lambda Functions
- JavaScript Object Notation (JSON): From a Python perspective, can be thought as a collection of Python objects nested inside each other.
- The Python json module contains a number of functions to make working with JSON objects easier. We can use the json.loads() method to convert JSON data contained in a string to the equivalent set of Python objects:
```py
json_string = """
[
  {
    "name": "Sabine",
    "age": 36,
    "favorite_foods": ["Pumpkin", "Oatmeal"]
  },
  {
    "name": "Zoe",
    "age": 40,
    "favorite_foods": ["Chicken", "Pizza", "Chocolate"]
  }
]
"""
import json
json_obj = json.loads(json_string)
print(type(json_obj))
# The order of the keys in the dictionary have changed. This is because (prior to version 3.6) Python dictionaries don't have fixed order.
```
- Application programming interface (API): can be used to send and transmit data between different computer systems.
- The json.loads() function: used for loading JSON data from a string ("loads" is short for "load string"), whereas the json.load() function is used to load from a file object
- json.dumps() function ("dump string") which does the opposite of the json.loads() function — it takes a JSON object and returns a string version of it
- del statement: delete a key from a dictionary 
```py
d = {'a': 1, 'b': 2, 'c': 3}
del d['a']
```
- *List comprehension* provides a concise way of creating/transforming/reducing lists in a single line of code.
```py
ints = [1,2,3]
plus_one = []
for i in ints:
    if i > 2:
        plus_one.append(i + 1)
# List comprehension
plus_one = [i+1 for i in ints if i>2]
```
- jprint(json_obj)
- The parentheses are what tells Python to execute the function, so if we omit the parentheses we can treat a function like a *variable*
    - Eg. assign a function to a new variable name
    ```
    greet_2 = greet
    greet_2()
    ```
- Run a function inside another function by passing it as an argument:
```py
def run_func(func):
    print("RUNNING FUNCTION: {}".format(func))
    return func()
run_func(greet)
# RUNNING FUNCTION: function greet at 0x12a64c400
#'hello'

# The format() method formats the specified value(s) and insert them inside the string's placeholder.
```
- There is a way we can actually tell functions like min(), max(), and sorted() how to sort complex objects like dictionaries and lists of lists: using the optional **key argument**. The official Python documentation describes it works:
    - The key argument specifies a one-argument ordering function like that used for list.sort(). Key specifies a function of one argument that is used to extract a comparison key from each list element. The key corresponding to each item in the list is calculated once and then used for the entire sorting process.
```py
# Find dict with min age in json dataset
def get_age(json_dict):
    return json_dict['age']
​
youngest = min(json_obj, key=get_age)
jprint(youngest)
```
- *Lambda functions* special syntax to create temporary functions 
    - If a function is particularly complex, it may be a better choice to define a regular function rather than create a lambda
```py
def add(x, y):
    return x + y
# Lambda function
add = lambda x, y: x + y
```
- Apply lambda function as a key argument for sorted(): `sorted(hn_clean, key=lambda d: d['age'])`
- Pandas has the pandas.read_json() function, which is designed to read JSON from either a file or a JSON string
    - Should prepare data as a list of dictionaries so pandas is easily able to convert to a dataframe.
- We can use the pandas.DataFrame() constructor and pass the list of dictionaries directly to it to convert the JSON to a dataframe: `hn_df = pd.DataFrame(hn_clean)`
- *Ternary operator*: `[on_true] if [expression] else [on_false]`