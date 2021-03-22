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

