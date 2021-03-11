##### Cleaning data intermediate
- Import csv file as list of lists
```python
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
```
##### Data analysis basic
- str.replace(oldSubString, newSubString)
- str.title(): first letter of each word transformed to uppercase
```
The Cool Thing About This String Is That It Has A Combination Of Uppercase And Lowercase Letters!
```
- str.split(): split the string into two
- To indicate the precision of two, we specify `:.2f` after the name or position of our argument: `"I own {:.2f}% of the company".format{32.554865}`
- To add commas as thoudsands separator, you would use the syntax `:,`: `"The population of India is {0:,}".format(132400000)`
- We can also combine the thousands separator and the precision by specifying them in this order: `"Your bank balance is USD{bal:,.2f}".format(bal=12345.678)`

##### Object-oriented
- In OOP, objects have types, but instead of "type" we use the word class. 
- Object and class:
    - An object is an entity that stores data.
    - An object's class defines specific properties objects of that class will have.
- There is a convention used for variables and functions in Python called **Snake Case**, where all lowercase letters are used with underscores between: `like_this`
- With classes, the convention is to use **Camel Case**, where no underscores are used between words, and the first letter of each word is capitalized: `LikeThis`.
- Methods have a "phantom" argument that gets passed to them when they are called. We need to include that in our method definition.
```
class MyClass:

    def first_method():   #should include 'self' in argument
        print("This is my first method")

my_instance = MyClass()

my_instance.first_method()
#Python automatically adds in an argument representing the instance we're calling the method on

```
- Data is stored inside objects using attributes.
- We define what is done with any arguments provided at instantiation using the init method.
- The init method — also called a constructor — is a special method that runs when an instance is created so we can perform any tasks to set up the instance.
```python
class ExampleClass:

    def __init__(self, string):
        print(string)

mc = ExampleClass("Hola!")
#Hola!
#ExampleClass.__init__(mc, "Hola!")
```
- The init method's most common usage is to store data as an attribute:
```python
class ExampleClass:

    def __init__(self, string):
        self.my_attribute = string

mc = ExampleClass("Hola!")
```
- Like methods, attributes are accessed using dot notation, but attributes don't have parentheses like methods do. 
- Example for creating append() and length() method:
```
class MyList:

    def __init__(self, initial_data):
        self.data = initial_data
        # Calculate the initial length
        self.length = 0
        for item in self.data:
            self.length += 1

    def append(self, new_item):
        self.data = self.data + [new_item]
        # Update the length here
        self.length += 1

        
my_list = MyList([1, 1, 2, 3, 5])
print(my_list.length)

my_list.append(8)
print(my_list.length)
```

##### Date and time
- Import module 
    - by name or alias:
    ```
    import csv as c
    c.reader()
    ```
    - Import one or more definitions from the module by name:
    ```
    from csv import reader, writer

    reader()
    writer()
    ```
    - Import all definitions with a wildcard
    ```python
    from csv import *

    reader()
    writer()
    get_dialect()

    #it may not be immediately clear where a definition comes from. This can also be a problem if we use this approach with multiple modules.
    #it's easier to accidentally overwrite an imported definition.
    ```
- The most useful module for working with data is the datetime module.
- The datetime.strptime() constructor returns a datetime object defined using a special syntax system to describe date and time formats called strftime.
```python 
datetime.strptime("24/12/1984", "%d/%m/%Y")
# The %d, %m, and %Y format codes specify a two-digit day, two-digit month, and four-digit year respectively, and the forward slashes between them specify the forward slashes in the original string. 
```
- Documentation: string parse time and string format time
    https://strftime.org/
    https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
- The datetime class has a number of attributes which make it easy to retrieve the various parts that make up the date stored within the object: datetime.day, datetime.month, datetime.year, datetime.hour, datetime.minute.
- Date string format
```python
dt_object = dt.datetime(1984, 12, 24)
dt_string = dt_object.strftime("%d/%m/%Y")
print(dt_string)
#24/12/1984

# Use %B to represent the month as a word:
dt_string = dt_object.strftime("%B %d, %Y")
print(dt_string)
#December 24, 1984

# Use %A, %I, %M, and %p to represent the day of the week, the hour of the day, the minute of the hour, and a.m./p.m.:
dt_string = dt_object.strftime("%A %B %d at %I:%M %p")
print(dt_string)
#Monday December 24 at 12:00 AM
```
- Apart from having no strptime constructor, time objects behave similarly to datetime objects: attributes like time.hour and time.second; time.strftime() method, which you can use to create a formatted string representation of the object.
- When we use the - operator with two date objects, the result is the time difference between the two datetime objects. The resultant object is a datetime.timedelta object. 
- we can also instantiate a timedelta class directly: `datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)`
- We can also use timedelta objects to add or subtract time from datetime objects.
```python
d1 = dt.date(1963, 2, 21)
d1_plus_1wk = d1 + dt.timedelta(weeks=1)
print(d1_plus_1wk)
```








