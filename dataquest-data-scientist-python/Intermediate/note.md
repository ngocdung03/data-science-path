#####Cleaning data intermediate
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