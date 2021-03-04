##### What is OOP
- In Python, everything is an object - built by 'class' keyword
- Object have methods: `.<attribute>`
- Allow us to go beyond Python: creat your own objects

##### OOP
- Define a class: 
```
#by convention, class name is camel case
class BigOject:
    pass

obj1 = BigObject()
```

```
class PlayerCharacter:
    #Constructor method
    def __init__(self, name, age):
        self.name = name   #self= this in Javascript?
        self.age = age
    def run(self):
        print('run')
        return 'done'   #function returns None without 'return'
        
player1 = PlayerCharacter('Cindy', 44)

print(player1.age)   #age is an attribute
print(player1.run()) #run is a method so it should has brackets when calling
```
- See blueprint of a class: help(class)
- Class method: can use without instantiating a class.
- Static method:
    - Similar to class method except for cannot access to cls (class)
    - When we don't care about the class state
- 4 pillars of OOP programming languages:
    - Encapsulation: packing data and functions into attributes and methods
    - Abstraction: hiding information and accessing what is necessary
        - Python doesn't have true private variable(cannot be accessed and modified). It is convention to note them by adding underscore: self._name
    - Inheritance: new objects can take property of existing objects
    - Polymorphism: different object classes can share method names but each one does different things based on the attribute

- Introspection: the ability to determine the type of an object at runtime
    - Eg: print(dir(<instance>)) - display all the methods and attributes of the instance