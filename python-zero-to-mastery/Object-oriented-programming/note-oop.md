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