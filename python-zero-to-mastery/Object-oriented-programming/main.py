#OOP
class PlayerCharacter:
    # Class Object Attribute
    membership = True  #unlike regular class attribute, it's static -> exist for all objects
    #Constructor method
    def __init__(self, name, age):
        if (self.membership):  #can be PlayerCharacter.membership for COA
        self.name = name   #self= this in Javascript?
        self.age = age
    def run(self):
        print('run')
        return 'done'   #function returns None without 'return'
    @classmethod # @ is decorator?
    def adding_things(cls, num1, num2)  #notice cls (class) instead of self
        return cls('Teddy', num1 + num2)    #can instantiated an object inside a class method
    
    @staticmethod 
    def adding_things2(num1, num2)  #notice cls (class) instead of self
        return num1 + num2
          
player1 = PlayerCharacter('Cindy', 44)

print(player1.age)   #age is an attribute
print(player1.run()) #run is a method so it should has brackets when calling

# Calling a class method
print(PlayerCharacter.adding_things)

# Instantiate player 3 using class method
player3 = PlayerCharacter.adding_things(2,3)
print(player3.age)  #5