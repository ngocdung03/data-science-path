##### Inheritance, Polymorphism

class User:
    def __init__(self, email):
        self.email = email

    def sign_in(self):
        print('logged in') #if we don't have any variables or attributes that we want to assign to the user, wouldn't need __init__ method
    def attack(self):
        print('do nothing')
# Subclasses
class Wizard(User):  #pass the inheritance of User
    def __init__(self, name, power, email):
        #User.__init__(self, email)
        super().__init__(email)  #super() referring to the superclass which is User, and don't need 'self' argument
        self.name = name
        self.power = power

    def attack(self):    #should pass self as argument, otherwise error
        User.attack(self)   #display 'do nothing' also
        print(f'attacking with power of {self.power}')

class Archer():
    def __init__(self, name, num_arrows):
        self.name = name
        self.num_arrows = num_arrows

    def attack(self):
        print(f'attacking with arrows: arrows left- {self.num_arrows}')

# introspection
wizard1 = Wizard('Merlin', 50, 'merlin@gmail.com')
archer1 = Archer('Robin', 100)

print(isinstance(wizard1, User))  #(instance, class)
print(isinstance(wizard1, object)) #everything in Python inherits from the base object 

for char in [wizard1, archer1]:
    char.attack()       #different output due to polymorphism

print(wizard1.email)