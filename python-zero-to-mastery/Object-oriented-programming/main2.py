##### Inheritance

class User:
    def sign_in(self):
        print('logged in') #if we don't have any variables or attributes that we want to assign to the user, wouldn't need __init__ method

# Subclasses
class Wizard(User):  #pass the inheritance of User
    def __init__(self, name, power):
        self.name = name
        self.power = power

    def attack(self):    #should pass self as argument, otherwise error
        print(f'attacking with power of {self.power}')

class Archer():
    def __init__(self, name, num_arrows):
        self.name = name
        self.num_arrows = num_arrows

    def attack(self):
        print(f'attacking with arrows: arrows left- {self.num_arrows}')

wizard1 = Wizard('Merlin', 50)
archer1 = Archer('Robin', 100)