
class Person:

    def set_details(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print('I am ', self.name)

    def greet(self):
        if self.age < 80:
            print('Hello, how are you doing?')
        else:
            print('Hello, how do you do?')
        # self.display()

p1 = Person()
# p2 = Person()


p1.set_details('Bob', 20)
# p1.name
# p1.age
# Call method
p1.display()
p1.greet()


# Data hiding
class Product:
    def __init__(self):
        self.data1 = 10
        self._data2 = 20
    
    def methodA(self):
        pass

    def _methodB(self):
        pass

p = Product()
print(p.data1)
print(p.methodA())
print(p._data2)
print(p._methodB())
dir(p)

# Property
class Product:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def display(self):
        print(self._x, self._y)

    @property
    def value(self):
        return self._x

    @value.setter    #assign alternate value for value
    def value(self,val):
        self._x = val

p = Product(12,24)
p.value = 10

# Property 2 with private variable
# This kind of code validation will break the interface without backward compatible
class Person:
    def __init__(self, name, age):
        self.name = name
        if 20 < age < 80:
            self._age = age
        else:
            raise ValueError('Age must be between 20 and 80')

    def display(self):
        print(self.name, self._age)

    def set_age(self, new_age):
        if 20 < new_age < 80:
            self._age = new_age
        else:
            raise ValueError('Age must be between 20 and 80')

    def get_age(self):
        return self._age

if __name__ == '__main__':
    p = Person('Raj', 30)
    p.display()

p.set_age(25)
p.display()

# Alternative with decorations
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(self.name, self.age)

    @property
    def age(self):         
        return self._age       #now can access age as an instance variable
    
    @age.setter
    def age(self, new_age):
        if 20 < new_age < 80:
            self._age = new_age
        else:
            raise ValueError('Age must be between 20 and 80')

if __name__ == '__main__':
    p = Person('Raj', 30)
    p.display()

p.age
p.age = 34

p1 = Person('Dev', 200)

# Property 3: read-only, write-only and editable method
class Employee:
    def __init__(self, name, password, salary):
        self._name = name
        self._password = password
        self._salary = salary
    
    @property    #read only
    def name(self):
        return self._name
    
    @property   #write only
    def password(self):
        raise AttributeError('password not readable')

    @password.setter
    def password(self, new_password):
        self._password = new_password

    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, new_salary):
        self._password = new_salary

e = Employee('Jill', 'feb31', 5000)
print(e.name)
e.name = 'xyz'

print(e.password)
e.password = 'abc'

e.salary
e.salary = 6000

# Property 4
class Rectangle():
    def __init__(self, length, breadth):
        self.length = length
        self.breadth = breadth
        self.diagonal = (self.length*self.length + self.breadth*self.breadth)**0.5

    def area(self):
        return self.length * self.breadth

    def perimeter(self):
        return 2*(self.length + self.breadth)

r = Rectangle(2,5)
r.diagonal
r.area()

r.length = 10
r.diagonal # not change
r.area()  #change because they are implemented as methods

# Property 4.2: making diagonal a dynamic instance variable
class Rectangle():
    def __init__(self, length, breadth):
        self.length = length
        self.breadth = breadth
    
    @property
    def diagonal(self):
        return(self.length*self.length + self.breadth*self.breadth)**0.5

    def area(self):
        return self.length * self.breadth

    def perimeter(self):
        return 2*(self.length + self.breadth)

r = Rectangle(2,5)
r.diagonal
r.area()

r.length = 10
r.diagonal 
r.area()

# Property 5: delete method
class Product:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def display(self):
        print(self._x, self._y)

    @property
    def value(self):
        return self._x

    @value.setter
    def value(self,val):
        self._x = val

    @value.deleter
    def value(self):
        print('value deleted')

p = Product(12,24)

# Class variable
class Person:
    species = 'Homo sapiens'

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(f'{self.name} is {self.age} years old')

p1 = Person('John', 20)
p2 = Person('Jack', 34)

p1.display()
p2.display()

Person.species
p1.species
p2.species
Person.name #error

# Class method
# Class method in instancializing by different type of data
class Employee:
    def __init__(self, first_name, last_name, birth_year, salary):
        self.first_name = first_name
        self.last_name = last_name
        self.birth_year = birth_year
        self.salary = salary
    
    def show(self):
        print(f'''I am {self.first_name} {self.last_name} born in {self.birth_year}''')
from employee import Employee
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print('I am', self.name, self.age, 'years old')


    @classmethod
    def from_str(cls, s):
        name,age = s.split(',')
        return cls(name, int(age))   #create a new Person instance object

    @classmethod
    def from_dict(cls, d):
        return cls(d['name'], d['age'])

    @classmethod
    def from_employee(cls, emp):
        name = emp.first_name + ' ' + emp.last_name
        age = datetime.today().year - emp.birth_year
        return cls(name, age) #send data to the Person class initializer

p1 = Person('John', 20)

s = 'Jim, 23'
p3 = Person.from_str(s)
p3.display()

d = {'name': 'Jane', 'age': 34}
p4 = Person.from_dict(d)
p4.display()

e1 = Employee('James', 'Smith', 1990, 6000)
p5 = Person.from_employee(e1)

# Static method 

# Magic method
# Magic Methods - 1
class Fraction:
    def __init__(self,nr,dr=1):
        self.nr = nr
        self.dr = dr
        if self.dr < 0:
            self.nr *= -1
            self.dr *= -1
        self._reduce()

    def show(self):
        print(f'{self.nr}/{self.dr}')

    def __add__(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.dr + other.nr * self.dr,
    self.dr * other.dr)
        f._reduce()
        return f

    def __sub__(self,other):   #subtract
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.dr - other.nr * self.dr,
    self.dr * other.dr)
        f._reduce()
        return f

    def __mul__(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.nr , self.dr * other.dr)
        f._reduce()
        return f

    def _reduce(self):
        h = Fraction.hcf(self.nr, self.dr)
        if h == 0:
            return
        self.nr //= h
        self.dr //= h

    @staticmethod
    def hcf(x,y):
        x=abs(x)
        y=abs(y)
        smaller = y if x>y else x
        s = smaller
        while s>0:
            if x%s==0 and y%s==0:
                break
            s-=1
        return s
>>>f1 = Fraction(2,3)
>>>f2 = Fraction(3,4)
>>>f3 = f1+f2
>>>f3 = f1*f2
>>>f3 = f1.add(f2)
>>>f3.show()
>>>f3 = f1.multiply(f2)

class Fraction:
    def __init__(self,nr,dr=1):
        self.nr = nr
        self.dr = dr
        if self.dr < 0:
            self.nr *= -1
            self.dr *= -1
        self._reduce()
    def show(self):
        print(f'{self.nr}/{self.dr}')
    def __add__(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.dr + other.nr * self.dr, self.dr * other.dr)
        f._reduce()
        return f

    def __sub__(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.dr - other.nr * self.dr, self.dr * other.dr)
        f._reduce()
        return f
    def __mul__(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.nr , self.dr * other.dr)
        f._reduce()
        return f
    def _reduce(self):
        h = Fraction.hcf(self.nr, self.dr)
        if h == 0:
            return
        self.nr //= h
        self.dr //= h
    @staticmethod
    def hcf(x,y):
        x=abs(x)
        y=abs(y)
        smaller = y if x>y else x
        s = smaller
        while s>0:
            if x%s==0 and y%s==0:
                break
            s-=1
        return s
>>>f3 = f1.__add__(f2)
>>>f3
>>>f3 = f1 + f2
>>>f3
>>>f3 = f1-f2
>>>f3
>>>f3 = f1*f2"Python in Depth" by Deepali Srivastava 19
>>f3
>>f3 = f1-2

# Magic Methods - 2
class Fraction:
    def __init__(self,nr,dr=1):
        self.nr = nr
        self.dr = dr
        if self.dr < 0:
            self.nr *= -1
            self.dr *= -1
        self._reduce()
    def show(self):
        print(f'{self.nr}/{self.dr}')
    def add(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.dr + other.nr * self.dr, self.dr * other.dr)
        f._reduce()
        return f
    def multiply(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        f = Fraction(self.nr * other.nr , self.dr * other.dr)
        f._reduce()
        return f
    def __eq__(self,other):
        return (self.nr * other.dr) == (self.dr * other.nr)
    def __lt__(self,other):
        return (self.nr * other.dr) < (self.dr * other.nr)
    def __le__(self,other):
        return (self.nr * other.dr) <= (self.dr * other.nr)
    def __str__(self):
        return f'{self.nr}/{self.dr}'
    def __repr__(self):
        return f'Fraction({self.nr},{self.dr})'
    def _reduce(self):
        h = Fraction.hcf(self.nr, self.dr)
        if h == 0:
            return
        self.nr //= h
        self.dr //= h
    
    @staticmethod
    def hcf(x,y):
        x=abs(x)
        y=abs(y)
        smaller = y if x>y else x
        s = smaller
        while s>0:
            if x%s==0 and y%s==0:
                break
            s-=1
        return s
>>>f1 = Fraction(2,3)
>>>f2 = Fraction(2,3)
>>>f3 = Fraction(4,6)
>>>f1 == f2
>>>f1 == f3
>>>f1 != f2
>>>f1 = Fraction(2,3)
>>>f2 = Fraction(2,3)
>>>f3 = Fraction(1,5)
>>>f1 < f2
>>>f1 <= f2
>>>f1 < f3
>>>f3 < f1
>>>str(f1)
>>>f1
>>>f1 = Fraction(3,4)
>>>f2 = Fraction(4,5)
>>>f3 = Fraction(1,5)
>>>L = [f1,f2,f3]
>>>print(L)

# Magic Methods - 3
def __radd__(self,other):
    return self.__add__(other)
>>f2 = f1+3
>>f2 = 3 + f1