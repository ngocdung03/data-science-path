# Decorator
def my_decorator(func):
    def wrap_func():
        print('*********')
        func()
        print('*********')
    return wrap_func

@my_decorator
def hello():
    print('helloo')

# hello()  #the decorator has enhanced the function hello()

# @my_decorator
def bye():
    print('see ya later')
# bye()

bye2 = my_decorator(bye)
bye2()

my_decorator(bye)()  #same output

# Decorator pattern: adding parameter
def my_decorator2(func):
    def wrap_func(*args, **kwargs):  #notice adding a parameter here
        print('*********')
        func(*args, **kwargs)
        print('*********')
    return wrap_func

@my_decorator2
def hello2(greeting, emoji = ':)'):
    print(greeting, emoji)

hello2('Hiiii')