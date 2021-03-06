- Decorators supercharge your functions - adding some features to a function
- Higher Order Function:
    - Function that accepts other function as parameter: greet(func)
    - Function that returns other function:
    ```
    def greet2():
        def func():
            return 5
        return func
    ```
- Decorator Pattern:
```
def my_decorator2(func):
    def wrap_func(*args, **kwargs):  #notice adding a parameter here
        print('*********')
        func(*args, **kwargs)
        print('*********')
    return wrap_func
```