- repl.it: can write Python codes online or other translator programs
- glot.io: similar
- Python documentation: docs.python.org
- String is immutable if we change one or more elements of it
- Best practices for commenting: realpython.com/python-comments-guide/#python-commenting-best-practices
- Docstrings convention: https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings
- List:
    - Reverse a list: basket.reverse = basket[::-1]
    - Copy a list: basket[:]
    - Get a continuous list: list(range(...))
    - `.join()`: `' '.join('hi', 'my', 'name', 'is', 'JOJO')`
    - Unpack a list: `a,b,c,*other,d = [1,2,3,4,5,6]` or `a,b,c = 1,2,3`
- Difference between list and dictionary:
    - List can be in order
    - Dictionary holds more information than list
- Truthy and Falsy: Falsy value: None, 0, anything empty (string, list,...)
- Ternary Operator: condiion_if_true if conditionelse condition_if_else
```
is_friend = True
can_message = "message allowed" if is_friend else "not allowed to message"
```
- Short circuiting
- enumerate(): for i, element in enumerate('Hello')
- break, continue, pass:
    - continue: back to the beginning of the loop?
    - pass: placeholder for the loop
- return: 
    - without `return`, function returns nothing
    - function should do one thing really well
- Docstrings for function: ``` Enter ```
    - Useful when hovering over the function call later
    - help(<functionname>) or print(<functionname>.__doc__)
- *args and *kwargs:
    - args
    ```
    def super_func(*args):   #accept one or multiple arguments
        print(*args)   
        print(args)    #print a tuple
    ```
    - kwargs: keyword arguments
    ```
    def super_func(*args, **kwargs):   
        print(kwargs)   
    
    print(super_func(1,2,3,4,num1=8, num2=10)) #print a dictionary with 2 elements
    ```
    - Rule: params, *args, default parameters, **kwargs: `super_func(name, *args, i='hi', **kwargs)`