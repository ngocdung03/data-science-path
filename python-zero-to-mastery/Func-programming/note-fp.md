##### Functional programming
- Separations of Concern
- 4 pillars:
    - Pure functions:
        - Same output everytime being given same input
        - Should not produce any side effect

- Function is different from method, it's seperated from the class in functional programming

- Lambda function: anonymous function, when you use a function only once. Make your code small but less readable
    lambda param: function(param)
    lambda param: action(param)  #instead of a function, can be a manipulator

- Comprehension: quick way to create list/set without looping or appending
    - my_list = [param for param in iterable]
    - my_dict = {key:value**2 for key,value in simple_dict.items()}
    - Not readable -> should include desriptive lines