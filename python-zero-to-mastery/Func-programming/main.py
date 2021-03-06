# Pure functions
# map, filter, zip, and reduce
# lambda function

# def mulitply_by2(list):
#     new_list = []
#     for item in list:
#         new_list.append(item*2)
#     #return print(new_list)   #this has side effect
#     return new_list
# print(mulitply_by2([1,2,3]))

my_list = [1,2,3]
# map
def multiply_by2(item):
    return item*2
print(list(map(multiply_by2, my_list)))  #doesn't change my list -> pure function

# filter
def only_odd(item):
    return item % 2 != 0

print(list(filter(only_odd, my_list)))

# zip
your_list = [10, 20, 30]
print(list(zip(my_list, your_list)))

# reduce
from functools import reduce  #functools is a tool belt used for functional tools that comes with python installation

def accumulator(acc, item):
    print(acc, item)
    return acc + item
print(reduce(accumulator, my_list, 0))
# reduce with return a value instead of a list, so doesn't need list()

# lambda expressions
#lambda param: function(param)
print(list(map(lambda item: item*2 , my_list)))
print(reduce(lambda acc, item: acc+item, my_list))
