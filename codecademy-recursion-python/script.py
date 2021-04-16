### Building Our Own Call Stack
# Execution contexts are a mapping between variable names and their values within the function during execution. We can use a list for our call stack and a dictionary for each execution context.
# define your sum_to_one() function above the function call
def sum_to_one(n):
  result = 1
  call_stack = []
  while n > 1:
    execution_context = {"n_value": n}
    call_stack.append(execution_context)
    n -= 1
    print(call_stack)
  #base case
  print("BASE CASE REACHED")
  
  # now address the conclusion of this function, where the separate values stored in the call stack are accumulated into a single return value.
  # This is the point in a recursive function when we would begin returning values as the function calls are “popped” off the call stack.
  while len(call_stack) != 0:
    return_value = call_stack.pop()
    print(call_stack)
    print("Return value of {0} adding to result {1}".format(return_value['n_value'], result))
    result += return_value['n_value']
  return result, call_stack

sum_to_one(4)
sum_to_one(4)
#Execution context:
# [{'n_value': 4}]
# [{'n_value': 4}, {'n_value': 3}]
# [{'n_value': 4}, {'n_value': 3}, {'n_value': 2}]
# BASE CASE REACHED


# [{'n_value': 4}, {'n_value': 3}]
# Return value of 2 adding to result 1
# [{'n_value': 4}]
# Return value of 3 adding to result 3
# []
# Return value of 4 adding to result 6

## Define a recursive function
def sum_to_one(n):
  # Base case
  if n == 1:
    return n
  # Recursive step
  # return value = current input + return value of sum_to_one()
  # invoke sum_to_one() with an argument that will get us closer to the base case
  print("Recursing with input: {0}".format(n))
  return n + sum_to_one(n-1)

print(sum_to_one(7))

## Another recursive function
def factorial(n):
  if n < 2:
    return 1
  return n*factorial(n-1)

print(factorial(122222))  # RecursionError: maximum recursion depth exceeded in comparison

## Flatten nested list
planets = ['mercury', 'venus', ['earth'], 'mars', [['jupiter', 'saturn']], 'uranus', ['neptune', 'pluto']]
def flatten(my_list):
  result = []
  for element in my_list:
    # base case
    if isinstance(element, list) == False:
      result.append(element)
    if isinstance(element, list):
      print("List found!")
      flat_list = flatten(element)  #without this recursive, result will be ['mercury', 'venus', 'earth', 'mars', ['jupiter', 'saturn'], 'uranus', 'neptune', 'pluto']
      result += flat_list
  print(result)
  return result

flatten(planets)

## Binary search tree: two children at most per tree node, one < parent and the other is greater.

def build_bst(my_list):  # sorted list of values as input.
  # base case
  if len(my_list)==0: 
    return "No Child"
  # recursive step
  middle_idx = len(my_list)//2 #round(len(my_list)/2)
  middle_value = my_list[middle_idx]
  print("Middle index: {0}".format(middle_idx))
  print("Middle value: {0}".format(middle_value))
  tree_node = {"data":middle_value}
  tree_node["left_child"] = build_bst(my_list[:middle_idx])
  tree_node["right_child"] = build_bst(my_list[middle_idx+1:])
  return tree_node

# For testing
sorted_list = [12, 13, 14, 15, 16]
binary_search_tree = build_bst(sorted_list)
print(binary_search_tree)

# fill in the runtime as a string
# 1, logN, N, N*logN, N^2, 2^N, N!
runtime = "N*logN"