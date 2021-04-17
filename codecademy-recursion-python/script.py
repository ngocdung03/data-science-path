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

## Return power set
def power_set(my_list):
    # COMMENT 1
    if len(my_list) == 0:
        return [[]]
    # END COMMENT 1
    
    # COMMENT 2
    power_set_without_first = power_set(my_list[1:])
    # END COMMENT 2
    
    # COMMENT 3
    with_first = [ [my_list[0]] + rest for rest in power_set_without_first ]
    # END COMMENT 3
    
    # COMMENT 4
    return with_first + power_set_without_first
    # END COMMENT 4

### Recursion vs iteration
## Factorial()
# runtime: Linear - O(N)
def factorial(n):  
  if n < 0:    
    ValueError("Inputs 0 or greater only") 
  if n <= 1:    
    return 1  
  return n * factorial(n - 1)
 
factorial(3)
# 6
factorial(4)
# 24
factorial(0)
# 1
factorial(-1)
# ValueError "Input must be 0 or greater"

def factorial(n):
  if n < 0:
    ValueError("Inputs 0 or greater only")
  elif n <= 1:
    return 1
  result = 1
  for i in range(1,n+1):
    result*=i
  return result   
# test cases
print(factorial(3) == 6)
print(factorial(0) == 1)
print(factorial(5) == 120)

## Fibonacci
# runtime: Exponential - O(2^N)
def fibonacci(n):
  if n < 0:
    ValueError("Input 0 or greater only!")
  if n <= 1:
    return n
  return fibonacci(n - 1) + fibonacci(n - 2)
 
fibonacci(3)
# 2
fibonacci(7)
# 13
fibonacci(0)
# 0

def fibonacci(n):
  # if n < 0:
  #   ValueError("Input 0 or greater only!")
  # if n <= 1:
  #   return n
  sequence = [0, 1]
  if n <= (len(sequence) - 1):
    return sequence[n]
  while n > (len(sequence) - 1):
    next_value = sequence[-1] + sequence[-2]
    sequence.append(next_value)
  return sequence[n]

# test cases
print(fibonacci(3) == 2)
print(fibonacci(7) == 13)
print(fibonacci(0) == 0)

## Sum_digit
# Linear - O(N), where "N" is the number of digits in the number
def sum_digits(n):
  if n < 0:
    ValueError("Inputs 0 or greater only!")
  result = 0
  while n is not 0:
    result += n % 10
    n = n // 10
  return result + n
 
sum_digits(12)
# 1 + 2
# 3
sum_digits(552)
# 5 + 5 + 2
# 12
sum_digits(123456789)
# 1 + 2 + 3 + 4...
# 45

def sum_digits(n):
  if n < 0:
    ValueError("Inputs 0 or greater only!")
  if n < 10:
    return n
  return (n % 10) + sum_digits(n//10)
# test cases
print(sum_digits(12) == 3)
print(sum_digits(552) == 12)
print(sum_digits(123456789) == 45)

## find_min
def find_min(my_list):
  min = None
  for element in my_list:
    if not min or (element < min):
      min = element
  return min
 
find_min([42, 17, 2, -1, 67])
# -1
find_mind([])
# None
find_min([13, 72, 19, 5, 86])
# 5

def find_min(my_list, min = None):
  if len(my_list) == 0:
    return min
  if min==None or my_list[0] < min: #if not min or (element < min):
    min = my_list[0]
  return find_min(my_list[1:], min) #Note!


# test cases
print(find_min([42, 17, 2, -1, 67])==-1)
print(find_min([]) == None)
print(find_min([13, 72, 19, 5, 86]) == 5)

## Is Palindrome
def is_palindrome(my_string):
  while len(my_string) > 1:
    if my_string[0] != my_string[-1]:
      return False
    my_string = my_string[1:-1]
  return True 
 
palindrome("abba")
# True
palindrome("abcba")
# True
palindrome("")
# True
palindrome("abcd")
# False

# Linear - O(N)
def is_palindrome(my_string):
  string_length = len(my_string)
  middle_index = string_length // 2
  for index in range(0, middle_index):
    opposite_character_index = string_length - index - 1
    if my_string[index] != my_string[opposite_character_index]:
      return False  
  return True

def is_palindrome(my_string):
  if len(my_string) < 2:
    return True 
  if my_string[0] != my_string[-1]:
    return False
  return is_palindrome(my_string[1:-1])

# test cases
print(is_palindrome("abba") == True)
print(is_palindrome("abcba") == True)
print(is_palindrome("") == True)
print(is_palindrome("abcd") == False)

## Multiplication
# This implementation isn’t quite as robust as the built-in operator. It won’t work with negative numbers, for example. We don’t expect your implementation to handle negative numbers either!
def multiplication(num_1, num_2):
  result = 0
  for count in range(0, num_2):
    result += num_1
  return result
 
multiplication(3, 7)
# 21
multiplication(5, 5)
# 25
multiplication(0, 4)
# 0

def multiplication(num_1, num_2):
  if num_1 ==0 or num_2 == 0:
    return 0 
  return num_1 + multiplication(num_1, num_2 -1)

# test cases
print(multiplication(3, 7) == 21)
print(multiplication(5, 5) == 25)
print(multiplication(0, 4) == 0)

## Binary tree
def depth(tree):
  result = 0
  # our "queue" will store nodes at each level
  queue = [tree]
  # loop as long as there are nodes to explore
  while queue:
    # count the number of child nodes
    level_count = len(queue)
    for child_count in range(0, level_count):
      # loop through each child
      child = queue.pop(0)
     # add its children if they exist
      if child["left_child"]:
        queue.append(child["left_child"])
      if child["right_child"]:
        queue.append(child["right_child"])
    # count the level
    result += 1
  return result
 
two_level_tree = {
"data": 6, 
"left_child":
  {"data": 2}
}
 
four_level_tree = {
"data": 54,
"right_child":
  {"data": 93,
   "left_child":
     {"data": 63,
      "left_child":
        {"data": 59}
      }
   }
}
 
 
depth(two_level_tree)
# 2
depth(four_level_tree)
# 4

def depth(tree):
  if not tree:
    return 0

  left_depth = depth(tree["left_child"])
  right_depth = depth(tree["right_child"])

  if left_depth > right_depth:
    return left_depth + 1
  else:
    return right_depth + 1

# HELPER FUNCTION TO BUILD TREES
def build_bst(my_list):
  if len(my_list) == 0:
    return None

  mid_idx = len(my_list) // 2
  mid_val = my_list[mid_idx]

  tree_node = {"data": mid_val}
  tree_node["left_child"] = build_bst(my_list[ : mid_idx])
  tree_node["right_child"] = build_bst(my_list[mid_idx + 1 : ])

  return tree_node

# HELPER VARIABLES
tree_level_1 = build_bst([1])
tree_level_2 = build_bst([1, 2, 3])
tree_level_4 = build_bst([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) 

# test cases
print(depth(tree_level_1) == 1)
print(depth(tree_level_2) == 2)
print(depth(tree_level_4) == 4)

