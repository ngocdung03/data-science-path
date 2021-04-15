##### Introduction to Recursion with Python
- The “function” of asking a person involved asking a person. The self-referential logic can seem like it goes on forever, but each question brings you closer to the front of the line where no more people are asked about the line.
- Recursion is a strategy for solving problems by defining the problem in terms of itself.
    - For example, to sum the elements of a list we would take the first element and add it to the sum of the remaining elements.
    - In programming, recursion means a function definition will include an invocation of the function within its own body.
- Parts:
    - Base case: NOT invoking the function under this condition
    - Section that solves a piece of the problem
    - Recursive step: calling the function with arguments which bring us closer to the base case. 
- A recursive approach requires the function invoking itself *with different arguments*
- Stacks, a data structure, follow a strict protocol for the order data enters and exits the structure: the last thing to enter is the first thing to leave.
- Your programming language often manages the call stack, which exists outside of any specific function. This call stack tracks the ordering of the different function invocations, so the last function to enter the call stack is the first function to exit the call stack.
- The base case dictates whether the function will recurse, or call itself. Without a base case, it’s the iterative equivalent to writing an infinite loop.
    - Because we’re using a call stack to track the function calls, your computer will throw an error due to a stack overflow if the base case is not sufficient.
- Execution contexts are a mapping between variable names and their values within the function during execution. We can use a list for our call stack and a dictionary for each execution context.
```py
# Iterative function
def sum_to_one(n):
  result = 0
  for num in range(n, 0, -1):
    result += num
  return result
 
sum_to_one(4)
# num is set to 4, 3, 2, and 1
# 10

# We can think of each recursive call as an iteration of the loop above. In other words, we want a recursive function that will produce the following function calls:
recursive_sum_to_one(4)
recursive_sum_to_one(3)
recursive_sum_to_one(2)
recursive_sum_to_one(1)
```
- recursive functions tend to be at least a little less efficient than comparable iterative solutions because of the call stack.
- The beauty of recursion is how it can reduce complex problems into an elegant solution of only a few lines of code. Recursion forces us to distill a task into its smallest piece, the base case, and the smallest step to get there, the recursive step.
- Improve the expensive runtime of recursive func: memoization https://en.wikipedia.org/wiki/Memoization
- Recursive Data Structures
    - Trees are a recursive data structure because their definition is self-referential. A tree is a data structure which contains a piece of data and references to other trees!
    - Trees which are referenced by other trees are known as children. Trees which hold references to other trees are known as the parents.