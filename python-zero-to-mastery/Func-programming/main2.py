# Comprehension: list, set, dictionary
# my_list = []

# for char in 'hello':
#     my_list.append(char)

#my_list = [param for param in iterable]
my_list = [char for char in 'hello']
print(my_list)

my_list2 = [num**2 for num in range(0, 100)]
print(my_list2)

# generate list of even number out of squared 
my_list4 = [num**2 for num in range(0, 100) if num%2==0]
print(my_list4)

# Similar with set
my_set = {char for char in 'hello'}  #only allow unique items

# Dictionary 
#my_dict = {key:value**2 for key,value in simple_dict.items()}
simple_dict = {
    'a': 1,
    'b': 2
}
my_dict = {key:value**2 for key,value in simple_dict.items() if value%2==0}  #.items: grab the key and value

print(my_dict)

# Making dictionary from a list
my_dict2 = {num:num*2 for num in [1,2,3]}
print(my_dict2)