some_list = ['a', 'b', 'c', 'b', 'd', 'n', 'n']

# duplicates = []
# for value in some_list:
#     if some_list.count(value) > 1:
#         if value not in duplicates:
#             duplicates.append(value)

# print(duplicates)

duplicates = list(set([char for char in some_list if some_list.count(char) > 1]))

print(duplicates)
