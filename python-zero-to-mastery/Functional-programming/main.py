##### Pure functions
def mulitply_by2(list):
    new_list = []
    for item in list:
        new_list.append(item*2)
    #return print(new_list)   #this has side effect
    return new_list

print(mulitply_by2([1,2,3]))