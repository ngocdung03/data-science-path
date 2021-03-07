# Error Handling

while True:         #a loop so user can retype if wrong input
    try:
        age = int(input('What is your age?:'))
        10/age
        #raise ValueError('hey cut it out')   #throw an error
    #if the codes above error, execute the below ones
    except ValueError: #refer to built-in exception
        print('please enter a number')
    except ZeroDivisionError: #refer to built-in exception
        print('please enter age higher than 0')
    else: 
        print('Thank you!')
        break  #if remove, print 'can you hear me?'
    finally: 
        print('ok, I am finally done')
    print('can you hear me?')