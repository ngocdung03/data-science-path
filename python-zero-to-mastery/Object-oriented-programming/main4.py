#####MRO - Method Resolution Order
class A:
    num = 10

class B(A):
    pass

class C(A):
    num = 1

class D(B, C):
    pass

# to check the order MRO
#print(D.__mro__)
print(D.mro)   # D -> B ->C -> A
print(D.num)   #1

#########Complicated MRO##########
class X: pass
class Y: pass
class Z: pass
class A(X, Y): pass
class B(Y, Z): pass
class M(B, A, Z): pass

print(M.__mro__)   # M - B - A - X - Y - Z - object  - Depth First Search