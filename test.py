class T:
    def __init__(self,x):
        self.x =x ;
a = T(2)
b = a
b.x = 10
print a.x
print b.x