lst=[1,2,3,4,5]
def gen():
    yield 1
    yield 2
ob=gen()
print(ob.__next__())
print(ob.__next__())