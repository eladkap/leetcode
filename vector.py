class Vector:
    def __init__(self, size=0):
        if size == 0:
            self._arr = []
        else:
            self._arr = [0] * size

    def __str__(self):
        return '[' + ' '.join(str(x) for x in self._arr) + ']'

    def __add__(self, other):
        v = Vector()
        for x in self._arr + other._arr:
            v._arr.append(x)
        return v

    def __sub__(self, other):
        pass

    def __pow__(self, power):
        B = Vector(len(self._arr))
        for i in range(len(self._arr)):
            B._arr[i] = self._arr[i] ** power
        return B

    def __iadd__(self, x):
        self._arr.append(x)
        return self

    def __cmp__(self, other):
        pass


if __name__ == '__main__':
    A = Vector()
    B = Vector(size=8)

    for x in range(1, 11):
        A += x

    print(A)
    A2 = A ** 2

    print(A2)
