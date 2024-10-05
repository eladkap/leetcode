import math


class Vector:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def __str__(self):
        return f'({self._x},{self._y},{self._z})'

    def __add__(self, other):
        return Vector(self._x + other._x, self._y + other._y, self._z + other._z)

    def __sub__(self, other):
        return Vector(self._x - other._x, self._y - other._y, self._z - other._z)

    def __pow__(self, p):
        return Vector(self._x ** p, self._y ** p, self._z ** p)

    def __iadd__(self, value):
        self._x += value
        self._y += value
        self._z += value
        return self

    def __isub__(self, value):
        self._x -= value
        self._y -= value
        self._z -= value
        return self

    def __cmp__(self, other):
        return self._x == other._x and self._y == other._y and self._z == other._z

    def __neg__(self):
        return Vector(-self._x, -self._y, -self._z)

    def __abs__(self):
        return math.sqrt(self._x ** 2 + self._y ** 2 + self._z ** 2)


if __name__ == '__main__':
    A = Vector(1, 4, 8)
    B = Vector(2, 4, 6)

    print(f'A = {A}')
    print(f'B = {B}')

    C = A + B
    print(C)

    D = A ** 2
    print(D)

    print(A == B)

    Am = -A
    print(Am)

    print(abs(A))
    print(abs(Am))
