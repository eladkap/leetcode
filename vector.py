import math


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f'({self.x},{self.y},{self.z})'

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __pow__(self, power):
        return Vector(self.x ** power, self.y ** power, self.z ** power)

    def __cmp__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z


if __name__ == '__main__':
    v1 = Vector(1, 2, 3)
    v2 = Vector(2, 4, 6)
    v3 = v1 + v2
    print(v3)

    v4 = v1 - v2
    print(v4)

    s1 = v1 ** 2
    print(s1)
