class Matrix(object):
    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._mat = []
        self._iterator = -1
        for i in range(rows):
            row = [0] * cols
            self._mat.append(row)

    def __str__(self):
        return '\n'.join([str(row) for row in self._mat])

    def set_all(self, value: int):
        for i in range(self._rows):
            for j in range(self._cols):
                self._mat[i][j] = value

    def set_range(self, start: int):
        for i in range(self._rows):
            for j in range(self._cols):
                self._mat[i][j] = start
                start += 1

    def set_matrix(self, mat: list):
        self._mat = mat.copy()

    def spiral_order(self):
        m = self._rows
        n = self._cols

        count = 0
        left = 0
        right = n
        top = 0
        bottom = m
        res = []
        while count < m * n:
            # go right
            for j in range(left, right):
                res.append(self._mat[top][j])
                count += 1
            top += 1

            if count == m * n:
                break

            # go down
            for i in range(top, bottom):
                res.append(self._mat[i][right - 1])
                count += 1
            right -= 1

            if count == m * n:
                break

            # go left
            for j in range(right - 1, left - 1, -1):
                res.append(self._mat[bottom - 1][j])
                count += 1
            bottom -= 1

            if count == m * n:
                break

            # go up
            for i in range(bottom - 1, top - 1, -1):
                res.append(self._mat[i][left])
                count += 1
            left += 1

        return res

    def invert(self):
        n = self._cols
        for i in range(self._rows):
            for j in range(self._cols // 2):
                self._mat[i][j], self._mat[i][n - j - 1] = self._mat[i][n - j - 1], self._mat[i][j]

    def transponse(self):
        if self._rows != self._cols:
            raise Exception('Error: cannot transpose matrix that is not square.')
        for i in range(self._rows):
            for j in range(i + 1, self._cols):
                self._mat[j][i], self._mat[i][j] = self._mat[i][j], self._mat[j][i]

    def reverse_columns(self):
        n = self._cols
        for i in range(self._rows):
            for j in range(self._cols // 2):
                self._mat[i][j], self._mat[i][n - j - 1] = self._mat[i][n - j - 1], self._mat[i][j]

    def rotate(self, clockwise=True):
        M.transponse()
        M.reverse_columns()

    def __len__(self):
        return self._rows * self._cols

    def __iter__(self):
        return self

    def __next__(self):
        self._iterator += 1
        if self._iterator < self._rows * self._cols:
            return self._mat[self._iterator // self._rows][self._iterator % self._rows]
        else:
            raise StopIteration

    def __invert__(self):
        return self.invert()

    def __mul__(self, value):
        for i in range(self._rows):
            for j in range(self._cols):
                self._mat[i][j] *= value
        return self


if __name__ == '__main__':
    M = Matrix(4, 4)

    M.set_range(1)
    print(M)

    M.set_all(1)

    print(M)

    M *= 2

    print(M)

    # for x in M:
    #     print(x)

    # print('-' * 30)

    # M.invert()
    # print(M)

    # print(len(M))
