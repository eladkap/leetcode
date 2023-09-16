class Matrix(object):
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.mat = []
        for i in range(rows):
            row = [0] * cols
            self.mat.append(row)

    def __str__(self):
        return '\n'.join([str(row) for row in self.mat])

    def set_all(self, value: int):
        for i in range(self.rows):
            for j in range(self.cols):
                self.mat[i][j] = value

    def set_range(self, start: int):
        for i in range(self.rows):
            for j in range(self.cols):
                self.mat[i][j] = start
                start += 1

    def set_matrix(self, mat: list):
        self.mat = mat.copy()

    def spiral_order(self):
        m = self.rows
        n = self.cols

        count = 0
        left = 0
        right = n
        top = 0
        bottom = m
        res = []
        while count < m * n:
            # go right
            for j in range(left, right):
                res.append(self.mat[top][j])
                count += 1
            top += 1

            if count == m * n:
                break

            # go down
            for i in range(top, bottom):
                res.append(self.mat[i][right - 1])
                count += 1
            right -= 1

            if count == m * n:
                break

            # go left
            for j in range(right - 1, left - 1, -1):
                res.append(self.mat[bottom - 1][j])
                count += 1
            bottom -= 1

            if count == m * n:
                break

            # go up
            for i in range(bottom - 1, top - 1, -1):
                res.append(self.mat[i][left])
                count += 1
            left += 1

        return res

    def invert(self):
        n = self.cols
        for i in range(self.rows):
            for j in range(self.cols // 2):
                self.mat[i][j], self.mat[i][n - j - 1] = self.mat[i][n - j - 1], self.mat[i][j]

    def transponse(self):
        if self.rows != self.cols:
            raise Exception('Error: cannot transpose matrix that is not square.')
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                self.mat[i][j], self.mat[j][i] = self.mat[j][i], self.mat[i][j]

    def reverse_columns(self):
        n = self.cols
        for i in range(self.rows):
            for j in range(self.cols // 2):
                self.mat[i][j], self.mat[i][n - j - 1] = self.mat[i][n - j - 1], self.mat[i][j]

    def rotate(self, clockwise=True):
        M.transponse()
        M.reverse_columns()


if __name__ == '__main__':
    M = Matrix(4, 4)

    M.set_range(1)
    print(M)

    print('-' * 30)

    M.rotate()
    print(M)
