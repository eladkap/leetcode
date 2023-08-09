class Matrix:
    def __init__(self, n: int, m: int):
        self.mat = []
        self.rows = n
        self.cols = m
        for i in range(n):
            self.mat.append([0] * m)

    def __str__(self):
        s = []
        for row in self.mat:
            row_str = '\t'.join([str(x) for x in row])
            s.append(row_str)
        return '\n'.join(s)

    def fill(self, value: int):
        for i in range(self.rows):
            for j in range(self.cols):
                self.mat[i][j] = value

    def fill_range(self, start: int, step: int):
        value = start
        for i in range(self.rows):
            for j in range(self.cols):
                self.mat[i][j] = value
                value += step

    def mirror(self):
        for i in range(self.rows):
            for j in range(self.cols // 2):
                k = self.cols - j - 1
                self.mat[i][j], self.mat[i][k] = self.mat[i][k], self.mat[i][j]

    def spiral(self) -> list:
        res = []
        rows = len(self.mat)
        cols = len(self.mat[0])

        left = 0
        right = cols - 1
        top = 0
        bottom = rows - 1

        cells = rows * cols
        count = 0

        while count < cells:
            # move right
            for j in range(left, right + 1):
                res.append(self.mat[top][j])
                count += 1

            # move down
            if count >= cells:
                break
            for i in range(top + 1, bottom + 1):
                res.append(self.mat[i][right])
                count += 1

            # move left
            if count >= cells:
                break
            for j in range(right - 1, left - 1, -1):
                res.append(self.mat[bottom][j])
                count += 1

            # move up
            if count >= cells:
                break
            for i in range(bottom - 1, top, -1):
                res.append(self.mat[i][left])
                count += 1

            right -= 1
            left += 1
            top += 1
            bottom -= 1

        return res

    def rotate(self, clockwise=True):
        pass

    def set_zeros(self):
        """
        We use first row and first column to determine if row/column should be zero
        by changing cells [i,0] and [0,j] to True if any cell [i,j] is zero
        We use extra variable col_0 since row 0 and col 0 have common cell
        """
        col_0 = False  # means if we zero column 0 or not
        n = len(self.mat)
        m = len(self.mat[0])

        for i in range(n):
            for j in range(m):
                if self.mat[i][j] == 0:
                    self.mat[i][0] = True
                    self.mat[0][j] = True
                    if j == 0:
                        col_0 = True
        for i in range(n):
            for j in range(m):
                if j == 0 and col_0:
                    self.mat[i][j] = 0
                elif self.mat[0][j] or self.mat[i][0]:
                    self.mat[i][j] = 0


if __name__ == '__main__':
    mat = Matrix(3, 4)
    mat.fill_range(1, 1)
    mat.mat = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    print(mat)
    print()
    mat.set_zeros()
    print(mat)
