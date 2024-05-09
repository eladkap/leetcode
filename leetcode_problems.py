class Solution:
    def is_well_structured(self, s: str):
        BRACKETS = {
            '{': '}',
            '[': ']',
            '(': ')'
        }

        stack = []
        for x in s:
            if x in BRACKETS.keys():
                stack.append(x)
            elif x in BRACKETS.values():
                if len(stack) == 0:
                    return False
                y = stack[-1]
                if BRACKETS[y] != x:
                    return False
                stack.pop(-1)
        return len(stack) == 0

    def calc_angle_hours_minutes(self, h: int, m: int) -> float:
        a = abs(5 * h - m) / 60 * 360
        b = m / 60 * 30
        angle = a + b
        return min(angle, 360 - angle)

    def pascal_triangle(self, n: int):
        rows = [[1]]
        for i in range(1, n):
            row = [0] + rows[-1] + [0]
            for j in range(len(rows[-1]) + 1):
                row[j] = row[j] + row[j + 1]
            rows.append(row[:-1])
        return rows

    def get_pascal_triangle_row(self, n: int):
        row = [1]
        for i in range(0, n):
            row = [0] + row + [0]
            print(row)
            for j in range(len(row) - 1):
                row[j] = row[j] + row[j + 1]
            row.pop(-1)
        return row

    def isMatch(self, s: str, p: str) -> bool:
        dp = {}

        def dfs(i, j):
            if (i, j) in dp.keys():
                return dp[(i, j)]

            if i >= len(s) and j >= len(p):
                return True

            if j >= len(p):
                return False

            match = i < len(s) and (s[i] == p[j] or p[j] == '.')
            # pattern has [a-z]* or .*
            if j + 1 < len(p) and p[j + 1] == '*':
                # do not use *
                dp[(i, j)] = (dfs(i, j + 2) or  # do not use *
                              match and dfs(i + 1, j))  # use *

                return dp[(i, j)]

            # pattern has just letter [a-z] or dot .
            if match:
                dp[(i, j)] = dfs(i + 1, j + 1)
                return dp[(i, j)]

            dp[(i, j)] = False
            return dp[(i, j)]

        return dfs(0, 0)

    def isMatch2(self, s: str, p: str) -> bool:
        p=p.replace('*', '?*')
        m = len(s) + 1
        n = len(p) + 1

        s = ' ' + s
        p = ' ' + p

        # initialize dp matrix with size m x n
        dp = []
        for i in range(m):
            row = [False] * n
            dp.append(row)

        # empty string and empty pattern match
        dp[0][0] = True

        # first row - empty string
        for j in range(1, n):
            if p[j] != '*':
                dp[0][j] = False
            else:
                dp[0][j] = dp[0][j - 2]  # 0 times letter can match empty string so we can ignore <CH>*

        # for column - empty pattern
        for i in range(1, m):
            dp[i][0] = False

        for i in range(1, m):
            for j in range(1, n):
                if p[j] == '*':
                    dp[i][j] = (dp[i][j - 2] or  # do not use *
                                (p[j - 1] == s[i] or p[j - 1] == '.'))  # use *
                    continue

                if s[i] == p[j] or p[j] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = False


        for row in dp:
            print(row)

        return dp[m - 1][n - 1]


if __name__ == '__main__':
    sol = Solution()

    s = 'aa'
    p = '*'



    res = sol.isMatch(s, p)
    print(res)
