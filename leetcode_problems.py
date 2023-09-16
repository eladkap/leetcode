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


if __name__ == '__main__':
    sol = Solution()
    # h = 10
    # m = 2
    # res = sol.calc_angle_hours_minutes(h, m)
    # print(res)
    res = sol.get_pascal_triangle_row(5)
    print(res)
