VISITED = '!'


def scan_ship_dfs(B: list, i: int, j: int) -> int:
    n = len(B)
    m = len(B[0])

    count = 0
    stack = [[i, j]]
    while len(stack) > 0:
        v = stack.pop()
        i, j = v[0], v[1]

        if B[i][j] == VISITED:
            continue

        if B[i][j] == '.':
            continue

        # visit cell
        B[i][j] = VISITED
        count += 1

        # go left
        if j - 1 >= 0:
            stack.append([i, j - 1])

        # go right
        if j + 1 < m:
            stack.append([i, j + 1])

        # go up
        if i - 1 >= 0:
            stack.append([i - 1, j])

        # go down
        if i + 1 < n:
            stack.append([i + 1, j])

    return count


def scan_ship(B: list, i: int, j: int, count_ref) -> None:
    n = len(B)
    m = len(B[0])

    if i < 0 or j < 0 or i >= n or j >= m:
        return

    if B[i][j] == VISITED or B[i][j] == '.':
        return

    # visit cell
    B[i][j] = VISITED
    count_ref[0] += 1

    scan_ship(B, i, j - 1, count_ref)
    scan_ship(B, i, j + 1, count_ref)
    scan_ship(B, i - 1, j, count_ref)
    scan_ship(B, i + 1, j, count_ref)


def find_ships(B: list) -> dict:
    d = {
        'P': 0,
        'S': 0,
        'D': 0
    }
    n = len(B)
    m = len(B[0])

    for i in range(n):
        for j in range(m):
            if B[i][j] == '.' or B[i][j] == VISITED:
                continue
            count_ref = [0]
            scan_ship(B, i, j, count_ref)
            count = count_ref[0]
            if count == 1:
                d['P'] += 1
            elif count == 2:
                d['S'] += 1
            elif count == 3:
                d['D'] += 1

    return d


if __name__ == '__main__':
    b = [
        '##......',
        '#.......',
        '..###..#',
        '#.....#.',
        '...#..#.'
    ]

    B = []
    for line in b:
        B.append(list(line))

    d = find_ships(B)
    print(d)
