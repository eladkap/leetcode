import math
import random

from disjoint_set import DisjointSet
from linked_list import *


def calc_manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    "Problem  #376"
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def find_closest_coin(curr_position: tuple, coin_positions: list) -> list:
    min_distance = math.inf
    closest_coin = coin_positions[0]
    for coin_position in coin_positions:
        d = calc_manhattan_distance(curr_position, coin_position)
        if d < min_distance:
            min_distance = d
            closest_coin = coin_position

    return closest_coin


def gen_all_possible_subsequences(s: str):
    "Problem #379"
    results = ['']

    # Use nested loop where i is the index of all start positions and j is the index of all end positions
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            word = s[i:j]
            results.append(word)

    return results


def divide(x: int, y: int) -> tuple:
    "Problem #380"
    if x < y:
        return 0, x
    d = 0
    while x >= y:
        x -= y
        d += 1
    r = x
    return d, r


def test_divide(start: int, end: int):
    for i in range(start, end):
        for j in range(start, end):
            expected = (i // j, i % j)
            actual = divide(i, j)
            assert expected == actual


def convert_hex_to_base64(s: str) -> str:
    "Problem #381"
    A_Z = ''.join([chr(i + ord('A')) for i in range(26)])
    a_z = ''.join([chr(i + ord('a')) for i in range(26)])
    digits = ''.join([chr(i + ord('0')) for i in range(10)])
    base64_arr = A_Z + a_z + digits + '+/'

    b_arr = []
    for l in s:
        b = bin(ord(l))[2:]
        b = '0' * (8 - len(b)) + b
        b_arr += b

    k = 0
    base64_letters = []
    while k <= len(b_arr) - 6:
        b = ''.join(b_arr[k:k + 6])
        index = int(b, 2)
        print(index)
        base64_letters.append(base64_arr[index])
        k += 6
    print(b_arr[k:])

    return ''.join(base64_letters)


def are_overlap_or_consecutive(s, w1, w2):
    s1 = s.index(w1)
    e1 = s1 + len(w1)
    s2 = s.index(w2)
    return e1 > s2 or e1 == s2


def embolden(s: str, lst: list, tag: str):
    "Problem #383"
    i = 0
    while i < len(lst):
        word = lst[i]
        start = s.index(word)
        end = start + len(word)
        j = i + 1
        while j < len(lst) and are_overlap_or_consecutive(s, word, lst[j]):
            word_j = lst[j]
            end_j = s.index(word_j) + len(word_j)
            end = end_j
            word = s[start:end_j]
            j += 1
        tagged_word = f'<{tag}>' + word + f'</{tag}>'

        s = s[:start] + tagged_word + s[end:]
        i = j

    return s


def calculate_amount(coins: list, used: list):
    "Problem #384"
    cur_amount = 0
    for i in range(len(coins)):
        cur_amount += coins[i] * used[i]
    return cur_amount


def compute_fewest_num_of_coins(coins: list, amount: int):
    def compute_fewest_num_of_coins_aux(coins: list, amount: int, used: list, options: list):
        curr_amount = calculate_amount(coins, used)

        print(used, curr_amount)

        if curr_amount == amount:
            options.append(used)
            return

        if curr_amount > amount:
            return

        for i in range(len(coins)):
            used[i] += 1
            compute_fewest_num_of_coins_aux(coins, amount, used, options)
            used[i] -= 1

    used = [0] * len(coins)
    coins.sort()
    options = []
    compute_fewest_num_of_coins_aux(coins, amount, used, options)
    return options


def sort_by_frequency(s):
    "Problem #386"
    d = {letter: 0 for letter in s}
    res = sorted(s, key=lambda letter: d[letter] + ord(letter), reverse=True)
    return ''.join(res)


def find_missing_numbers(arr: list, amount: int) -> list:
    """
    Problem #390
    T(n) = O(n)
    S(n) = O(n)
    :param arr:
    :param amount:
    :return:
    """
    missing = []
    d = {num: False for num in arr}
    for num in range(1, amount + 1):
        if num not in d.keys():
            missing.append(num)
    return missing


def calc_island_perimeter(mat: list) -> int:
    """
    Problem #392
    :param mat:
    :return:
    """
    perimeter = 0

    if len(mat) == 0:
        return 0

    rows = len(mat)
    cols = len(mat[0])

    p = dict()
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            p[(i, j)] = 0

    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] == 1:
                # check left
                if j == 0:
                    p[(i, j)] += 1
                elif mat[i][j - 1] == 0:
                    # mat[i][j - 1] = 2
                    p[(i, j)] += 1

                # check right
                if j == cols - 1:
                    p[(i, j)] += 1
                elif mat[i][j + 1] == 0:
                    # mat[i][j + 1] = 2
                    p[(i, j)] += 1

                # check up
                if i == 0:
                    p[(i, j)] += 1
                elif mat[i - 1][j] == 0:
                    # mat[i - 1][j] = 2
                    p[(i, j)] += 1

                # check down
                if i == rows - 1:
                    p[(i, j)] += 1
                elif mat[i + 1][j] == 0:
                    # mat[i + 1][j] = 2
                    p[(i, j)] += 1

    print(p)

    for cell in p.keys():
        perimeter += p[cell]

    return perimeter


def find_largest_range(arr: list) -> tuple:
    """
    Problem #393
    T(n) = O(n)
    S(n) = O(n)
    :param arr:
    :return:
    """
    if len(arr) == 0:
        return ()

    ds = DisjointSet()
    for num in arr:
        ds.insert(num)

    # merge sets
    for num in arr:
        next_num = num + 1
        if next_num in ds:
            ds.union(num, next_num)

    print(ds)

    # find largest group
    largest_group_repr = arr[0]
    largest_size = 0
    for num in ds.sets:
        values = ds.sets[num]
        if len(values) > largest_size:
            largest_group_repr = num
            largest_size = len(values)
    largest_group = ds.get_group(largest_group_repr)
    start = min(largest_group)
    end = max(largest_group)

    return (start, end)


def group_anagrams(arr: list):
    """
    problem: #395
    T(n) = O(n)
    S(n) = O(n)
    :param arr:
    :return:
    """
    d = dict()
    for word in arr:
        sorted_word = ''.join(sorted(word))
        d[sorted_word] = []

    for word in arr:
        sorted_word = ''.join(sorted(word))
        d[sorted_word].append(word)

    return list(d.values())


def preprocess(L: list):
    A = [0] * len(L)
    partial_sum = 0
    for i in range(len(L)):
        A[i] = partial_sum + L[i]
        partial_sum += L[i]
    return A


def sum_smart(L: list, i: int, j: int):
    """
    Problem #400
    """
    if i - 1 < 0:
        return L[j - 1]
    return L[j - 1] - L[i - 1]


def apply_permutation(arr: list, permute: list) -> list:
    """
    Problem #401
    T(n) = O(n)
    S(n) = O(1)
    :param arr:
    :param permute:
    :return:
    """
    p = [arr[index] for index in permute]
    return p


def is_strobogrammatic(num: int):
    """
    Problem #402
    :param num: number
    :return: True/False if number equals to the rotated number bt 180 degress
    """
    rotated180dict = {
        '0': '0',
        '1': '1',
        '2': '-',
        '3': '-',
        '4': '-',
        '5': '-',
        '6': '9',
        '7': '-',
        '8': '8',
        '9': '6'
    }
    num_str1 = str(num)
    num_str2 = ''.join([rotated180dict[digit] for digit in num_str1])[::-1]
    return num_str1 == num_str2


def rand7():
    """
    Problem #403
    Return randomly number between 1 to 7 using rand5
    """

    def rand5():
        return random.randint(1, 5)

    x = rand5() - 1  # x - [0, 4]
    y = 6 * x  # y - [0, 24]
    z = y // 4  # z - [0, 6]
    w = z + 1  # w - [1, 7]
    return w


def problem_507():
    candidate_count_dict = {}
    votes_dict = {}
    lines = []
    with open('voters_507.txt', 'r') as reader:
        lines = reader.readlines()
    for line in lines:
        fields = line.split(',')
        voter_id = fields[0]
        candidate_id = fields[1].strip('\n')
        if voter_id in votes_dict.keys():
            print('Fraud')
            return
        votes_dict[voter_id] = candidate_id

        if candidate_id in candidate_count_dict.keys():
            candidate_count_dict[candidate_id] += 1
        else:
            candidate_count_dict[candidate_id] = 1

    # print(candidate_count_dict)

    # print(votes_dict)

    candidate_ids = list(candidate_count_dict.keys())
    # print(candidate_ids)

    candidate_ids.sort(key=lambda candidate_id: candidate_count_dict[candidate_id], reverse=True)
    # print(candidate_ids)

    return candidate_ids[:3]


def problem_505(arr: list, k: int):
    n = len(arr)
    for i in range(k):
        # rotate one place right
        right_most = arr[n - 1]
        j = n - 1
        while j >= 0:
            arr[j] = arr[j - 1]
            j -= 1
        arr[0] = right_most

    return arr


def problem_500(mat, current, end):
    # reached end
    if current == end:
        return 0

    # visit tile
    mat[current[0], current[1]] = ''

    # go right
    if current[0] + 1 < len(mat):
        current = (current[0] + 1, current[1])
        return 1 + problem_500(mat, current, end)


def find_intersection_node_problem_517(list1: List, list2: List):
    """
    idea: reverse both lists.Then scan both lists from first node and
    find first node that differs. We return one node before it.
    """
    list1.reverse()
    list2.reverse()
    node1 = list1.head
    node2 = list2.head
    prev = None
    while node1 is not None and node1.val == node2.val:
        prev = node1
        node1 = node1.next
        node2 = node2.next
    if prev is None:
        return None
    return prev.val


def problem_516(n: int) -> int:
    """
    find nth seventh number
    7^n
    1 7 (1+7) 49 (1+7+49) 343 (1+7+49+343)
    if n is odd -> return sum of squares till n//2
    if n is even -> return 7 ^ (n//2)
    """
    if n == 1:
        return 1
    if n == 2:
        return 7
    sum_squares = 1
    x = 7
    if n % 2 == 1:
        for i in range(n // 2):
            sum_squares += x
            x *= 7
        return sum_squares
    else:
        return 7 ** (n // 2)


if __name__ == '__main__':
    print('main')
    # mat = [
    #     [False, False, False, False],
    #     [True, True, False, True],
    #     [False, False, False, False],
    #     [False, False, False, False]
    # ]
    # start = (3, 0)
    # end = (0, 0)
    # problem_500(mat, start, end)
    #
    # print(res)

    # list1 = generate_list([3, 7, 8, 10])
    # list2 = generate_list([99, 1, 8, 10])
    # list1.show()
    # list2.show()
    # res = find_intersection_node_problem_517(list1, list2)
    # print(res)

    for n in range(1, 10):
        res = problem_516(n)
        print(f'{n}: {res}')
