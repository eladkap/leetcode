def minimalHeaviestSetA(arr):
    n = len(arr)
    sum_all = sum(arr)
    print(sum_all)
    arr.sort(reverse=True)
    A = []
    i = 0
    sumA = 0
    while i < n and sumA <= sum_all / 2:
        A.append(arr[i])
        sumA += arr[i]
        i += 1
    return A[::-1]


def count_items(s, start, end):
    items = 0

    # find most left pipe
    i = start
    while i <= end and s[i] == '*':
        i += 1
    left_pipe = i

    # find most right pipe
    i = end
    while i >= start and s[i] == '*':
        i -= 1
    right_pipe = i

    if left_pipe >= right_pipe:
        return 0

    i = left_pipe
    while i <= right_pipe:
        if s[i] == '*':
            items += 1
        i += 1

    return items


def numberOfItems(s, startIndices, endIndices):
    res = []
    for i in range(len(startIndices)):
        start = startIndices[i]
        end = endIndices[i]
        res.append(count_items(s, start - 1, end - 1))
    return res


def maximumBookCopies(portalUpdate):
    res = []
    d = {}

    max_copies = 0
    max_book_ids = set()
    copies_set = set()

    for book_id in portalUpdate:
        x = book_id
        # book added from inventory
        if book_id > 0:
            if book_id in d.keys():
                d[book_id] += 1
            else:
                d[book_id] = 1

            copies_set.add(d[book_id])

            if d[book_id] > max_copies:
                max_copies = d[book_id]
                max_book_ids.add(book_id)

        # book removed from inventory
        else:
            book_id *= -1
            if book_id in d.keys():
                d[book_id] -= 1

            if book_id in max_book_ids:
                max_book_ids.remove(book_id)
                if len(max_book_ids) == 1 and book_id in max_book_ids:
                    copies_set.remove(max_copies)
                    max_copies -= 1

        # print(d, max_copies, max_book_ids)
        print(x, d, copies_set, max_book_ids, max_copies)

        res.append(max_copies)

    return res


def find_p(priority):
    max_p = 0
    d = {}
    for p in priority:
        if p in d.keys():
            d[p] += 1
            return p
        else:
            d[p] = 1
    # for p in d.keys():
    #     if d[p] >= 2:
    #         return
    # max_p = max(max_p, p)
    return max_p


def getPrioritiesAfterExecution(priority):
    while True:
        p = find_p(priority)

        print(f'p={p}')

        if p == 0:
            break

        processes = []
        for i in range(len(priority)):
            if priority[i] == p:
                processes.append(i)
                if len(processes) == 2:
                    break
        print(processes)

        i = processes[0]
        if len(processes) == 2:
            priority[processes[1]] //= 2
            priority.pop(i)
        else:
            break

        print(priority)
        print('----------')

    return priority


def is_power_of_two(num: int) -> bool:
    return num & (num - 1) == 0 and num > 0


def get_max_sub_array_with_powers_of_two(arr: list):
    max_length = 0
    start_index = 0
    end_index = 0
    n = len(arr)

    i = 0
    while i < n:
        j = i
        count = 0
        while j < n and is_power_of_two(arr[j]):
            j += 1
            count += 1
        if count > max_length:
            max_length = count
            start_index = j - count
            end_index = j - 1
        i += 1

    return arr[start_index:end_index + 1]


def map_letters_to_digits(word: str):
    digits_dict = {k: [] for k in range(9)}
    count_dict = {}
    for letter in word:
        if letter in count_dict.keys():
            count_dict[letter] += 1
        else:
            count_dict[letter] = 1

    sorted_word = sorted(list(set(word)), key=lambda letter: count_dict[letter], reverse=True)

    digit = 0
    for letter in sorted_word:
        digits_dict[digit].append(letter)
        digit = (digit + 1) % 9

    return digits_dict


def gen_word():
    w = ''
    for i in range(26):
        letter = chr(ord('a') + i)
        w += (letter * (i + 1))
    return w


def get_longest_sorted_subarray(arr: list):
    n = len(arr)
    i = 0
    max_len = 0
    s = 0
    e = 0
    while i < n:
        j = i
        count = 1
        while j + 1 < n and arr[j] <= arr[j + 1]:
            j += 1
            count += 1
        j += 1

        if count > max_len:
            max_len = count
            s = j - count
            e = j - 1
        i += 1
    return arr[s:e + 1], max_len


def are_there_sum_to_k(arr: list, k: int) -> list:
    d = {}
    for num in arr:
        y = k - num
        if y in d.keys():
            return k - y, y
        d[num] = y
    return None


def create_count_dict(s: str) -> dict:
    d = {}
    for l in s:
        if l in d.keys():
            d[l] += 1
        else:
            d[l] = 1
    return d


def get_max_times_gen_string_from_log(s: str, log: str):
    s_dict = create_count_dict(s)
    log_dict = create_count_dict(log)
    div_dict = {}
    for l in s_dict.keys():
        if l in log_dict.keys():
            div_dict[l] = log_dict[l] // s_dict[l]
    min_div_value = len(log)
    print(s_dict)
    print(log_dict)
    for l in div_dict.keys():
        min_div_value = min(min_div_value, div_dict[l])
    return min_div_value


def sum_squares(num: int) -> int:
    return sum([int(d) ** 2 for d in str(num)])


def is_happy_number(num: int) -> bool:
    s = set([num])
    while num != 1:
        num = sum_squares(num)
        if num in s:
            return False
        s.add(num)
    return True


def compress_string(s: str):
    n = len(s)
    t = []
    i = 0
    while i < n:
        j = i
        count = 0
        while j < n and s[j] == s[i]:
            j += 1
            count += 1
        if count > 1:
            t.extend([s[i], str(count)])
        else:
            t.append(s[i])
        i = j
    return ''.join(t)


def mult_besides_curr(arr: list) -> list:
    n = len(arr)
    left_arr = [1] * n
    right_arr = [1] * n

    m = 1
    for i in range(n):
        left_arr[i] = m
        m *= arr[i]

    m = 1
    i = n - 1
    while i >= 0:
        right_arr[i] = m
        m *= arr[i]
        i -= 1

    print(left_arr)
    print(right_arr)

    res = [1] * n
    for i in range(n):
        res[i] = left_arr[i] * right_arr[i]
    return res


if __name__ == '__main__':
    arr = [1, 2, 3, 4]
    res = mult_besides_curr(arr)
    print(res)
