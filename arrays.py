import math
import random


def is_sorted(arr: list) -> bool:
    n = len(arr)
    if n == 1:
        return True
    for i in range(n - 1):
        if arr[i] > arr[i + 1]:
            return False

    return True


def bubble_sort(arr: list) -> None:
    n = len(arr)
    for k in range(n - 1, -1, -1):
        for i in range(k):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]


def merge_arrays(arr1, arr2) -> list:
    i1 = 0
    i2 = 0
    i3 = 0
    n1 = len(arr1)
    n2 = len(arr2)
    arr3 = [0] * (len(arr1) + len(arr2))

    while i1 < n1 and i2 < n2:
        if arr1[i1] < arr2[i2]:
            arr3[i3] = arr1[i1]
            i1 += 1
            i3 += 1
        else:
            arr3[i3] = arr2[i2]
            i2 += 1
            i3 += 1

    while i1 < n1:
        arr3[i3] = arr1[i1]
        i1 += 1
        i3 += 1

    while i2 < n2:
        arr3[i3] = arr2[i2]
        i2 += 1
        i3 += 1

    return arr3


def merge_sort(arr: list) -> list:
    if len(arr) == 1:
        return arr
    m = len(arr) // 2
    left_arr = merge_sort(arr[:m])
    right_arr = merge_sort(arr[m:])
    merged_arr = merge_arrays(left_arr, right_arr)
    return merged_arr


def partition(arr: list, low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i


def quick_sort(arr: list) -> None:
    def quick_sort_aux(arr: list, low: int, high: int) -> None:
        if low < high:
            pivot = partition(arr, low, high)
            quick_sort_aux(arr, low, pivot - 1)
            quick_sort_aux(arr, pivot + 1, high)

    quick_sort_aux(arr, 0, len(arr) - 1)


def generate_array(start: int, end: int) -> list:
    arr = []
    size = random.randint(start, end)
    for i in range(size):
        x = random.randint(1, 100)
        arr.append(x)
    return arr


def run_tests():
    arr = generate_array(10, 20)

    print(arr)
    # arr = merge_sort(arr)
    # bubble_sort(arr)
    quick_sort(arr)
    print(arr)
    if is_sorted(arr):
        print('PASS')
    else:
        print('FAIL')


def test_partition(arr: list):
    pivot = partition(arr, 0, len(arr) - 1)
    print(arr)
    print(arr[pivot])
    assert all(arr[i] < arr[pivot] for i in range(pivot))
    assert all(arr[i] >= arr[pivot] for i in range(pivot + 1, len(arr)))


# ------------------------Merge k Sorted Arrays----------------------------------------#

def merge_sorted_arrays(merged_arr: list, s1: int, s2: int, n1: int, n2: int):
    arr1 = merged_arr[s1:s1 + n1]
    arr2 = merged_arr[s2:s2 + n2]
    i1 = 0
    i2 = 0
    i3 = s1

    print(f'Merge arrays: {arr1}, {arr2}')

    while i1 < n1 and i2 < n2:
        if arr1[i1] < arr2[i2]:
            merged_arr[i3] = arr1[i1]
            i1 += 1
            i3 += 1
        else:
            merged_arr[i3] = arr2[i2]
            i2 += 1
            i3 += 1

    while i1 < n1:
        merged_arr[i3] = arr1[i1]
        i1 += 1
        i3 += 1

    while i2 < n2:
        merged_arr[i3] = arr2[i2]
        i2 += 1
        i3 += 1


def merge_k_arrays(arrays: list):
    indices = [0]
    size = 0
    merged_arr = []
    for arr in arrays:
        merged_arr.extend(arr)
        size += len(arr)
        indices.append(size)

    while len(indices) > 2:
        for i in range(0, len(indices) - 1, 2):
            if i + 2 >= len(indices):
                continue

            j = indices[i]
            s1 = j
            s2 = indices[i + 1]
            n1 = indices[i + 1] - indices[i]
            n2 = indices[i + 2] - indices[i + 1]

            merge_sorted_arrays(merged_arr, s1, s2, n1, n2)

        indices = indices[0:-1:2] + [indices[-1]]

        print(merged_arr)

    return merged_arr


def test_merge_k_arrays():
    k = 10
    arrays = []
    for i in range(k):
        arr = generate_array(4, 7)
        arr.sort()
        arrays.append(arr)
        print(arr)

    merged_arr = merge_k_arrays(arrays)
    print(merged_arr)
    print(f'total size: {len(merged_arr)}')
    print(is_sorted(merged_arr))
    # assert is_sorted(merged_arr)


if __name__ == '__main__':
    test_merge_k_arrays()
