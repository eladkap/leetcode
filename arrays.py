import random


def is_sorted(arr: list) -> bool:
    if len(arr) <= 1:
        return True

    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False

    return True


def bubble_sort(arr: list) -> None:
    n = len(arr)
    for k in range(n - 1, -1, -1):
        for i in range(k):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]


def merge_arrays(arr1: list, arr2: list) -> list:
    n1 = len(arr1)
    n2 = len(arr2)
    n3 = n1 + n2
    arr3 = [0] * n3

    i1 = 0
    i2 = 0
    i3 = 0
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


def merge_sort(arr: list) -> None:
    def merge_sorted_arrays(arr1: list, arr2: list, arr3: list) -> None:
        n1 = len(arr1)
        n2 = len(arr2)
        i = 0
        j = 0
        k = 0
        while i < n1 and j < n2:
            if arr1[i] < arr2[j]:
                arr3[k] = arr1[i]
                i += 1
                k += 1
            else:
                arr3[k] = arr2[j]
                j += 1
                k += 1
        while i < n1:
            arr3[k] = arr1[i]
            i += 1
            k += 1
        while j < n2:
            arr3[k] = arr2[j]
            j += 1
            k += 1

    n = len(arr)
    if n == 1:
        return

    middle = n // 2

    left_arr = arr[:middle]
    right_arr = arr[middle:]

    merge_sort(left_arr)
    merge_sort(right_arr)

    merge_sorted_arrays(left_arr, right_arr, arr)


def partition(arr: list) -> int:
    print(arr)
    n = len(arr)
    pivot_index = n - 1
    print(arr, n)
    pivot = arr[pivot_index]
    print(f'pivot: {pivot_index}, len: {n}')
    tmp = [0] * n
    left = 0
    right = n - 2
    for i in range(n):
        if arr[i] <= pivot:
            tmp[left] = arr[i]
            left += 1
        else:
            tmp[right] = arr[i]
            right -= 1

    arr[right] = pivot

    for i in range(n):
        arr[i] = tmp[i]

    return right


def quick_sort(arr: list) -> None:
    n = len(arr)

    if n == 1:
        return

    pivot_index = partition(arr)
    left_arr = arr[:pivot_index]
    right_arr = arr[pivot_index + 1:]
    quick_sort(left_arr)
    quick_sort(right_arr)


def generate_array() -> list:
    size = random.randint(10, 20)
    arr = []
    for i in range(size):
        x = random.randint(0, 100)
        arr.append(x)
    return arr


def bubble_sort_test():
    arr = generate_array()

    print(arr)
    bubble_sort(arr)
    print(arr)
    print('PASS' if is_sorted(arr) else 'FAIL')
    assert is_sorted(arr)


def merge_sorted_arrays_test():
    arr1 = generate_array()
    arr2 = generate_array()
    bubble_sort(arr1)
    bubble_sort(arr2)
    print(arr1)
    print(arr2)
    arr3 = merge_arrays(arr1, arr2)
    print(arr3)
    print(is_sorted(arr3))
    assert is_sorted(arr3)


def merge_sort_test():
    arr = generate_array()
    print(arr)
    merge_sort(arr)
    print(arr)
    print('PASS' if is_sorted(arr) else 'FAIL')
    assert is_sorted(arr)


def quick_sort_test():
    arr = generate_array()

    print(arr)
    quick_sort(arr)
    print(arr)
    print('PASS' if is_sorted(arr) else 'FAIL')
    assert is_sorted(arr)


if __name__ == '__main__':
    # merge_sorted_arrays_test()
    # merge_sort_test()
    quick_sort_test()
