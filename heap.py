import random


class Heap:
    def __init__(self, arr: list, type: str):
        self.arr = arr[:]
        self.height = 0
        self.type = type

    def __str__(self):
        if len(self.arr) == 0:
            return '<>'
        nodes = []
        queue = []
        level = 0
        queue.append((0, level))
        while len(queue) > 0:
            curr_index, level = queue.pop(0)
            curr_value = self.arr[curr_index]
            nodes.append((curr_value, level))
            left_index = 2 * curr_index + 1
            right_index = 2 * curr_index + 2

            if left_index < len(self.arr):
                queue.append((left_index, level + 1))
            if right_index < len(self.arr):
                queue.append((right_index, level + 1))

        level_nodes_dict = {}
        max_level = 0
        for node in nodes:
            value, level = node
            max_level = max(max_level, level)
            if level not in level_nodes_dict.keys():
                level_nodes_dict[level] = [value]
            else:
                level_nodes_dict[level].append(value)

        self.height = max_level

        s = ''
        spaces = (2 ** (self.height + 3)) - 1
        for level in level_nodes_dict:
            spaces //= 2
            row = ''
            for i, value in enumerate(level_nodes_dict[level]):
                if i == 0:
                    row += ' ' * spaces
                row += str(value)
                row += ' ' * (2 * spaces + 1)
            s += row + '\n'

        return s

    def is_empty(self) -> bool:
        return len(self.arr) == 0

    def swap(self, i, j):
        self.arr[i], self.arr[j] = self.arr[j], self.arr[i]

    def find_first_non_leaf_index(self):
        for i in range(len(self.arr) - 1, -1, -1):
            if self.get_left(i) < len(self.arr) or self.get_right(i) < len(self.arr):
                return i
        return 0

    def heapify(self) -> None:
        k = self.find_first_non_leaf_index()
        while k >= 0:
            self.sift_down(k)
            k -= 1

    def insert(self, x) -> None:
        pass

    def delete(self, x) -> None:
        pass

    def pop(self) -> int:
        "Pop max/min node"
        self.swap(0, len(self.arr) - 1)
        x = self.arr.pop(-1)
        self.sift_down(0)
        return x

    def peek(self):
        if self.is_empty():
            return None
        return self.arr[0]

    def get_parent(self, index) -> int:
        return (index - 1) // 2

    def get_left(self, index) -> int:
        return 2 * index + 1

    def get_right(self, index) -> int:
        return 2 * index + 2

    def sift_up(self, index):
        while index >= 0:
            parent = self.get_parent(index)
            if type == 'max':
                if self.arr[parent] < self.arr[index]:
                    self.swap(parent, index)
                else:
                    return
            else:
                if self.arr[parent] > self.arr[index]:
                    self.swap(parent, index)
                else:
                    return

    def sift_down(self, index):
        while not self.is_valid_node(index):
            left = self.get_left(index)
            right = self.get_right(index)
            largest = index
            if self.type == 'max':
                if left < len(self.arr) and self.arr[left] > self.arr[largest]:
                    largest = left
                if right < len(self.arr) and self.arr[right] > self.arr[largest]:
                    largest = right
            else:
                if left < len(self.arr) and self.arr[left] < self.arr[largest]:
                    largest = left
                if right < len(self.arr) and self.arr[right] < self.arr[largest]:
                    largest = right
            self.swap(index, largest)
            index = largest

    def is_valid_node(self, index) -> bool:
        bl = True
        br = True
        left = self.get_left(index)
        right = self.get_right(index)

        if left < len(self.arr):
            bl = self.arr[index] >= self.arr[left] if self.type == 'max' else self.arr[index] <= self.arr[left]
        if right < len(self.arr):
            br = self.arr[index] >= self.arr[right] if self.type == 'max' else self.arr[index] <= self.arr[right]

        return bl and br

    def is_valid_heap(self):
        for i, x in enumerate(self.arr):
            if not self.is_valid_node(i):
                return False
        return True


def generate_array() -> list:
    arr = []
    size = random.randint(10, 20)
    for i in range(size):
        x = random.randint(1, 99)
        arr.append(x)
    return arr


def findKthLargest(nums: list, k: int) -> int:
    heap = Heap(nums, 'max')
    heap.heapify()
    for i in range(k - 1):
        x = heap.pop()
    return heap.peek()


if __name__ == '__main__':
    nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]
    k = 4
    print(findKthLargest(nums, k))

    # arr = generate_array()
    # print(arr)
    # heap = Heap(arr, 'min')
    # print(heap)
    # heap.heapify()
    # print(heap)
    # print(heap.is_valid_heap())
    #
    # values = []
    # while not heap.is_empty():
    #     x = heap.pop()
    #     values.append(x)
    #     print(heap)
    # print(values)
