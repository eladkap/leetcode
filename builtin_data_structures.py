"""
Built-in data structures usable in Python:
- Dictionary - dict
- Set - set
- Queue - use list
- Stack - use list
- Min Heap - heapq

Implement you own
- Matrix
- Binary tree
- BST - Binary Search Tree (AVL Tree)
- Trie
- Graph (BFS, DFS, Dijkstra)

"""

import queue
import heapq


def run_queue():
    q = queue.Queue()
    for i in range(10):
        q.put(1)


def run_heap():
    arr = [3, 2, 1, 5, 6, 4]
    arr = [-x for x in arr]
    heapq.heapify(arr)
    print(arr)
    k = 2
    n = len(arr)
    for i in range(k - 1):
        heapq.heappop(arr)
    x = -heapq.heappop(arr)
    print(x)


if __name__ == '__main__':
    run_queue()
    run_heap()
