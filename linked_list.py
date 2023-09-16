import heapq


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __str__(self):
        return str(self.val) + ' -> '


class List:
    def __init__(self):
        self.head = None

    def __str__(self):
        s = '| -> '
        node = self.head
        while node:
            s += str(node)
            node = node.next
        s += '||'
        return s

    def is_empty(self):
        return self.head is None

    def length(self):
        node = self.head
        count = 0
        while node:
            node = node.next
            count += 1
        return count

    def append(self, val):
        if self.head is None:
            self.head = ListNode(val)
            return
        node = self.head
        while node.next:
            node = node.next

        new_node = ListNode(val)
        node.next = new_node

    def get_middle(self):
        node1 = self.head
        node2 = self.head
        while node1 and node2:
            node1 = node1.next
            node2 = node2.next
            if node2:
                node2 = node2.next
        return node1.val

    def reverse(self):
        prev = None
        node = self.head
        while node:
            nxt = node.next
            node.next = prev
            prev = node
            node = nxt
        self.head = prev


def append_to_node(head, last_node, val: int):
    if head is None:
        head = ListNode(val)
        return head, head
    new_node = ListNode(val)
    if last_node:
        last_node.next = new_node
    return head, new_node


def mergeSortedLists(head1: ListNode, head2: ListNode) -> ListNode:
    node1 = head1
    node2 = head2
    head3 = None
    last_node3 = None

    while node1 != None and node2 != None:
        if node1.val < node2.val:
            head3, last_node3 = append_to_node(head3, last_node3, node1.val)
            node1 = node1.next
        else:
            head3, last_node3 = append_to_node(head3, last_node3, node2.val)
            node2 = node2.next

    while node1 != None:
        head3, last_node3 = append_to_node(head3, last_node3, node1.val)
        node1 = node1.next

    while node2 != None:
        head3, last_node3 = append_to_node(head3, last_node3, node2.val)
        node2 = node2.next

    return head3


def mergeKLists(lists: list):
    merged_head = None
    last_node = None

    while any([h for h in lists]):
        values = []
        sorted_values = []
        for i, head in enumerate(lists):
            if lists[i]:
                values.append(head.val)
                lists[i] = lists[i].next
        print(values)
        heapq.heapify(values)
        for i in range(len(values)):
            val = heapq.heappop(values)
            sorted_values.append(val)

        # print(sorted_values)
        for val in sorted_values:
            merged_head, last_node = append_to_node(merged_head, last_node, val)

    return merged_head


def print_list(head: ListNode):
    node = head
    values = []
    while node != None:
        values.append(node.val)
        node = node.next
    print(' -> '.join([str(val) for val in values]))


def generate_list(values: list):
    head = None
    last_node = None
    for i in values:
        head, last_node = append_to_node(head, last_node, i)
    return head


def test_heap():
    arr = [5, 7, 9, 1, 3]
    heapq.heapify(arr)
    for i in range(len(arr)):
        x = heapq.heappop(arr)
        print(x)


if __name__ == '__main__':
    head1 = generate_list([1, 4, 5])
    head2 = generate_list([1, 3, 4])
    head3 = generate_list([2, 6])

    lists = [head1, head2, head3]

    head = mergeKLists(lists)
    print_list(head)
