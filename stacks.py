class Stack(object):
    def __init__(self):
        self.stack = []
        self.minimum = None

    def __str__(self):
        return str(self.stack)

    def push(self, x):
        self.stack.append(x)
        if not self.minimum:
            self.minimum = x
        else:
            self.minimum = min(self.minimum, x)

    def top(self):
        if len(self.stack) == 0:
            raise Exception('stack is empty')
        return self.stack[-1]

    def pop(self):
        if len(self.stack) == 0:
            raise Exception('stack is empty')
        t = self.stack[-1]
        self.stack.pop(-1)
        return t

    @property
    def is_empty(self):
        return len(self.stack) == 0

    def get_min(self):
        return self.minimum


class Queue(object):
    def __init__(self):
        self.S = Stack()

    def enqueue(self, x):
        pass

    def front(self):
        pass

    def dequeue(self):
        pass

    @property
    def is_empty(self):
        return self.S.is_empty


if __name__ == '__main__':
    S = Stack()
    print(S.is_empty)

    for x in range(1, 6):
        S.push(x)
        assert S.top() == x
        print(S)
        assert S.get_min() == min(S.stack)

    x = 5
    for i in range(5):
        t = S.pop()
        assert t == x
        x -= 1
        print(S)

    print(S.is_empty)
