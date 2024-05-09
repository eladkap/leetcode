class TrieNode(object):
    def __init__(self, value):
        self.value = value
        self.neighbors = {}

    def __str__(self):
        return self.value

    def is_leaf(self):
        return len(self.neighbors) == 0


class Trie(object):
    def __init__(self):
        self.root = TrieNode('Root')

    def show(self):
        queue = []
        queue.append([self.root, 0])
        rows = {}
        while len(queue) > 0:
            node, level = queue.pop(0)

            if level in rows.keys():
                rows[level].append(node.value)
            else:
                rows[level] = [node.value]

            for ch in node.neighbors.keys():
                queue.append([node.neighbors[ch], level + 1])

        for level in rows.keys():
            print(f'{level}: {rows[level]}')

    def insert(self, word: str):
        node = self.root
        for c in word:
            if c in node.neighbors.keys():
                node = node.neighbors[c]
            else:
                new_node = TrieNode(c)
                node.neighbors[c] = new_node
                node = new_node

    def delete(self, word: str):
        pass

    def search(self, word: str):
        node = self.root
        for c in word:
            if c in node.neighbors.keys():
                node = node.neighbors[c]
            else:
                return False
        return node.is_leaf()


if __name__ == '__main__':
    T = Trie()
    words = [
        'ball',
        'bat',
        'beer',
        'bell',
        'tea'
    ]

    for word in words:
        T.insert(word)

    T.show()
    print(T.search('bell'))
    print(T.search('bel'))
