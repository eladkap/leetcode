"""
Disjoint Sets dictionary
Key: vertex
Value: parent
"""


class DisjointSets:
    def __init__(self):
        self.parent_dict = {}
        self.disjoint_sets = {}

    def __str__(self):
        s = ''
        for p in self.parent_dict.keys():
            s += f'{p}: {self.parent_dict[p]}\n'
        for ds in self.disjoint_sets.keys():
            s += f'{ds}: {self.disjoint_sets[ds]}\n'
        return s

    def add_vertex(self, v):
        self.parent_dict[v] = v
        self.disjoint_sets[v] = set([v])

    def union(self, v1: int, v2: int) -> None:
        if v1 > v2:
            v1, v2 = v2, v1
        p1 = self.find(v1)
        for u in self.disjoint_sets[v2]:
            self.parent_dict[u] = p1
            self.disjoint_sets[p1].add(u)
        self.disjoint_sets.pop(v2)

    def find(self, v: int) -> int:
        return self.parent_dict[v]

    def are_connected(self, v1: int, v2: int) -> bool:
        return self.find(v1) == self.find(v2)


if __name__ == '__main__':
    ds = DisjointSets()

    for v in range(0, 10):
        ds.add_vertex(v)

    ds.union(0, 1)
    ds.union(0, 2)
    ds.union(1, 3)
    ds.union(8, 4)
    ds.union(6, 5)
    ds.union(5, 7)
    print(ds)
    print(ds.find(1))

    print(ds.are_connected(0, 3))
    print(ds.are_connected(1, 5))
    print(ds.are_connected(7, 8))

    ds.union(1, 5)
    print(ds)
