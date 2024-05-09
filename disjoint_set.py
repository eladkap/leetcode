class DisjointSet:
    def __init__(self):
        self.sets = dict()

    def __str__(self):
        return str(self.sets)

    def __contains__(self, x):
        return x in self.sets.keys()

    def get_group(self, x):
        if x in self.sets.keys():
            return self.sets[x]
        return None

    def insert(self, x):
        if x not in self.sets:
            self.sets[x] = set([x])

    def find(self, x):
        if x not in self.sets.keys():
            return None
        for k in self.sets.keys():
            values = self.sets[k]
            if x in values:
                return k

    def union(self, x, y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        for value in self.sets[parent_y]:
            self.sets[parent_x].add(value)
        self.sets[parent_y].clear()


# ds = DisjointSet()
#
# arr = 'abcdefghij'
# for x in arr:
#     ds.insert(x)
#
# ds.union('a', 'b')
# ds.union('b', 'd')
# ds.union('c', 'f')
# ds.union('c', 'i')
# ds.union('j', 'e')
# ds.union('g', 'j')
#
# print(ds)
