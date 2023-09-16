class Vertex(object):
    def __init__(self, value: str):
        self.value = value
        self.visited = False

    def __str__(self):
        return self.value


class Edge(object):
    def __init__(self, v1: Vertex, v2: Vertex, weight: int):
        self.v1 = v1
        self.v2 = v2
        self.weight = weight

    def __str__(self):
        return f'{self.v1}-{self.v2}'


class Graph(object):
    def __init__(self):
        self.vertices = dict()
        self.edges = []

    def __str__(self):
        s = ''
        s += f"V=[{','.join([str(v) for v in self.vertices])}]\n"
        s += f"E=[{','.join([str(e) for e in self.edges])}]\n"
        return s

    def add_vertex(self, v: Vertex):
        self.vertices[v] = set()

    def add_vertices(self, vertices: list):
        for vertex in vertices:
            self.add_vertex(vertex)

    def add_edge(self, edge: Edge):
        self.vertices[edge.v1].add(edge.v2)
        self.vertices[edge.v2].add(edge.v1)
        self.edges.append(edge)

    def show(self):
        for v in self.vertices:
            print(f"{v}: [{','.join([str(u) for u in self.vertices[v]])}]")

    def set_all_visited(self, visited: bool):
        for v in self.vertices:
            v.visited = visited

    def bfs(self, start: Vertex):
        queue = []
        queue.append(start)
        res = []
        self.set_all_visited(False)
        while len(queue) > 0:
            v = queue.pop(0)
            if v.visited:
                continue
            v.visited = True
            res.append(str(v))
            for u in self.vertices[v]:
                queue.append(u)
        return res

    def dfs(self, start: Vertex):
        stack = []
        stack.append(start)
        res = []
        self.set_all_visited(False)
        while len(stack) > 0:
            v = stack.pop(-1)
            if v.visited:
                continue
            v.visited = True
            res.append(str(v))
            for u in self.vertices[v]:
                stack.append(u)
        return res


if __name__ == '__main__':
    G = Graph()

    a = Vertex('A')
    b = Vertex('B')
    c = Vertex('C')
    d = Vertex('D')
    e = Vertex('E')
    f = Vertex('F')
    g = Vertex('G')

    ac = Edge(a, c, 1)
    ab = Edge(a, b, 1)
    cd = Edge(c, d, 2)
    bd = Edge(b, d, 3)
    be = Edge(b, e, 2)
    de = Edge(d, e, 3)
    ef = Edge(e, f, 4)
    eg = Edge(e, g, 2)

    G.add_vertices([a, b, c, d, e, f, g])

    G.add_edge(ab)
    G.add_edge(ac)
    G.add_edge(bd)
    G.add_edge(cd)
    G.add_edge(be)
    G.add_edge(de)
    G.add_edge(ef)
    G.add_edge(eg)

    G.show()

    res = G.bfs(a)
    print(res)

    res = G.dfs(a)
    print(res)
