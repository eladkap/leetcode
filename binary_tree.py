class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        left_value = self.left.value if self.left else '-'
        right_value = self.right.value if self.right else '-'
        return f'[{self.value}, {left_value}, {right_value}]'


class BinaryTree:
    def __init__(self, root: TreeNode):
        self.root = root

    def perorder_scan(self):
        def preorder_scan_aux(root):
            if not root:
                return
            print(root)
            preorder_scan_aux(root.left)
            preorder_scan_aux(root.right)

        preorder_scan_aux(self.root)

    def inorder_scan(self):
        def inorder_scan_aux(root):
            if not root:
                return
            inorder_scan_aux(root.left)
            print(root)
            inorder_scan_aux(root.right)

        inorder_scan_aux(self.root)

    def postorder_scan(self):
        def postorder_scan_aux(root):
            if not root:
                return
            postorder_scan_aux(root.left)
            postorder_scan_aux(root.right)
            print(root)

        postorder_scan_aux(self.root)

    def level_scan(self):
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            if not node:
                continue
            print(node)
            queue.append(node.left)
            queue.append(node.right)

    def invert(self):
        def invert_aux(root):
            if not root:
                return None
            new_node = TreeNode(root.value)
            new_node.left = invert_aux(root.right)
            new_node.right = invert_aux(root.left)
            return new_node

        mroot = invert_aux(self.root)
        return BinaryTree(mroot)

    def get_height(self):
        def get_height(root) -> int:
            if not root:
                return 0
            left_height = get_height(root.left)
            right_height = get_height(root.right)
            return 1 + max(left_height, right_height)

        return get_height(self.root)


if __name__ == '__main__':
    a = TreeNode(1)
    b = TreeNode(2)
    c = TreeNode(3)
    d = TreeNode(4)
    e = TreeNode(5)
    f = TreeNode(6)
    g = TreeNode(7)
    h = TreeNode(8)
    i = TreeNode(9)
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.left = f
    c.right = g
    e.left = h
    e.right = i
    tree = BinaryTree(a)

    print(tree.get_height())
