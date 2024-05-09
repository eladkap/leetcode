class TreeNode:
    def __init__(self, value, index):
        self.value = value
        self.index = index
        self.left = None
        self.right = None
        self.parent = None

    def __str__(self):
        return f'({self.value}, {self.left}, {self.right})'

    def is_leaf(self):
        return not self.left and not self.right


class BinaryTree:
    def __init__(self, values: list):
        self.root = None
        nodes = []
        for i, value in enumerate(values):
            node = TreeNode(value, i)
            nodes.append(node)

        for i, node in enumerate(nodes):
            left_index = 2 * i + 1
            if left_index < len(nodes):
                node.left = nodes[left_index]
                node.left.parent = node

            right_index = 2 * i + 2
            if right_index < len(nodes):
                node.right = nodes[right_index]
                node.right.parent = node

            if i == 0:
                self.root = node

    def get_preorder_scan(self):
        def preorder_scan_aux(root: TreeNode, result: list):
            if not root:
                return
            result.append(root.value)
            preorder_scan_aux(root.left, result)
            preorder_scan_aux(root.right, result)

        result = []
        preorder_scan_aux(self.root, result)
        return result

    def get_inorder_scan(self):
        def inorder_scan_aux(root: TreeNode, result: list):
            if not root:
                return
            inorder_scan_aux(root.left, result)
            result.append(root.value)
            inorder_scan_aux(root.right, result)

        result = []
        inorder_scan_aux(self.root, result)
        return result

    def get_postorder_scan(self):
        def postorder_scan_aux(root: TreeNode, result: list):
            if not root:
                return
            postorder_scan_aux(root.left, result)
            postorder_scan_aux(root.right, result)
            result.append(root.value)

        result = []
        postorder_scan_aux(self.root, result)
        return result

    def get_level_order_scan(self):
        queue = []
        queue.append(self.root)
        result = []
        while len(queue) > 0:
            node = queue.pop(0)
            result.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    def get_height(self):
        def get_height_aux(root):
            if not root:
                return 0
            left_height = get_height_aux(root.left)
            right_height = get_height_aux(root.right)
            return max(left_height, right_height) + 1

        return get_height_aux(self.root)

    def mirror(self):
        def mirror_aux(root):
            if not root:
                return None
            left_node = mirror_aux(root.left)
            right_node = mirror_aux(root.right)
            node = TreeNode(root.value, root.index)
            node.left = right_node
            node.right = left_node
            return node

        mirror_tree_root = mirror_aux(self.root)
        mirror_tree = BinaryTree([])
        mirror_tree.root = mirror_tree_root
        return mirror_tree

    def get_paths(self):
        def get_paths_aux(root: TreeNode, path: list, paths: list):
            if not root:
                return

            if root.is_leaf():
                path.append(root.value)
                paths.append(path[:])
                return

            path.append(root.value)
            get_paths_aux(root.left, path, paths)
            path.pop(-1)
            get_paths_aux(root.right, path, paths)
            path.pop(-1)

        paths = []
        path = []
        get_paths_aux(self.root, path, paths)
        return paths

    def find_path_with_sum(self, k: int):
        "Problem #394"
        def find_path_with_sum_aux(root: TreeNode, k: int, path: list):
            if not root:
                return False

            if root.is_leaf():
                path.append(root.value)
                if sum(path) == k:
                    return True, path[:]
                else:
                    return False

            path.append(root.value)
            res_left = find_path_with_sum_aux(root.left, k, path)
            path.pop(-1)
            res_right = find_path_with_sum_aux(root.right, k, path)
            path.pop(-1)
            return res_left or res_right

        path = []
        res = find_path_with_sum_aux(self.root, k, path)
        return res


if __name__ == '__main__':
    arr = list(range(1, 16))
    tree = BinaryTree(arr)
    # print(tree.get_preorder_scan())
    # print(tree.get_inorder_scan())
    # print(tree.get_postorder_scan())
    # print(tree.get_level_order_scan())
    # print(tree.get_height())
    # mirror_tree = tree.mirror()
    # print(mirror_tree.get_level_order_scan())
    # paths = tree.get_paths()
    # for path in paths:
    #     print(path)
    print(tree.find_path_with_sum(18))
