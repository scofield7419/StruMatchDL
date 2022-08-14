import torch
import collections.abc
import collections

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

class Vocabulary(object):
    def __init__(self):
        self.frozen = False
        self.values = []
        self.indices = {}
        self.counts = collections.defaultdict(int)

    @property
    def size(self):
        return len(self.values)

    def value(self, index):
        assert 0 <= index < len(self.values)
        return self.values[index]

    def index(self, value):
        if not self.frozen:
            self.counts[value] += 1

        if value in self.indices:
            return self.indices[value]

        elif not self.frozen:
            self.values.append(value)
            self.indices[value] = len(self.values) - 1
            return self.indices[value]

        else:
            raise ValueError("Unknown value: {}".format(value))

    def count(self, value):
        return self.counts[value]

    def freeze(self):
        self.frozen = True


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal(param)
            weight.append(param)
    return torch.cat(weight)


class Tree(object):
    def __init__(self, index):
        self.parent = None
        self.is_left = False
        self.index = index
        self.left_children = list()
        self.left_num = 0
        self.right_children = list()
        self.right_num = 0
        self._depth = -1
        self.order = []

    def add_left(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        child.is_left = True
        self.left_children.append(child)
        self.left_num += 1

    def add_right(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        child.is_left = False
        self.right_children.append(child)
        self.right_num += 1

    def size(self):  # compute the total size of the Tree
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.left_num):
            count += self.left_children[i].size()
        for i in range(self.right_num):
            count += self.right_children[i].size()
        self._size = count
        return self._size

    def depth(self):  # compute the depth of the Tree
        if self._depth > 0:
            return self._depth
        count = 0
        if self.left_num + self.right_num > 0:
            for i in range(self.left_num):
                child_depth = self.left_children[i].depth()
                if child_depth > count:
                    count = child_depth
            for i in range(self.right_num):
                child_depth = self.right_children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def traverse(self):  # traverse the Tree
        if len(self.order) > 0:
            return self.order

        for i in range(self.left_num):
            left_order = self.left_children[i].traverse()
            self.order.extend(left_order)
        for i in range(self.right_num):
            right_order = self.right_children[i].traverse()
            self.order.extend(right_order)
        self.order.append(self.index)  # append the root
        return self.order


def creatTree(heads):
    tree = []
    # current sentence has already been numberized [form, head, rel]
    root = None
    for idx, head in enumerate(heads):
        tree.append(Tree(idx))

    for idx, head in enumerate(heads):
        if head == -1:  # -1 mszhang, 0 kiro
            root = tree[idx]
            continue
        if head < 0:
            print('error: multi roots')
        if head > idx:
            tree[head].add_left(tree[idx])
        if head < idx:
            tree[head].add_right(tree[idx])
        if head == idx:
            print('error: head is it self.')

    return root, tree



class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index)) # index if the left-side bound
            index = children[-1].right # right-side bound

        return InternalParseNode(tuple(sublabels), children)

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word) # index: left-side bound

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    # convert from InternalParseNode back to InternalTreebankNode
    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees