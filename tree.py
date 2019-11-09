
class Node:
    attrib = None  # its own attrib
    value  = None   # value of parent_attrib
    label  = None   # label if this is leaf
    left   = None  # node, if False
    right  = None  # node, if True

    n_label1 = None
    n_label2 = None
    is_leaf = False

# def stuff(n: Node):
#
#     n.attrib = 3
#
# n1 = Node()
# n1.attrib = 1
# stuff(n1)
#
# print(n1.attrib)

