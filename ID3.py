from typing import List, Set, Dict, Tuple
import pandas
import math
import time
import sys
import warnings

# local file
from tree import Node
import preprocess

label           = 'salaryLevel'
index_col       = 'index_col'  # use index_col to speedup
label1          = True   # ' >50K'
label2          = False  # ' <=50K'
dom_label = label2


def log2(num: float) -> float:
    return math.log(num, 2)

# entropy = uncertainty level of data
def get_entropy(S: pandas.DataFrame) -> float:

    # only 2 label: T/F
    p1 = len(S[ S[index_col] == label1 ]) / len(S)
    p2 = 1 - p1

    if p1 == 0 or p2 == 0:
        return 0

    entropy = -p1*log2(p1) - p2*log2(p2)
    return entropy

def get_info_gain(S: pandas.DataFrame, A, entropy_s=-1.0) -> float:
    """

    :param S:
    :param A:
    :param entropy_s: compute first, to save time
    :return: info gain
    """
    if entropy_s == -1:
        entropy_s = get_entropy(S)

    num_s = len(S)
    attrib_unique = S[A].unique()
    sum_entr = 0

    for value in attrib_unique:

        # Sv = set_ex where attrib A=v
        Sv = S[ S[A] == value ]
        num_Sv = len(Sv)

        sum_entr += (num_Sv/num_s) * get_entropy(Sv)

    info_gain = entropy_s - sum_entr

    return info_gain

def get_num_nodes(node: Node):
    if node is None:
        return 0

    if node.is_leaf:
        return 1

    return 1 + get_num_nodes(node.left) + get_num_nodes(node.right)

def get_height(root: Node) -> int:

    if root is None:
        return 0

    if root.is_leaf:
        return 0

    return 1 + max(get_height(root.left), get_height(root.right))


class DecisionTree:

    train_set: pandas.DataFrame = None
    valid_set: pandas.DataFrame = None
    test_set : pandas.DataFrame = None
    root: Node = None

    # all dataset should be preprocessed before input
    def __init__(self, train_set, valid_set=None, test_set=None, max_depth=15):
        self.a = 0
        self.train_set = train_set
        self.test_set  = test_set
        self.valid_set = valid_set
        self.max_accuracy = -1  # max_accuracy so far, based on valid_set
        self.max_depth = 0

        # for debug
        self.n_nodes = 0
        self.n_leaves = 0

    def set_valid_set(self, valid_set):
        self.valid_set = valid_set

    def predict(self, row: pandas.core.series.Series):

        node = self.root
        while not node.is_leaf:

            # what if tree has attrib, that row doesn't have
            try:
                value = row[node.attrib]
            except KeyError:
                value = False

            # turn, True->right, False->left
            if value:
                node = node.right
            else:
                node = node.left

        return node.label

    def build_normal(self):
        # build tree
        attrib_list = list(self.train_set.columns)
        attrib_list.remove(label)
        attrib_list.remove(index_col)

        t1 = time.time()
        self.root = self.ID3(X=self.train_set, y=attrib_list)
        print("Train time: %.4f seconds" % (time.time() - t1))

    def build_max_depth(self, max_depth):
        attrib_list = list(self.train_set.columns)
        attrib_list.remove(label)
        attrib_list.remove(index_col)

        self.max_depth = max_depth
        t1 = time.time()
        self.root = self.ID3_depth(X=self.train_set, y=attrib_list, depth=0)
        print("Train time: %.4f seconds" % (time.time() - t1))

    def ID3(self, X: pandas.DataFrame, y: List):

        # init node
        node: Node = Node()
        self.n_nodes += 1
        if self.n_nodes % 20 == 0:
            print('n_node =', self.n_nodes)

        # -----------
        # condition to return node
        # -----------

        # check x empty
        if len(X) == 0:
            node.is_leaf = True
            node.label = dom_label
            return node

        # count label1, label2
        n_label1 = len(X[X[index_col] == label1])
        n_label2 = len(X) - n_label1
        node.n_label1 = n_label1
        node.n_label2 = n_label2

        # check if all rows only have single label
        if n_label1 == 0 or n_label2 == 0:
            node.label = label1 if n_label1 > n_label2 else label2
            node.is_leaf = True
            return node

        # check if 1 attribs left
        # select most common label
        if len(y) == 1:
            node.label = label1 if n_label1 > n_label2 else label2
            node.is_leaf = True
            return node

        # -----------
        # create branches for node
        # since all attrib binary => only 2 value: T/F
        # -----------

        # find best attrib
        max_attrib = -1
        max_info_gain = -1
        entr = get_entropy(S=X)
        for attrib in y:
            info_gain = get_info_gain(S=X, A=attrib, entropy_s=entr)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_attrib = attrib

        node.attrib = max_attrib

        # new label list
        new_y = y[:]
        new_y.remove(max_attrib)

        # left (False)
        X_left = X[X[max_attrib] == label2]
        node.left = self.ID3(X_left, new_y)

        # right (True)
        X_right = X.drop(X_left.index)
        node.right = self.ID3(X_right, new_y)

        return node

    def ID3_depth(self, X: pandas.DataFrame, y: List, depth=0):
        # init node
        node: Node = Node()
        self.n_nodes += 1
        if self.n_nodes % 20 == 0:
            print('n_node =', self.n_nodes)

        # -----------
        # condition to return node
        # -----------

        # check x empty
        if len(X) == 0:
            node.is_leaf = True
            node.label = dom_label
            return node

        # count label1, label2
        n_label1 = len(X[X[index_col] == label1])
        n_label2 = len(X) - n_label1
        node.n_label1 = n_label1
        node.n_label2 = n_label2

        # if exceed max_depth
        if depth >= self.max_depth:
            node.label = label1 if n_label1 > n_label2 else label2
            node.is_leaf = True
            return node

        # check if all rows only have single label
        if n_label1 == 0 or n_label2 == 0:
            node.label = label1 if n_label1 > n_label2 else label2
            node.is_leaf = True
            return node

        # check if 1 attribs left
        # select most common label
        if len(y) == 1:
            node.label = label1 if n_label1 > n_label2 else label2
            node.is_leaf = True
            return node

        # -----------
        # create branches for node
        # since all attrib binary => only 2 value: T/F
        # -----------

        # find best attrib
        max_attrib = -1
        max_info_gain = -1
        entr = get_entropy(S=X)
        for attrib in y:
            info_gain = get_info_gain(S=X, A=attrib, entropy_s=entr)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_attrib = attrib

        node.attrib = max_attrib

        # new label list
        new_y = y[:]
        new_y.remove(max_attrib)

        # left (False)
        X_left = X[X[max_attrib] == label2]
        node.left = self.ID3_depth(X_left, new_y, depth=depth+1)

        # right (True)
        X_right = X.drop(X_left.index)
        node.right = self.ID3_depth(X_right, new_y, depth=depth+1)

        return node

    def prune_tree(self):
        if self.valid_set is None:
            print("[ERROR] valid_set not set.")
            return

        self.max_accuracy = self.validate()
        print("--- pruning ---")
        print("before prune, valid test= %.4f" % self.max_accuracy)

        t1 = time.time()
        self.prune(node=self.root)
        print("prune duration:%d seconds" % (time.time() - t1))

        accuracy = self.validate()
        print("after prune, valid test= %.4f" % accuracy)

        n_nodes = get_num_nodes(self.root)
        print("Number of nodes after prune:", n_nodes)

        h = get_height(self.root)
        print("height after prune:", h)

    def prune(self, node: Node):
        if not node:
            return

        # try delete node
        node.is_leaf = True
        node.label = label1 if node.n_label1 > node.n_label2 else label2
        accuracy = self.validate()

        # if non-decreasing, delete node
        if accuracy > self.max_accuracy - 0.00005:
            self.max_accuracy = accuracy
            print('del one')
            return

        # if not good, try children
        node.is_leaf = False
        node.label = None
        if not node.left.is_leaf:
            self.prune(node.left)

        if not node.right.is_leaf:
            self.prune(node.right)

    def validate(self):

        # faster than loop row-by-row
        predict_col = self.valid_set.apply(
            lambda x: self.predict(row=x), axis=1
        )

        # correct_col: series<boolean>
        correct_col = (predict_col == self.valid_set[index_col])

        # now it only contain True
        n_correct = len(correct_col[correct_col])

        accuracy = n_correct / len(self.valid_set)

        # for index, row in self.valid_set.iterrows():
        #     output = self.predict(row=row)
        #     if output == row[index_col]:
        #         test_correct += 1
        #
        # accuracy = test_correct / len(self.valid_set)
        return accuracy

    def print_leaves(self):
        print('n_node =', self.n_nodes)
        self.n_leaves += 1
        if self.n_leaves % 10 == 0:
            print('n_leaf =', self.n_leaves)


def process_arg() -> dict:

    # ex: commands
    # python3 ID3.py train-file test-file vanilla 80
    # python3 ID3.py train-file test-file prune 50 40
    # python3 ID3.py train-file test-file maxDepth 50 40 5

    arg = dict()
    arg['train_path'] = sys.argv[1]
    arg['test_path']  = sys.argv[2]
    arg['option']     = sys.argv[3]
    arg['train_percent'] = int(sys.argv[4])
    arg['valid_percent'] = 100  # default for now

    if arg['option'] == 'prune':
        arg['valid_percent'] = int(sys.argv[5])

    if arg['option'] == 'maxDepth':
        arg['valid_percent'] = int(sys.argv[5])
        arg['max_depth'] = int(sys.argv[6])

    return arg

def main():

    arg: dict = process_arg()

    # --------------
    # read dataset
    # --------------
    df: pandas.DataFrame = pandas.read_csv(arg['train_path'], header=None)
    # df = preprocess.pre_process(df)

    # train set,
    num_ex = arg['train_percent'] * len(df) / 100
    num_ex = int(num_ex)
    df_train = df[:num_ex]
    df_train = preprocess.pre_process(df_train)
    print('train shape:', df_train.shape)

    # validation set
    num_ex = arg['valid_percent'] * len(df) / 100
    num_ex = int(num_ex)
    df_valid = df[-num_ex:]
    df_valid = preprocess.pre_process(df_valid)
    print('valid shape:', df_valid.shape)

    # test set
    df_test: pandas.DataFrame = pandas.read_csv(arg['test_path'], header=None)
    df_test = preprocess.pre_process(df_test)
    print('test shape:', df_test.shape)

    # --------------
    # build tree
    # --------------

    decision_tree = DecisionTree(train_set=df_train,
                                 valid_set=df_valid,
                                 test_set=df_test)

    print('build tree option:', arg['option'])
    if arg['option'] != 'maxDepth':
        decision_tree.build_normal()
    else:
        decision_tree.build_max_depth(arg['max_depth'])
        valid_accur = decision_tree.validate()
        print("validation set accuracy: %.4f" % valid_accur)

    # prune tree
    n_nodes = get_num_nodes(decision_tree.root)
    print("Number of nodes before prune:", n_nodes)
    h = get_height(decision_tree.root)
    print("height before prune:", h)

    if arg['option'] == 'prune':
        decision_tree.prune_tree()

    # --------------
    # testing
    # --------------

    train_correct = 0
    for index, row in df_train.iterrows():
        output = decision_tree.predict(row=row)
        if output == row[index_col]:
            train_correct += 1

    test_correct = 0
    for index, row in df_test.iterrows():
        output = decision_tree.predict(row=row)
        if output == row[index_col]:
            test_correct += 1

    train_accuracy = train_correct / len(df_train)
    test_accuracy  = test_correct  / len(df_test)
    print("Train set accuracy: %.4f" % train_accuracy)
    print("Test set accuracy: %.4f"  % test_accuracy)


warnings.filterwarnings('ignore')
main()





