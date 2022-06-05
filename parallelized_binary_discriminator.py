import math
import json
import copy
import pickle
import dask
import numpy as np
import pandas as pd
from dask.distributed import LocalCluster, Client, progress
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed, parallel_backend
from pprint import pprint
from tqdm import tqdm

class TreeNode:
    def __init__(self, data, yname, natural, numeric_lst = [], min_sample = 0, is_imbalanced = True):
        self.index = data.index
        self.data = data
        self.yname = yname
        self.natural = natural
        self.min_sample = min_sample
        self.y0 = data[data[yname] == 0].shape[0]
        self.y1 = data[data[yname] == 1].shape[0]
        
        self.childnodes = {}
        self.info = self.get_info()
        self.info_tree = {self.info: {}}
        self.split_feat = None
        self.numeric_lst = numeric_lst
        self.is_imbalanced = is_imbalanced
    
    def get_info(self, df = None, natural = None):
        if df is None:
            y0 = self.y0
            y1 = self.y1
        else:
            y0 = df[df[self.yname] == 0].shape[0]
            y1 = df[df[self.yname] == 1].shape[0]
        if y0 > y1:
            vote = 0
        else:
            vote = 1
        if y0 + y1 == 0:
            lapse = 0
        else:
            lapse = y1 / (y1 + y0)
        if natural is None:
            natural = self.natural
        return f"(Pos%: {round(lapse * 100, 2)}% ({round(lapse / natural, 2)}), #0: {y0} & #1: {y1})"
    
    def get_entropy(self, y0 = None, y1 = None):
        if y0 is None or y1 is None:
            y0 = self.y0
            y1 = self.y1
        adj_factor = - np.log2(1-self.natural) - np.log2(self.natural)
        if y0 == 0 or y1 == 0:
            return adj_factor
        if self.is_imbalanced:
            return -y0/(y0 + y1)/(1-self.natural)*np.log2(y0/(y0 + y1)/(1-self.natural)) - y1/(y0 + y1)/self.natural*np.log2(y1/(y0 + y1)/self.natural) + adj_factor
        else:
            return -y0/(y0 + y1)*np.log2(y0/(y0 + y1)) - y1/(y0 + y1)*np.log2(y1/(y0 + y1))
    
    def extend(self, xname):
        data = self.data
        if xname not in self.numeric_lst:
            levels = list(set(data[xname].dropna()))
        else:
            df_curr = data[[xname, self.yname]].dropna().sort_values(xname)
            arr = np.array(df_curr[self.yname].astype(int))
            opt_idx = 0
            min_entropy = np.inf
            num_one = 0
            num_zero = 0
            total_one = np.sum(arr)
            total_zero = len(arr) - total_one
            for i in range(len(arr) - 1):
                if arr[i] == 0:
                    num_zero += 1
                else:
                    num_one += 1
                entropy = self.get_entropy(y0 = num_zero, y1 = num_one) / (num_zero + num_one) + self.get_entropy(y0 = total_zero - num_zero, y1 = total_one - num_one) / (len(arr) - num_zero - num_one)
                if entropy < min_entropy and num_one > 0 and num_zero > 0 and num_one < total_one and num_zero < total_zero and i + 1 >= self.min_sample and len(arr) - 1 - i >= self.min_sample:
                    min_entropy = entropy
                    opt_idx = i
            if df_curr.shape[0] > opt_idx + 1:
                opt_cut = (df_curr.iloc[opt_idx][xname] + df_curr.iloc[opt_idx+1][xname]) / 2
            else:
                opt_cut = df_curr.iloc[opt_idx][xname]
            levels = [f"Less than {opt_cut}", f"At least {opt_cut}"]
        self.split_feat = xname
        entro = 0
        min_sample = np.inf
        if len(levels) <= 1:
            return np.inf, 0
        for lev in levels:
            if xname not in self.numeric_lst:
                data_child = data[data[xname] == lev]
            else:
                if lev.startswith("Less than"):
                    data_child = data[data[xname] < opt_cut]
                else:
                    data_child = data[data[xname] >= opt_cut]
            child = TreeNode(data_child, self.yname, self.natural, self.numeric_lst, self.min_sample)
            self.childnodes[(xname, lev)] = child
            self.info_tree[self.info][f"({xname}, {lev})"] = child.info_tree
            entro += child.get_entropy() #/ len(self.index) * data_child.shape[0]
            min_sample = min(min_sample, data_child.shape[0])
        return entro, min_sample
    
    def reset(self):
        self.childnodes = {}
        self.split_feat = None
        self.info_tree = {self.info: {}}

class ParallelizedBinaryCategoricalDiscriminator:
    def __init__(self, yname, max_depth = 3, min_sample = 0, nested_parallel = False, cpu_split = 10):
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.yname = yname
        self.tree = None
        self.info_tree = None
        self.natural = 0
        self.numeric_lst = []
        self.nested_parallel = nested_parallel
        self.cpu_split = cpu_split
    
    def train(self, df, xname_lst, numeric_lst = []):
        self.natural = df[self.yname].mean()
        self.numeric_lst = numeric_lst
        self.tree = TreeNode(df, self.yname, self.natural, self.numeric_lst, self.min_sample)
        self.train_recurse(self.tree, self.max_depth, xname_lst)
        self.info_tree = {"Root": self.tree.info_tree}
    
    def extend_helper(self, xname_lst, node):
        node = copy.deepcopy(node)
        min_entro = np.inf
        opt_split = None
        for xname in tqdm(xname_lst):
            entro, sample_size = node.extend(xname)
            if entro < min_entro and sample_size >= self.min_sample:
                min_entro = entro
                opt_split = xname
            node.reset()
        return opt_split, min_entro
    
    def train_recurse(self, node, max_depth, xname_lst):
        if max_depth > 0 and node.get_entropy() > 0 and len(node.index) >= self.min_sample:
            opt_split = None
            min_entro = np.inf
            batch_size = int(math.ceil(len(xname_lst) / self.cpu_split))
            idx_lst = np.random.choice(len(xname_lst), size = len(xname_lst), replace=False)
            xname_lst_shuffled = [xname_lst[x] for x in idx_lst]
            res = Parallel(n_jobs=self.cpu_split)(delayed(self.extend_helper)(
                xname_lst_shuffled[(i * batch_size):min((i + 1) * batch_size, len(xname_lst))], node
            ) for i in range(self.cpu_split))
            for split_feat, entro in res:
                if split_feat is not None and entro < min_entro:
                    opt_split = split_feat
                    min_entro = entro
            if opt_split is not None:
                node.extend(opt_split)
                if opt_split not in self.numeric_lst:
                    child_xname_lst = [x for x in xname_lst if x != opt_split]
                else:
                    child_xname_lst = [x for x in xname_lst]
                key_lst = list(node.childnodes.keys())
                
                if self.nested_parallel:
                    n_cpu = len(node.childnodes)
                    res = Parallel(n_jobs=n_cpu, max_nbytes=1e6)(delayed(self.train_recurse)(
                        node.childnodes[key_lst[i]], max_depth - 1, child_xname_lst
                    ) for i in range(n_cpu))
                    for i in range(len(key_lst)):
                        node.childnodes[key_lst[i]] = res[i]
                        node.info_tree[node.info][f"({key_lst[i][0]}, {key_lst[i][1]})"] = res[i].info_tree
                else:
                    for i in range(len(key_lst)):
                        res = self.train_recurse(node.childnodes[key_lst[i]], max_depth - 1, child_xname_lst)
                        node.childnodes[key_lst[i]] = res
                        node.info_tree[node.info][f"({key_lst[i][0]}, {key_lst[i][1]})"] = res.info_tree
        return node
    
    def print_tree(self):
        pprint(self.info_tree)
        return self.info_tree
    
    def dump_tree_to_file(self, fname, tree = None):
        if tree is None:
            tree_to_dump = self.info_tree
        else:
            tree_to_dump = tree
        with open(fname, "w") as f:
            self.dump_tree_to_file_helper(f, tree_to_dump)
    
    def dump_tree_to_file_helper(self, f, dct, indent = ""):
        key_lst = list(dct.keys())
        if len(key_lst) > 0:
            for name in key_lst:
                inner_key_lst = list(dct[name].keys())
                if len(inner_key_lst) > 0:
                    vote = inner_key_lst[0]
                    inner_dct = dct[name][vote]
                    f.write(f"{indent}{name} -- {vote}:\n")
                    self.dump_tree_to_file_helper(f, inner_dct, indent + "\t")
            
    def predict(self, df_test):
        df_test = df_test.copy()
        df_test[self.yname] = self.natural
        leaf_lst = self.get_all_leaves()
        for leaf in leaf_lst:
            df_test.loc[df_test.index.isin(leaf.index), self.yname] = leaf.y1 / (leaf.y1 + leaf.y0)
        return df_test[self.yname]
    
    def get_all_leaves(self):
        lst = [self.tree]
        leaf_lst = []
        while len(lst) > 0:
            node = lst[0]
            lst = lst[1:]
            key_lst = list(node.childnodes.keys())
            if len(key_lst) > 0:
                for key in key_lst:
                    lst.append(node.childnodes[key])
            else:
                leaf_lst.append(node)
        return leaf_lst
    
    def evaluate(self, df_test, cutoff = None):
        if cutoff is None:
            cutoff = self.natural
        pred_prob = self.predict(df_test)
        pred = binarize([pred_prob], threshold = cutoff)[0]
        actual = df_test[self.yname]
        cm = confusion_matrix(actual, pred)
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        total = TP + FN + TN + FP
        misclass = (FP+FN)/(TP+FP+FN+TN)
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1_score = 2 * ((precision*recall)/(precision+recall))
        return accuracy, recall, precision, f1_score
    
    def eva(self, df_test):
        natural = df_test[self.yname].mean()
        tree = {"Root": self.eval_helper(df_test, self.tree, natural)}
        return tree
        
    def eval_helper(self, df_test, node, natural):
        info = node.get_info(df = df_test, natural = natural)
        tree = {info: {}}
        key_lst = list(node.childnodes.keys())
        for key in key_lst:
            child_node = node.childnodes[key]
            if key[0] not in self.numeric_lst:
                df_curr = df_test[df_test[key[0]] == key[1]]
            else:
                if key[1].startswith("Less than"):
                    cut = float(key[1].split("than")[1].strip())
                    df_curr = df_test[df_test[key[0]] < cut]
                else:
                    cut = float(key[1].split("least")[1].strip())
                    df_curr = df_test[df_test[key[0]] >= cut]
            tree[info][f"({key[0]}, {key[1]})"] = self.eval_helper(df_curr, child_node, natural)
        return tree
    
    def save_tree(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)
