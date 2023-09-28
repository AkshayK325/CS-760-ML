import numpy as np
import math
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, random_seed=None):
        self.tree = None
        self.random_seed = random_seed
        self.node_count = 0  

        if random_seed is not None:
            np.random.seed(random_seed)

    def entropy(self, data):
        n = len(data)
        if n == 0:
            return 0
        count0 = np.sum(data[:, -1] == 0)
        count1 = n - count0
        p0 = count0 / n
        p1 = count1 / n
        if p0 == 0 or p1 == 0:
            return 0
        return -p0 * np.log2(p0) - p1 * np.log2(p1)

    def information_gain(self, data, split):
        n = len(data)
        left_mask = data[:, split[0]] >= split[1]
        right_mask = ~left_mask
        entropy_parent = self.entropy(data)
        # print("Entropy Value=",entropy_parent)
        entropy_left = self.entropy(data[left_mask])
        entropy_right = self.entropy(data[right_mask])
        weight_left = np.sum(left_mask) / n
        weight_right = np.sum(right_mask) / n
        gain = entropy_parent - (weight_left * entropy_left + weight_right * entropy_right)
        return gain

    def find_best_split(self, data):
        best_split = None
        best_gain = -1
        num_features = data.shape[1] - 1  
        for j in range(num_features):
            values = np.unique(data[:, j])
            for c in values:
                split = (j, c)
                gain = self.information_gain(data, split)
                # print("Split and Gain = ",split,gain)
                if gain > best_gain:
                    best_gain = gain
                    best_split = split
        return best_split, best_gain

    class TreeNode:
        def __init__(self, feature_index=None, threshold=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = None
            self.right = None
            self.is_leaf = False
            self.predicted_class = None

    def build_decision_tree(self, data):
        if len(data) == 0:
            return None
        unique_classes = np.unique(data[:, -1])
        if len(unique_classes) == 1:
            leaf = self.TreeNode()
            leaf.is_leaf = True
            leaf.predicted_class = unique_classes[0]
            return leaf
        best_split, best_gain = self.find_best_split(data)
        if best_gain == 0:
            leaf = self.TreeNode()
            leaf.is_leaf = True
            leaf.predicted_class = np.argmax(np.bincount(data[:, -1]))
            return leaf
        left_mask = data[:, best_split[0]] >= best_split[1]
        right_mask = ~left_mask
        left_branch = data[left_mask]
        right_branch = data[right_mask]
        node = self.TreeNode(feature_index=best_split[0], threshold=best_split[1])
        node.left = self.build_decision_tree(left_branch)
        node.right = self.build_decision_tree(right_branch)
        return node

    def fit(self, data):
        self.tree = self.build_decision_tree(data)

    def predict_instance(self, node, instance):
        if node.is_leaf:
            return node.predicted_class
        if instance[node.feature_index] >= node.threshold:
            return self.predict_instance(node.left, instance)
        else:
            return self.predict_instance(node.right, instance)

    def predict(self, instance):
        if self.tree is None:
            raise Exception("The decision tree has not been fitted yet.")
        return self.predict_instance(self.tree, instance)
        
    def count_nodes_recursive(self, node):
        if node is None:
            return 0
        left_count = self.count_nodes_recursive(node.left)
        right_count = self.count_nodes_recursive(node.right)
        return 1 + left_count + right_count

    def count_nodes(self):
        if self.tree is None:
            raise Exception("The decision tree has not been fitted yet.")
        self.node_count = self.count_nodes_recursive(self.tree)
        return self.node_count

    def plot_tree(self, node=None, depth=0, parent_x=0, parent_y=0, feature_names=None, class_names=None, fontSize=10):
        if node is None:
            node = self.tree
            self.levels = 0  
    
        if depth == 0:
            self.levels = self.count_levels(self.tree)
    
            total_height = 2.0  # Adjust this
            self.level_height = total_height / (self.levels + 1)
        
            self.font_size = fontSize
    
        if not node.is_leaf:
            if feature_names is None:
                feature_name = f'Feature {node.feature_index}'
            else:
                feature_name = feature_names[node.feature_index]
    
            threshold = f'Threshold: {node.threshold:.2f}'
            node_x = parent_x
            node_y = parent_y - self.level_height
    
            if depth == 0:
                level_width = 1000.0  # Adjust this
            else:
                level_width = 800.0*1/depth  
    
            plt.text(node_x, node_y, f'{feature_name}\n{threshold}', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgray'), fontsize=self.font_size)
    
            num_nodes_in_subtree = self.count_nodes_recursive(node)
            node_spacing = level_width / (num_nodes_in_subtree + 1)
    
            if node.left is not None:
    
                left_x = node_x - (num_nodes_in_subtree / 2) * node_spacing
                left_y = node_y - self.level_height
                plt.plot([node_x, left_x], [parent_y, left_y], 'k-')
                self.plot_tree(node.left, depth + 1, left_x, left_y, feature_names=feature_names, class_names=class_names, fontSize=self.font_size)
    
            if node.right is not None:
    
                right_x = node_x + (num_nodes_in_subtree / 2) * node_spacing
                right_y = node_y - self.level_height
                plt.plot([node_x, right_x], [parent_y, right_y], 'k-')
                self.plot_tree(node.right, depth + 1, right_x, right_y, feature_names=feature_names, class_names=class_names, fontSize=self.font_size)
    
        else:
            if class_names is None:
                class_label = f'Class: {node.predicted_class}'
            else:
                class_label = class_names[int(node.predicted_class)]
    
            font_size = 8  # Adjust this
            plt.text(parent_x, parent_y, class_label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black'), fontsize=self.font_size)
    
        if depth == 0:
            plt.show()

    def count_nodes_recursive(self, node):
        if node is None:
            return 0
        left_count = self.count_nodes_recursive(node.left)
        right_count = self.count_nodes_recursive(node.right)
        return 1 + left_count + right_count
        
    def count_levels(self, node):
        if node is None:
            return 0
        left_levels = self.count_levels(node.left)
        right_levels = self.count_levels(node.right)
        return max(left_levels, right_levels) + 1