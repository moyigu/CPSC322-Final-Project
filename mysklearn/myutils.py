"""
Name: Lida Chen
Date: 10/18
Assignment: PA4
Description: a class for pa4.ipynb and myclassifiers.py
"""
import graphviz
import numpy as np # use numpy's random number generation
import mysklearn.myclassifiers as mc
import mysklearn.myevaluation as me
from tabulate import tabulate

def calculate_dist(x_test, x_train, k_n):
    """Calculate the distances between test and train samples using Euclidean distance.

    Args:
        x_test (list of list of numeric vals): The list of testing samples.
        x_train (list of list of numeric vals): The list of training samples.
        k_n (int): The number of nearest neighbors to consider.

    Returns:
        tuple: A tuple containing:
            - distances (list of list of float): 2D list of distances to k nearest neighbors for each test instance.
            - neighbor_indices (list of list of int): 2D list of indices of the k nearest neighbors in the training set.
    """
    distances = []
    neighbor_indices = []
    for i in range(len(x_test)):
        tuples = []
        distance = []
        neighbor_index = []
        for j in range(len(x_train)):
            sum = 0
            for k in range(len(x_train[0])):
                sum += (x_test[i][k] - x_train[j][k]) ** 2
            tuples.append((sum ** 0.5, j))
            distance.append(sum ** 0.5)
        temp = sorted(tuples)[:k_n]
        for i in range(k_n):
            neighbor_index.append(temp[i][1])
        distances.append(sorted(distance)[:k_n])
        neighbor_indices.append(sorted(neighbor_index))
    return distances, neighbor_indices

def is_close(a, b, tol=1e-10):
    """Check if two numbers are close within a given tolerance.

    Args:
        a (float): The first number.
        b (float): The second number.
        tol (float, optional): The tolerance. Defaults to 1e-10.

    Returns:
        bool: True if the two numbers are close, False otherwise.
    """
    return abs(a - b) <= tol

def most_frequent(lst):
    """Find the most frequent item in a list.

    Args:
        lst (list of obj): The list of items.

    Returns:
        obj: The most frequent item in the list.
    """
    frequency = {} 
    max_count = -1
    max_item = None
    for item in lst:
        if item in frequency:
            frequency[item] += 1
        else:
            frequency[item] = 1
        if frequency[item] > max_count:
            max_count = frequency[item]
            max_item = item
    return max_item

def DOE_rating_convert(mpg):
    """Convert miles per gallon (mpg) to a DOE rating on a scale of 1 to 10.

    Args:
        mpg (float): The miles per gallon value.

    Returns:
        int: The DOE rating.
    """
    if mpg >= 45:
        return 10
    elif mpg > 37 and mpg <= 45:
        return 9
    elif mpg > 31 and mpg <= 37:
        return 8
    elif mpg > 27 and mpg <= 31:
        return 7
    elif mpg > 24 and mpg <= 27:
        return 6
    elif mpg > 20 and mpg <= 24:
        return 5
    elif mpg > 17 and mpg <= 20:
        return 4
    elif mpg > 15 and mpg <= 17:
        return 3
    elif mpg > 13 and mpg <= 15:
        return 2
    elif mpg <= 13:
        return 1

def y_list_convert(y):
    """Convert a list of mpg values to DOE ratings.

    Args:
        y (list of float): The list of mpg values.

    Returns:
        list of int: The converted list of DOE ratings.
    """
    y_c = []
    for i in range(len(y)):
        y_c.append(DOE_rating_convert(y[i]))
    return y_c

def print_predict_result(x_test, y_pred, y_test):
    """Print the prediction results alongside the actual values and the accuracy.

    Args:
        x_test (list of list of numeric vals): The list of test instances.
        y_pred (list of obj): The predicted target values.
        y_test (list of float): The actual target values (in mpg).
    """
    y_test_c = y_list_convert(y_test)
    for i in range(len(x_test)):
        print(f"instance: {x_test[i]}")
        print(f"class: {y_pred[i]} actual: {y_test_c[i]}")
    print(f"accuracy: {accuracy(y_pred, y_test_c)}")

def accuracy(y_pred, y_test):
    """Calculate the accuracy of predictions.

    Args:
        y_pred (list of obj): The predicted target values.
        y_test (list of obj): The actual target values.

    Returns:
        float: The accuracy of the predictions as a percentage.
    """
    correct = 0
    total = len(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    
    return correct / total

def calculate_frequency(y):
    """Calculate the frequency of each label in a list.

    Args:
        y (list of obj): The list of labels.

    Returns:
        dict: A dictionary mapping each label to its frequency.
    """
    stat = {}
    for i in y:
        if i not in stat:
            stat[i] = 1
        else:
            stat[i] += 1
    return stat

def ramdonly_choose(stat):
    """Randomly choose a label based on the frequency distribution.

    Args:
        stat (dict): A dictionary mapping labels to their frequencies.

    Returns:
        list of obj: A list containing the randomly chosen label.
    """
    labels = list(stat.keys())
    probs = list(stat.values())
    return [np.random.choice(labels, p=np.array(probs) / np.sum(probs))]

def random_subsample(X, y, k=10, test_size=0.33, random_state=None):
    """Calculate predictive accuracy and error rate for KNN and Dummy classifiers using random subsampling.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target labels (parallel to X).
        k (int): Number of random subsampling iterations to perform.
        test_size (float): Proportion of the dataset to include in the test split (e.g., 0.33 for a 2:1 split).
        random_state (int): Seed for random number generator (for reproducibility).

    Returns:
        tuple: A tuple containing:
            - knn_accuracy (float): The average accuracy of the KNN classifier across all subsamples.
            - knn_error (float): The average error rate of the KNN classifier across all subsamples.
            - dummy_accuracy (float): The average accuracy of the Dummy classifier across all subsamples.
            - dummy_error (float): The average error rate of the Dummy classifier across all subsamples.

    Notes:
        - Each iteration, the data is split into training and test sets, maintaining the specified `test_size`.
        - The KNN and Dummy classifiers are then trained and tested on each split.
        - The results are averaged over `k` iterations.
    """
    knn_accuracy = 0
    knn_error = 0
    dummy_accuracy = 0
    dummy_error = 0

    for i in range(k):
        X_train, X_test, y_train, y_test = me.train_test_split(X, y, test_size=test_size, random_state=random_state + i if random_state is not None else None, shuffle=True)

        knn = mc.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = me.accuracy_score(y_test, y_pred_knn)
        knn_accuracy += accuracy_knn
        knn_error += 1 - accuracy_knn

        dummy = mc.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        accuracy_dummy = me.accuracy_score(y_test, y_pred_dummy)
        dummy_accuracy += accuracy_dummy
        dummy_error += 1 - accuracy_dummy

    knn_accuracy /= k
    knn_error /= k
    dummy_accuracy /= k
    dummy_error /= k

    return knn_accuracy, knn_error, dummy_accuracy, dummy_error

def cross_val_predict(X, y, k=10, stratify=False, random_state=None):
    """Compute cross-validated predictions for each instance in X.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target labels (parallel to X).
        k (int): Number of folds for cross-validation.
        stratify (bool): If True, perform stratified cross-validation.
        random_state (int): Seed for random number generator (for reproducibility).

    Returns:
        tuple: A tuple containing:
            - knn_accuracy (float): The average accuracy of the KNN classifier.
            - knn_error (float): The average error rate of the KNN classifier.
            - dummy_accuracy (float): The average accuracy of the Dummy classifier.
            - dummy_error (float): The average error rate of the Dummy classifier.
    """
    knn_accuracy = 0
    knn_error = 0
    dummy_accuracy = 0
    dummy_error = 0

    if stratify:
        folds = me.stratified_kfold_split(X, y, n_splits=k, random_state=random_state, shuffle=True)
    else:
        folds = me.kfold_split(X, n_splits=k, random_state=random_state, shuffle=True)

    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        knn = mc.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = me.accuracy_score(y_test, y_pred_knn)
        knn_accuracy += accuracy_knn
        knn_error += 1 - accuracy_knn

        dummy = mc.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        accuracy_dummy = me.accuracy_score(y_test, y_pred_dummy)
        dummy_accuracy += accuracy_dummy
        dummy_error += 1 - accuracy_dummy

    knn_accuracy /= k
    knn_error /= k
    dummy_accuracy /= k
    dummy_error /= k

    return knn_accuracy, knn_error, dummy_accuracy, dummy_error

def bootstrap_method(X, y, k=10, n_samples=None, random_state=None):
    """Calculate the predictive accuracy and error rate using the bootstrap method.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target labels (parallel to X).
        k (int): Number of bootstrap samples to generate.
        n_samples (int): Number of samples to generate per bootstrap. Defaults to len(X).
        random_state (int): Seed for random number generator (for reproducibility).

    Returns:
        tuple: A tuple containing:
            - knn_accuracy (float): The average accuracy of the KNN classifier.
            - knn_error (float): The average error rate of the KNN classifier.
            - dummy_accuracy (float): The average accuracy of the Dummy classifier.
            - dummy_error (float): The average error rate of the Dummy classifier.
    """
    knn_accuracy = 0
    knn_error = 0
    dummy_accuracy = 0
    dummy_error = 0

    for i in range(k):
        X_train, X_out_of_bag, y_train, y_out_of_bag = me.bootstrap_sample(X, y, n_samples=n_samples, random_state=random_state + i if random_state is not None else None)

        knn = mc.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_out_of_bag)
        accuracy_knn = me.accuracy_score(y_out_of_bag, y_pred_knn)
        knn_accuracy += accuracy_knn
        knn_error += 1 - accuracy_knn

        dummy = mc.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_out_of_bag)
        accuracy_dummy = me.accuracy_score(y_out_of_bag, y_pred_dummy)
        dummy_accuracy += accuracy_dummy
        dummy_error += 1 - accuracy_dummy

    knn_accuracy /= k
    knn_error /= k
    dummy_accuracy /= k
    dummy_error /= k

    return knn_accuracy, knn_error, dummy_accuracy, dummy_error

def compute_confusion_matrices(X, y, k=10, stratify=False, random_state=None):
    """Compute the confusion matrices for both KNN and Dummy classifiers using k-fold cross-validation.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target labels (parallel to X).
        k (int): Number of folds for cross-validation.
        stratify (bool): If True, perform stratified cross-validation.
        random_state (int): Seed for random number generator (for reproducibility).

    Returns:
        tuple: A tuple containing:
            - knn_conf_matrix (list of list of int): Confusion matrix for the KNN classifier.
            - dummy_conf_matrix (list of list of int): Confusion matrix for the Dummy classifier.
    """
    y_pred_knn = [None] * len(y)
    y_pred_dummy = [None] * len(y)

    if stratify:
        folds = me.stratified_kfold_split(X, y, n_splits=k, random_state=random_state, shuffle=True)
    else:
        folds = me.kfold_split(X, n_splits=k, random_state=random_state, shuffle=True)

    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]

        knn = mc.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_fold_knn = knn.predict(X_test)
        
        dummy = mc.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_fold_dummy = dummy.predict(X_test)

        for idx, pred_knn, pred_dummy in zip(test_indices, y_pred_fold_knn, y_pred_fold_dummy):
            y_pred_knn[idx] = pred_knn
            y_pred_dummy[idx] = pred_dummy

    knn_conf_matrix = me.confusion_matrix(y, y_pred_knn, labels=sorted(set(y)))
    dummy_conf_matrix = me.confusion_matrix(y, y_pred_dummy, labels=sorted(set(y)))
    
    return knn_conf_matrix, dummy_conf_matrix

def display_confusion_matrix(matrix, labels, classifier_name="Classifier"):
    """Display a confusion matrix using the tabulate package in a formatted table.

    Args:
        matrix (list of list of int): The confusion matrix to display.
        labels (list of str): List of labels for the rows and columns of the confusion matrix.
        classifier_name (str): The name of the classifier (e.g., "KNN" or "Dummy").

    Notes:
        The table includes additional columns for the total count per row
        and the recognition percentage for each label.
    """
    headers = [""] + labels + ["Total", "Recognition (%)"]

    table = []
    for i, row in enumerate(matrix):
        total = sum(row)
        recognition = (row[i] / total * 100) if total > 0 else 0
        table.append([labels[i]] + row + [total, f"{recognition:.2f}%"])

    print(f"{classifier_name} Confusion Matrix:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

def calculate_categorical_dist(x_test, x_train, k_n):
    """Calculate the distances between test and train samples using Hamming distance for categorical attributes.

    Args:
        x_test (list of list of categorical vals): The list of testing samples.
        x_train (list of list of categorical vals): The list of training samples.
        k_n (int): The number of nearest neighbors to consider.

    Returns:
        tuple: A tuple containing:
            - distances (list of list of int): 2D list of distances to k nearest neighbors for each test instance.
            - neighbor_indices (list of list of int): 2D list of indices of the k nearest neighbors in the training set.
    """
    distances = []
    neighbor_indices = []

    for test_instance in x_test:
        # List to store (distance, index) tuples
        tuples = []
        for index, train_instance in enumerate(x_train):
            # Calculate Hamming distance
            distance = sum(1 for i in range(len(train_instance)) if test_instance[i] != train_instance[i])
            tuples.append((distance, index))

        # Sort tuples by distance and take the k nearest neighbors
        tuples.sort(key=lambda x: x[0])
        k_nearest = tuples[:k_n]

        # Separate distances and indices
        distances.append([d[0] for d in k_nearest])
        neighbor_indices.append([d[1] for d in k_nearest])

    return distances, neighbor_indices

def evaluate_and_display_metrics(y_true, y_pred, labels=None, classifier_name="Classifier"):
    """Evaluate and display the performance metrics of a classifier using confusion matrix,
       accuracy, precision, recall, and F1 score.

    Args:
        y_true (list of obj): The ground truth target y values.
        y_pred (list of obj): The predicted target y values (parallel to y_true).
        labels (list of str, optional): List of labels for the classification problem. Defaults to None.
        classifier_name (str, optional): The name of the classifier (e.g., "KNN" or "Dummy"). Defaults to "Classifier".
    """
    if labels is None:
        labels = sorted(set(y_true))

    conf_matrix = me.confusion_matrix(y_true, y_pred, labels)

    accuracy = me.accuracy_score(y_true, y_pred)
    precision = me.binary_precision_score(y_true, y_pred, labels=labels)
    recall = me.binary_recall_score(y_true, y_pred, labels=labels)
    f1 = me.binary_f1_score(y_true, y_pred, labels=labels)

    display_confusion_matrix(conf_matrix, labels, classifier_name)

    print(f"{classifier_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Error: {1 - accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")

def cross_val_predict2(X, y, k=10, stratify=False, random_state=None):
    """Compute cross-validated predictions for each instance in X.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target labels (parallel to X).
        k (int): Number of folds for cross-validation.
        stratify (bool): If True, perform stratified cross-validation.
        random_state (int): Seed for random number generator (for reproducibility).

    Returns:
        tuple: A tuple containing:
            - y_pred_knn (list of obj): The predicted target values by KNN for all instances.
            - y_pred_dummy (list of obj): The predicted target values by Dummy classifier for all instances.
            - y_pred_naive (list of obj): The predicted target values by Naive Bayes classifier for all instances.
            - y_pred_tree (list of obj): The predicted target values by Decision Tree classifier for all instances.
    """
    y_pred_knn = [None] * len(y)
    y_pred_dummy = [None] * len(y)
    y_pred_naive = [None] * len(y)
    y_pred_tree = [None] * len(y) 

    if stratify:
        folds = me.stratified_kfold_split(X, y, n_splits=k, random_state=random_state, shuffle=True)
    else:
        folds = me.kfold_split(X, n_splits=k, random_state=random_state, shuffle=True)

    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]

        knn = mc.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_fold_knn = knn.predict(X_test)

        dummy = mc.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_fold_dummy = dummy.predict(X_test)

        naive = mc.MyNaiveBayesClassifier()
        naive.fit(X_train, y_train)
        y_pred_fold_naive = naive.predict(X_test)

        tree = mc.MyDecisionTreeClassifier()
        tree.fit(X_train, y_train)
        y_pred_fold_tree = tree.predict(X_test)

        for idx, pred_knn, pred_dummy, pred_naive, pred_tree in zip(
            test_indices, y_pred_fold_knn, y_pred_fold_dummy, y_pred_fold_naive, y_pred_fold_tree
        ):
        
            y_pred_knn[idx] = pred_knn
            y_pred_dummy[idx] = pred_dummy
            y_pred_naive[idx] = pred_naive
            y_pred_tree[idx] = pred_knn

    return y_pred_knn, y_pred_dummy, y_pred_naive, y_pred_tree


def tdidt(current_instances, available_attributes):
    """Builds a decision tree using the Top-Down Induction of Decision Trees (TDIDT) algorithm.

    Args:
        current_instances (list of list of obj): The current subset of data instances being considered.
        available_attributes (list of str): The list of available attributes to split on.

    Returns:
        list: A nested list representation of the decision tree.
    """
    length = len(current_instances)
    all_entropy = []
    splits = []

    for att_name in available_attributes:
        att_index = int(att_name[3:]) 
        data, value = split_data_for_dt(current_instances, att_index)
        splits.append((data, value))
        entropy = cal_entropy(data, length)
        all_entropy.append(entropy)

    min_entropy_index = all_entropy.index(min(all_entropy))
    best_split_data, best_split_values = splits[min_entropy_index]
    best_attribute = available_attributes[min_entropy_index] 

    if len(available_attributes) == 1:
        tree = ["Attribute", best_attribute]
        for value, subset in zip(best_split_values, best_split_data):
            if len(subset) == 0:
                class_label = majority_class(current_instances)
                tree.append(["Value", value, ["Leaf", class_label, 0, len(current_instances)]])
            else:
                class_label = majority_class(subset)
                tree.append(["Value", value, ["Leaf", class_label, len(subset), len(current_instances)]])
        return tree

    tree = ["Attribute", best_attribute]

    for value, subset in zip(best_split_values, best_split_data):
        if len(subset) == 0: 
            class_label = majority_class(current_instances)
            tree.append(["Value", value, ["Leaf", class_label, 0, len(current_instances)]])
        elif all_same_class(subset): 
            class_label = subset[0][-1]
            tree.append(["Value", value, ["Leaf", class_label, len(subset), len(current_instances)]])
        else:
            sub_attributes = available_attributes.copy()
            sub_attributes.remove(best_attribute)
            subtree = tdidt(subset, sub_attributes)
            tree.append(["Value", value, subtree])

    return tree


def split_data_for_dt(instances, att_index):
    """Splits data into subsets based on the unique values of a specific attribute.

    Args:
        instances (list of list of obj): The dataset to split.
        att_index (int): The index of the attribute to split on.

    Returns:
        tuple: A tuple containing:
            - data (list of list of list of obj): A list of subsets, one for each unique value of the attribute.
            - value (list of obj): A list of unique values for the attribute.
    """
    data = []
    value = []
    for instance in instances:
        if instance[att_index] not in value:
            value.append(instance[att_index])
            data.append([instance])
        else:
            value_index = value.index(instance[att_index])
            data[value_index].append(instance)
    
    return data, value

def cal_entropy(data, length):
    """Calculates the entropy of a dataset.

    Args:
        data (list of list of obj): The dataset split into subsets.
        length (int): The total number of instances in the dataset.

    Returns:
        float: The calculated entropy.
    """
    each_len = []
    each_entro = []
    for split in data:
        each_len.append(len(split))
        probs = []
        value = []
        for instance in split:
            if instance[-1] not in value:
                value.append(instance[-1])
                probs.append(1)
            else:
                probs[value.index(instance[-1])] += 1
        E = 0
        for prob in probs:
            E += (prob / each_len[-1]) * np.log2(prob / each_len[-1])
        E *= -1
        each_entro.append(E)
    entropy = 0
    for i, e in enumerate(each_entro):
        entropy += e * (each_len[i] / length)
    return entropy

def all_same_class(instances):
    """Checks if all instances in a dataset belong to the same class.

    Args:
        instances (list of list of obj): The dataset to check.

    Returns:
        bool: True if all instances belong to the same class, False otherwise.
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    return True

def majority_class(instances):
    """Finds the most common class label in a dataset.

    Args:
        instances (list of list of obj): The dataset to analyze.

    Returns:
        obj: The most common class label.
    """
    counts = {}
    for instance in instances:
        label = instance[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return max(counts, key=counts.get)

def traverse_tree(node, rule, attribute_names, class_name):
    """Recursively traverses the decision tree to generate and print rules.

    Args:
        node (list): The current node in the decision tree.
        rule (list): The current rule being constructed as a list of conditions.
        attribute_names (list of str or None): A list of attribute names to use in the decision rules.
        class_name (str): A string to use for the class name in the decision rules.
    """
    if node[0] == "Leaf":
        class_label = node[1]
        print("IF " + " AND ".join(rule) + f" THEN {class_name} = {class_label}")
        return

    attribute = node[1]
    for i in range(2, len(node)):
        value_node = node[i]
        value = value_node[1]
        new_rule = rule + [f"{attribute_names[int(attribute[3:])] if attribute_names else attribute} == {value}"]
        traverse_tree(value_node[2], new_rule, attribute_names, class_name)

def predict_instance(instance, node, fallback_label):
    """Helper function to predict the class label for a single instance.

    Args:
        instance (list of obj): The test instance to predict
        node (list): The current node in the decision tree

    Returns:
        obj: The predicted class label
    """
    if node[0] == "Leaf":
        return node[1] 

    attribute = node[1]
    attribute_index = int(attribute[3:]) 

    for i in range(2, len(node)):
        value_node = node[i]
        if instance[attribute_index] == value_node[1]:
            return predict_instance(instance, value_node[2], fallback_label)

    return fallback_label

def add_nodes_edges(dot, node, parent_id=None, edge_label="", attribute_names=None):
    """Helper function to recursively add nodes and edges to a Graphviz Digraph.

    Args:
        dot (graphviz.Digraph): The Graphviz Digraph object to add nodes and edges to.
        node (list): The current node in the decision tree.
        parent_id (str or None): The ID of the parent node (None for the root node).
        edge_label (str): The label for the edge connecting the parent to the current node.
        attribute_names (list of str or None): A list of attribute names to use in the decision rules.
    """
    node_id = str(id(node))

    if node[0] == "Leaf":
        label = f"Class: {node[1]}"
        dot.node(node_id, label, shape="ellipse")
    else:
        attribute = node[1]
        if attribute_names:
            attribute_index = int(attribute[3:])
            label = attribute_names[attribute_index]
        else:
            label = attribute
        dot.node(node_id, label, shape="box")

    if parent_id is not None:
        dot.edge(parent_id, node_id, label=edge_label)

    if node[0] != "Leaf":
        for i in range(2, len(node)):
            value_node = node[i]
            value_label = value_node[1]
            add_nodes_edges(dot, value_node[2], node_id, str(value_label), attribute_names)

def is_numeric_data(X):
    """Check if all elements in X are numeric.

    Args:
        X (list of list): The list of data to check.

    Returns:
        bool: True if all elements in X are numeric, False otherwise.
    """
    for row in X:
        for element in row:
            if not isinstance(element, (int, float)):
                return False
    return True

def determine_and_calculate_dist(X_test, X_train, k_n):
    """Determine the data type of X_train and X_test and call the appropriate distance function.

    Args:
        X_test (list of list): The list of testing samples.
        X_train (list of list): The list of training samples.
        k_n (int): The number of nearest neighbors to consider.

    Returns:
        tuple: A tuple containing:
            - distances (list of list): 2D list of distances to k nearest neighbors for each test instance.
            - neighbor_indices (list of list): 2D list of indices of the k nearest neighbors in the training set.
    """
    if is_numeric_data(X_train) and is_numeric_data(X_test):
        return calculate_dist(X_test, X_train, k_n)
    elif is_numeric_data(X_train) is False and is_numeric_data(X_test) is False: 
        return calculate_categorical_dist(X_test, X_train, k_n)
    else:
        raise ValueError("Mixed or unsupported data types in X_train or X_test.")
    
def cross_val_predict3(X, y, k=10, stratify=False, random_state=None):
    """Compute cross-validated predictions for each instance in X.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target labels (parallel to X).
        k (int): Number of folds for cross-validation.
        stratify (bool): If True, perform stratified cross-validation.
        random_state (int): Seed for random number generator (for reproducibility).

    Returns:
        tuple: A tuple containing:
            - y_pred_knn (list of obj): The predicted target values by KNN for all instances.
            - y_pred_dummy (list of obj): The predicted target values by Dummy classifier for all instances.
            - y_pred_naive (list of obj): The predicted target values by Naive Bayes classifier for all instances.
            - y_pred_tree (list of obj): The predicted target values by Decision Tree classifier for all instances.
    """
    y_pred_knn = [None] * len(y)
    y_pred_dummy = [None] * len(y)
    y_pred_naive = [None] * len(y)
    y_pred_tree = [None] * len(y) 

    if stratify:
        folds = me.stratified_kfold_split(X, y, n_splits=k, random_state=random_state, shuffle=True)
    else:
        folds = me.kfold_split(X, n_splits=k, random_state=random_state, shuffle=True)
    i = 0
    for train_indices, test_indices in folds:
        print(f"Processing fold {i + 1}/{k}...")
        i += 1
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]

        knn = mc.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_fold_knn = knn.predict(X_test)

        # dummy = mc.MyDummyClassifier()
        # dummy.fit(X_train, y_train)
        # y_pred_fold_dummy = dummy.predict(X_test)

        # naive = mc.MyNaiveBayesClassifier()
        # naive.fit(X_train, y_train)
        # y_pred_fold_naive = naive.predict(X_test)

        # tree = mc.MyDecisionTreeClassifier()
        # tree.fit(X_train, y_train)
        # y_pred_fold_tree = tree.predict(X_test)

        # for idx, pred_knn, pred_dummy, pred_naive, pred_tree in zip(
        #     test_indices, y_pred_fold_knn, y_pred_fold_dummy, y_pred_fold_naive, y_pred_fold_tree
        # ):
        for idx, pred_knn in zip(
            test_indices,  y_pred_fold_knn
        ):
            y_pred_knn[idx] = pred_knn
            # y_pred_dummy[idx] = pred_dummy
            # y_pred_naive[idx] = pred_naive
            # y_pred_tree[idx] = pred_knn

    return y_pred_knn, y_pred_dummy, y_pred_naive, y_pred_tree

def bootstrap_method(X, y, classifiers, k=10, random_state=None):
    accuracy_scores = {name: [] for name in classifiers}
    error_rates = {name: [] for name in classifiers}
    for i in range(k):
        # Set a unique random state for each iteration
        rs = random_state + i if random_state is not None else None

        # Get the bootstrapped sample and out-of-bag sample
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = eval.bootstrap_sample(X, y, random_state=rs)

        # Iterate through classifiers
        for name, clf in classifiers.items():
            # Train on the bootstrap sample
            clf.fit(X_sample, y_sample)

            # Predict on the out-of-bag samples
            if len(X_out_of_bag) > 0:  # Ensure there are out-of-bag samples to predict
                y_pred = clf.predict(X_out_of_bag)

                # Calculate accuracy and error rate
                accuracy = eval.accuracy_score(y_out_of_bag, y_pred)
                error_rate = 1 - accuracy

                # Append scores
                accuracy_scores[name].append(accuracy)
                error_rates[name].append(error_rate)

    # Calculate average accuracy and error rates
    avg_accuracy = {name: sum(scores) / len(scores) if len(scores) > 0 else 0 for name, scores in accuracy_scores.items()}
    avg_error_rate = {name: sum(rates) / len(rates) if len(rates) > 0 else 0 for name, rates in error_rates.items()}

    return avg_accuracy, avg_error_rate

