# TODO: your reusable general-purpose functions here
import math
import random
import numpy as np # use numpy's random number generation
import mysklearn.myevaluation as eval
from tabulate import tabulate
from mysklearn import myevaluation

def euclidean_distance(x_test, x_train, k_n):
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



def predict(y_test):
    y_predicted = []
    for i in range(len(y_test)):
        y_store = {}
        if isinstance(y_test[i], (list, tuple)):
            for j in range(len(y_test[i])):
                if y_test[i][j] not in y_store:
                    y_store[y_test[i][j]] = 1
                else:
                    y_store[y_test[i][j]] += 1
            y_predicted.append(max(y_store,key=y_store.get))
        else:
            if y_test[i] not in y_store:
                y_store[y_test[i]] = 1
            else:
                y_store[y_test[i]] += 1
    if y_predicted == []:
        y_predicted.append(max(y_store,key=y_store.get))
    return y_predicted


def mpg_to_rating(mpg_value):
    """
    Convert MPG values to ratings (1-10) based on DOE standards.
    
    Args:
        mpg_value (float): The MPG value to convert.
    
    Returns:
        int: The MPG rating on a scale from 1 to 10.
    """
    if mpg_value >= 45:
        return 10
    elif 37 <= mpg_value <= 44:
        return 9
    elif 31 <= mpg_value <= 36:
        return 8
    elif 27 <= mpg_value <= 30:
        return 7
    elif 24 <= mpg_value <= 26:
        return 6
    elif 20 <= mpg_value <= 23:
        return 5
    elif 17 <= mpg_value <= 19:
        return 4
    elif 15 <= mpg_value <= 16:
        return 3
    elif mpg_value == 14:
        return 2
    else:
        return 1
    
def accuracy(pred, actual):
    true = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            true += 1
    return (true / len(pred))

def random_subsample(X, y, classifiers, k=10, test_size=0.33, random_state=None):
    accuracy_scores = {name: [] for name in classifiers}
    error_rates = {name: [] for name in classifiers}

    for i in range(k):
        rs = random_state + i if random_state is not None else None
        X_train, X_test, y_train, y_test = eval.train_test_split(X, y, test_size=test_size, random_state=rs)
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = eval.accuracy_score(y_test,y_pred)
            error_rate = 1 - accuracy
            accuracy_scores[name].append(accuracy)
            error_rates[name].append(error_rate)
    avg_accuracy = {name: np.mean(scores) for name, scores in accuracy_scores.items()}
    avg_error_rate = {name: np.mean(rates) for name, rates in error_rates.items()}

    return avg_accuracy, avg_error_rate

def cross_val_predict(X, y, classifiers, k=10, random_state=None, shuffle=False,stratify=False):
    """Performs k-fold cross-validation using a custom kfold_split function."""
    
    # Initialize dictionaries to store accuracy scores and error rates
    accuracy_scores = {name: [] for name in classifiers}
    error_rates = {name: [] for name in classifiers}
    precisions = {name: [] for name in classifiers}
    recalls = {name: [] for name in classifiers}
    f1s ={name: [] for name in classifiers}
    # Use the custom kfold_split function to create folds
    if stratify:
        folds = eval.stratified_kfold_split(X,y,n_splits=10,random_state=random_state,shuffle=shuffle)
    else:
        folds = eval.kfold_split(X, n_splits=10, random_state=random_state, shuffle=shuffle)

    # Iterate through k-folds
    for train_indices, test_indices in folds:
        # Create train and test sets using list comprehensions
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        # Iterate through classifiers
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # Calculate accuracy and error rate
            accuracy = eval.accuracy_score(y_test, y_pred)
            error_rate = 1 - accuracy
            precision = eval.binary_precision_score(y_test, y_pred)
            recall = eval.binary_recall_score(y_test, y_pred)
            f1 = eval.binary_f1_score(y_test, y_pred)
            # Append scores
            accuracy_scores[name].append(accuracy)
            error_rates[name].append(error_rate)
            precisions[name].append(precision)
            recalls[name].append(recall)
            f1s[name].append(f1)

    # Calculate average accuracy and error rates
    avg_accuracy = {name: sum(scores) / len(scores) for name, scores in accuracy_scores.items()}
    avg_error_rate = {name: sum(rates) / len(rates) for name, rates in error_rates.items()}
    avg_precision = {name: sum(precisions) / len(precisions) for name, precisions, in precisions.items()}
    avg_recall = {name: sum(recalls) / len(recalls) for name, recalls in recalls.items()}
    avg_f1 = {name: sum(f1s) / len(f1s) for name, f1s in f1s.items()}
    
    return avg_accuracy, avg_error_rate, avg_precision, avg_recall, avg_f1

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

def display_confusion_matrix(matrix, labels, classifier_name, method,name):
    """Display the confusion matrix using the tabulate package."""
    # Prepare the header for the table
    header = [name] + labels + ["Total", "Recognition (%)"]
    
    # Prepare the table rows
    rows = []
    for i, label in enumerate(labels):
        row = [label] + matrix[i] + [sum(matrix[i]), round(100 * matrix[i][i] / sum(matrix[i]), 2) if sum(matrix[i]) > 0 else 0]
        rows.append(row)

    # Print the formatted table
    print("===========================================")
    print(f"Confusion Matrices")
    print("===========================================")
    print(f"{classifier_name} ({method}):")
    print(tabulate(rows, headers=header, tablefmt="grid"))

def evaluate_confusion_matrices(X, y, sign, classifiers, k=10, stratify=False, random_state=None, shuffle=False):
    # Cross-validation results
    avg_accuracy, avg_error_rate, avg_precision, avg_recall, avg_f1 = cross_val_predict(X, y, classifiers, k=k, stratify=stratify, random_state=random_state, shuffle=shuffle)

    # Generate confusion matrices for each classifier
    for name, clf in classifiers.items():
        # Assume we have collected true and predicted labels for each fold
        y_true_all, y_pred_all = [], []
        
        # Simulate k-fold or stratified k-fold cross-validation
        folds = eval.stratified_kfold_split(X, y, n_splits=k, random_state=random_state, shuffle=shuffle) if stratify else eval.kfold_split(X, n_splits=k, random_state=random_state, shuffle=shuffle)
        
        for train_indices, test_indices in folds:
            X_train = [X[i] for i in train_indices]
            X_test = [X[i] for i in test_indices]
            y_train = [y[i] for i in train_indices]
            y_test = [y[i] for i in test_indices]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Collect true and predicted labels for confusion matrix calculation
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
        # Define unique labels
        labels = sorted(set(y))

        # Calculate confusion matrix
        matrix = eval.confusion_matrix(y_true_all, y_pred_all, labels)

        # Display the confusion matrix
        method = "Stratified 10-Fold Cross Validation" if stratify else "10-Fold Cross Validation"
        display_confusion_matrix(matrix, labels, name, method, sign)


def calculate_entropy(labels):
    """Calculate entropy for a list of labels."""
    total = len(labels)
    counts = {label: labels.count(label) for label in set(labels)}
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

def calculate_enew(data, attribute_index):
    """Calculate the weighted average of entropies (Enew) for a specific attribute."""
    partitions = {}
    for row in data:
        key = row[attribute_index]
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(row)

    weighted_entropy = sum(
        (len(subset) / len(data)) * calculate_entropy([row[-1] for row in subset])
        for subset in partitions.values()
    )
    return weighted_entropy

def majority_vote(labels):
    """Return the majority label, breaking ties alphabetically."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    majority_label = None
    max_count = 0

    for label, count in sorted(counts.items()):  # Sort labels alphabetically for deterministic result
        if count > max_count or (count == max_count and label < majority_label):
            majority_label = label
            max_count = count

    return majority_label

def tdidt(data, attributes, total_count):
    """Recursive TDIDT algorithm with deterministic leaf result."""
    labels = [row[-1] for row in data]

    # Base case: If all labels are the same, return a leaf
    if len(set(labels)) == 1:
        return ["Leaf", labels[0], len(data), total_count]

    # Base case: If no attributes are left, handle tie-breaking deterministically
    if not attributes:
        # Count occurrences of each label
        label_counts = {label: labels.count(label) for label in set(labels)}
        # Select the label with the highest count, breaking ties alphabetically
        chosen_label = max(label_counts, key=lambda label: (label_counts[label], -ord(label[0])))
        return ["Leaf", chosen_label, len(data), total_count]

    # Use Enew to select the best attribute
    enews = [calculate_enew(data, i) for i in range(len(attributes))]
    best_attribute_index = min(
        range(len(enews)), key=lambda i: (enews[i], attributes[i])
    )  # Break ties by attribute name
    best_attribute = attributes[best_attribute_index]

    # Partition data by the best attribute
    partitions = {}
    for row in data:
        key = row[best_attribute_index]
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(row)

    if len(attributes) == 1:
        tree = ["Attribute",best_attribute]
        for att, value in partitions.items():
            class_label = majority_class(value)
            tree.append(['Value',att,["Leaf",class_label,len(value),len(data)]])
        return tree


    # Build decision node
    tree = ["Attribute", best_attribute]
    for key in sorted(partitions.keys()):  # Sort keys to ensure deterministic order
        subset = partitions[key]
        new_attributes = attributes[:]
        new_attributes.pop(best_attribute_index)
        filtered_subset = [
            [value for col_index, value in enumerate(row) if col_index != best_attribute_index]
            for row in subset
        ]

        # Recursive call to create the subtree
        subtree = tdidt(filtered_subset, new_attributes, len(data))
        tree.append(["Value", key, subtree])

    return tree

def majority_class(instances):
    counts = {}
    for instance in instances:
        label = instance[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return max(counts, key=counts.get)
