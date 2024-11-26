import numpy as np # use numpy's random number generation

from mysklearn import myutils

# TODO: copy your myevaluation.py solution from PA5 here
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state + 100)
    n_samples = len(X)
    if isinstance(test_size, float):
        if int(round(test_size * n_samples)) < test_size * n_samples:
            test_size = int(round(test_size * n_samples) + 1)
        else:
            test_size = int(round(test_size * n_samples))
    elif isinstance(test_size, int):
        if test_size > n_samples:
            raise ValueError("test_size cannot be greater than the number of samples")

    if shuffle:
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        X_shuffled = [X[i] for i in indices]
        y_shuffled = [y[i] for i in indices]
    else:
        X_shuffled = X
        y_shuffled = y

    X_train = X_shuffled[:-test_size]
    X_test = X_shuffled[-test_size:]
    y_train = y_shuffled[:-test_size]
    y_test = y_shuffled[-test_size:]

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = list(range(n_samples))
    if shuffle:
        np.random.shuffle(indices)
    base_size = n_samples // n_splits
    extra = n_samples % n_splits
    # Calculate fold sizes
    fold_sizes = []
    for i in range(n_splits):
        if i < extra:
            fold_sizes.append(base_size+1)
        else:
            fold_sizes.append(base_size)
    # Create the folds
    current = 0
    folds = []
    for size in fold_sizes:
        test_indices = indices[current:current + size]
        train_indices = indices[:current] + indices[current + size:]
        folds.append((train_indices, test_indices))
        current += size

    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = list(range(n_samples))
    if shuffle:
        np.random.shuffle(indices)
    dist = {}
    for idx in indices:
        if y[idx] not in dist:
            dist[y[idx]] = []
        dist[y[idx]].append(idx)
    folds = [[] for _ in range(n_splits)]
    for _, label_indices in dist.items():
        fold_sizes = [len(label_indices) // n_splits] * n_splits
        for i in range(len(label_indices) % n_splits):
            fold_sizes[i] += 1
        current = 0
        for i, fold_size in enumerate(fold_sizes):
            folds[i].extend(label_indices[current:current + fold_size])
            current += fold_size
    stratified_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for j in range(n_splits) if j != i for idx in folds[j]]
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state is not None:
        np.random.seed(random_state)
    if n_samples is None:
        n_samples = len(X)
    indices = np.random.randint(0, len(X), size=n_samples)

    X_sample = [X[i] for i in indices]
    if y is not None:
        y_sample = [y[i] for i in indices]
    else:
        y_sample = None

    out_of_bag_indices = set(range(len(X))) - set(indices)
    X_out_of_bag = [X[i] for i in out_of_bag_indices]
    if y is not None:
        y_out_of_bag = [y[i] for i in out_of_bag_indices]
    else:
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    label_to_index = {label: index for index, label in enumerate(labels)}

    matrix = [[0] * len(labels) for _ in range(len(labels))]

    true_indices = [label_to_index[true] for true in y_true]
    pred_indices = [label_to_index[pred] for pred in y_pred]

    for true_idx, pred_idx in zip(true_indices, pred_indices):
        matrix[true_idx][pred_idx] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = 0
    for true, pred in zip(y_true,y_pred):
        if true == pred:
            correct += 1
    if normalize:
        score = correct / len(y_pred)
    else:
        score = correct
    return score # TODO: fix this

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    tp = 0
    fp = 0
    for true, pred in zip(y_true, y_pred):
        if pred == pos_label:
            if true == pos_label:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    tp = 0
    fn = 0
    for true, pred in zip(y_true,y_pred):
        if true == pos_label:
            if pred == pos_label:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    return recall # TODO: fix this

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    precision = binary_precision_score(y_true,y_pred,labels,pos_label)
    recall = binary_recall_score(y_true,y_pred,labels,pos_label)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1 # TODO: fix this

