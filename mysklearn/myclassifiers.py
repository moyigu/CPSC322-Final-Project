"""
Name: Danni Du, Lida Chen
Date: 11/20
Assignment: PA7
Description: a class for pa7.ipynb.
            Define various classifiers.
"""
import graphviz
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import random
from mysklearn import myevaluation

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        instances_d = myutils.discretize_data(X_train)
        instances = [instances_d[i] + [str(y_train[i])] for i in range(len(X_train))]
        attributes = [f"att{i}" for i in range(len(X_train[0]))]
        self.tree = myutils.tdidt(instances, attributes)
        return instances

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_d = [f"{i}" for i in self.y_train]
        fallback_label = myutils.majority_class(y_d)
        y_predicted = []
        X_test_d = myutils.discretize_data(X_test)
        for instance in X_test_d:
            prediction = myutils.predict_instance(instance, self.tree, fallback_label)
            y_predicted.append(prediction)
        
        return myutils.convert_str_to_int(y_predicted)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.traverse_tree(self.tree, [], attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        dot = graphviz.Digraph()

        # Use the helper function to add nodes and edges
        myutils.add_nodes_edges(dot, self.tree, attribute_names=attribute_names)

        # Save and render the DOT file
        dot.render(dot_fname, format="pdf", cleanup=True)


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        prediction = self.regressor.predict(X_test)
        prediction = [self.discretizer(y) for y in prediction]
        return prediction

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        return myutils.determine_and_calculate_dist(X_test, self.X_train, self.n_neighbors)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for i in range(len(distances)):
            top_k_y = []
            for index in neighbor_indices[i]:
                top_k_y.append(self.y_train[index])
            y_predicted.append(myutils.most_frequent(top_k_y))
        return y_predicted
            

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self, strategy = "most_frequent"):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None
        self.strategy = strategy

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        if self.strategy == "most_frequent":
            self.most_common_label = myutils.most_frequent(y_train)
        elif self.strategy == "stratified":
            self.most_common_label = myutils.calculate_frequency(y_train)
        return self.most_common_label

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.strategy == "most_frequent":
            return [self.most_common_label] * len(X_test)
        elif self.strategy == "stratified":
            pred = []
            for i in range(len(X_test)):
                pred.append(myutils.ramdonly_choose(self.most_common_label)[0])
            return pred

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        X_train_d = myutils.discretize_data(X_train)
        y_train = myutils.convert_int_to_str(y_train)
        total_instances = len(y_train)
        unique_classes = set(y_train)
        for c in unique_classes:
            self.priors[c] = y_train.count(c) / total_instances

        self.posteriors = {}
        num_features = len(X_train_d[0])

        for c in unique_classes:
            self.posteriors[c] = [{} for _ in range(num_features)]
            class_indices = [i for i, label in enumerate(y_train) if label == c]
            class_feature_values = [X_train_d[i] for i in class_indices]
            for j in range(num_features):
                feature_values = [row[j] for row in class_feature_values]
                value_counts = {}
                for value in feature_values:
                    value_counts[value] = value_counts.get(value, 0) + 1
                for value, count in value_counts.items():
                    self.posteriors[c][j][value] = count / len(feature_values)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        X_test_d = myutils.discretize_data(X_test)
        most_common_label = max(self.priors, key=self.priors.get)

        for instance in X_test_d:
            class_probabilities = {}
            for c in self.priors:
                probability = self.priors[c]
                for j, feature_value in enumerate(instance):
                    if j < len(self.posteriors[c]) and feature_value in self.posteriors[c][j]:
                        probability *= self.posteriors[c][j][feature_value]
                    else:
                        probability *= 1e-6
                class_probabilities[c] = probability
            
            if all(p == 0 for p in class_probabilities.values()):
                predicted_class = most_common_label
            else:
                predicted_class = max(class_probabilities, key=class_probabilities.get)
            y_predicted.append(predicted_class)

        return myutils.convert_str_to_int(y_predicted)

class MyRandomForestClassifier:
    """
    Represents a random forest classifier using multiple decision trees.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_features (int): Maximum number of features to consider for each tree.
        trees (list of MyDecisionTreeClassifier): The ensemble of decision trees in the forest.
    """
    def __init__(self, n_trees=10, max_features=None):
        """
        Initialize the random forest classifier.

        Args:
            n_trees (int): Number of trees in the forest.
            max_features (int): Maximum number of features to consider for each tree. Defaults to None,
                                which means using all features.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X_train, y_train):
        """
        Train the random forest classifier on the training data.

        Args:
            X_train (list of list of obj): The training feature data.
            y_train (list of obj): The training target labels.
        """
        n_samples, n_features = len(X_train), len(X_train[0])
        if self.max_features is None:
            self.max_features = n_features  # Use all features if not specified
        
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling
            sampled_indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_bootstrap = [X_train[i] for i in sampled_indices]
            y_bootstrap = [y_train[i] for i in sampled_indices]
            
            # Random feature selection
            feature_indices = random.sample(range(n_features), self.max_features)
            
            # Subset the features
            X_bootstrap_subset = [[row[i] for i in feature_indices] for row in X_bootstrap]
            
            # Train a decision tree
            tree = MyDecisionTreeClassifier()
            tree.fit(X_bootstrap_subset, y_bootstrap)
            
            # Store the tree and its corresponding feature indices
            self.trees.append((tree, feature_indices))

    def predict(self, X_test):
        """
        Predict the target labels for the test data using majority voting.

        Args:
            X_test (list of list of obj): The test feature data.

        Returns:
            list of obj: The predicted target labels.
        """
        # Collect predictions from all trees
        all_predictions = []
        for tree, feature_indices in self.trees:
            # Subset test data to match the tree's trained features
            X_test_subset = [[row[i] for i in feature_indices] for row in X_test]
            predictions = tree.predict(X_test_subset)
            all_predictions.append(predictions)
        
        # Transpose to get predictions per instance
        all_predictions = list(zip(*all_predictions))
        
        # Majority voting for each instance
        final_predictions = []
        for instance_preds in all_predictions:
            # Count occurrences of each class
            vote_count = {}
            for pred in instance_preds:
                if pred in vote_count:
                    vote_count[pred] += 1
                else:
                    vote_count[pred] = 1
            
            # Find the class with the maximum votes
            majority_vote = max(vote_count, key=vote_count.get)
            final_predictions.append(majority_vote)

        return myutils.convert_str_to_int(final_predictions)

    def print_forest_summary(self):
        """
        Print a summary of the forest including the number of trees and max features used.
        """
        print(f"Random Forest Summary:")
        print(f"- Number of Trees: {self.n_trees}")
        print(f"- Max Features per Tree: {self.max_features}")
