from mysklearn import utils

# TODO: copy your myclassifiers.py solution from PA4-6 here
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import random
import os
from graphviz import Digraph

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
        predictions = self.regressor.predict(X_test)
        y_predicted = [self.discretizer(pred) for pred in predictions]
        
        return y_predicted

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
            neighbor_neighbor_indices(list of list of int): 2D list of k nearest neighbor
                neighbor_indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        # Iterate through each test instance
        for x_test in X_test:
            tuples = []  # (distance, index) pairs for each training instance

            # Compare against each training instance
            for i, x_train in enumerate(self.X_train):
                total_distance = 0

                # Calculate distance for each feature
                for j, feature in enumerate(x_test):
                    if isinstance(feature, (int, float)):  # Numeric feature
                        total_distance += (feature - x_train[j]) ** 2
                    else:  # Categorical feature
                        if feature == x_train[j]:  # Same value
                            total_distance += 0
                        else:  # Different value
                            total_distance += 1  # Nominal difference; or use an ordinal encoding if applicable

                # Append the distance and index for the current training instance
                tuples.append((total_distance ** 0.5, i))

            # Sort by distance and select the k nearest neighbors
            sorted_tuples = sorted(tuples)[:self.n_neighbors]
            distances.append([dist for dist, idx in sorted_tuples])
            neighbor_indices.append([idx for dist, idx in sorted_tuples])

        return distances, neighbor_indices
                
        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        dist, neighbor = self.kneighbors(X_test)
        y_test = []
        for j in range(len(neighbor)):
            temp = []
            for i in neighbor[j]:
                temp.append(self.y_train[i])
            y_test.append(temp)
        y_predicted = utils.predict(y_test)
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
            # Calculate the most frequent class label for 'most_frequent' strategy
            frequency = {item: y_train.count(item) for item in y_train}
            self.most_common_label = max(frequency, key=frequency.get)

        elif self.strategy == "stratified":
            # Calculate class probabilities for 'stratified' strategy
            total_count = len(y_train)
            frequency = {item: y_train.count(item) for item in set(y_train)}
            self.class_probabilities = {label: count / total_count for label, count in frequency.items()}


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        if self.strategy == "most_frequent":
            # Return the most common label for all test instances
            return [self.most_common_label for _ in range(len(X_test))]

        elif self.strategy == "stratified":
            # Use the class probabilities to generate stratified predictions
            labels = list(self.class_probabilities.keys())
            probabilities = list(self.class_probabilities.values())
            return [random.choices(labels, weights=probabilities)[0] for _ in range(len(X_test))]

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
        # prior
        self.priors = {}
        total_y = len(y_train)
        for value in y_train:
            if value not in self.priors:
                self.priors[value] = 1
            else:
                self.priors[value] += 1
        self.priors = {label: count/total_y for label, count in self.priors.items()}
        # posteriors
        att_num = len(X_train[0])
        self.posteriors= {}
        for value in self.priors:
            self.posteriors[value] = []
            for i in range(att_num):
                self.posteriors[value].append({})
        for x, label in zip(X_train,y_train):
            for i in range(att_num):
                if x[i] not in self.posteriors[label][i]:
                    self.posteriors[label][i][x[i]] = 1
                else:
                    self.posteriors[label][i][x[i]] += 1
        for label in self.posteriors:
            for i in range(att_num):
                total = sum(self.posteriors[label][i].values())
                self.posteriors[label][i] = {value: count/total for value, count in self.posteriors[label][i].items()}


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for value in X_test:
            prob = {}
            for label in self.priors:
                prob[label] = self.priors[label]
                for i, feature in enumerate(value):
                    if feature in self.posteriors[label][i]:
                        prob[label] *= self.posteriors[label][i][feature]
                    else:
                        prob[label] *= 0
            pred = max(prob,key=prob.get)
            y_pred.append(pred)  


        return y_pred

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

        # Generate default attribute names
        attribute_names = [f"att{i}" for i in range(len(X_train[0]))]

        # Combine training data with labels
        training_data = [row + [label] for row, label in zip(X_train, y_train)]

        # Use external TDIDT function to build the tree
        self.tree = utils.tdidt(training_data, attribute_names, len(training_data))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predict = []
        for instance in X_test:
            tree = self.tree
            while tree[0] != "Leaf":
                attribute_index = int(tree[1][3:])
                instance_value = instance[attribute_index]

                found_branch = False
                for i in range(2, len(tree)):
                    if tree[i][1] == instance_value:
                        tree = tree[i][2]
                        found_branch = True
                        break
                
                if not found_branch:
                # Randomly select a branch if no matching branch is found
                    random_branch = random.choice(tree[2:])
                    tree = random_branch[2]
            y_predict.append(tree[1])
        return y_predict

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
        element = [(self.tree, [])]  # Each element is (current_node, rule_conditions)
    
        while element:
            current_node, rule_conditions = element.pop()
            
            if current_node[0] == "Leaf":
                # Leaf node: Print the rule
                rule = "IF " + " AND ".join(rule_conditions)
                rule += f" THEN {class_name} = {current_node[1]}"
                print(rule)
            else:
                # Attribute node
                attribute = current_node[1]
                attribute_name = (
                    attribute_names[int(attribute[3:])] if attribute_names else attribute
                )
                
                for i in range(2, len(current_node)):
                    value = current_node[i][1]
                    # Add new rule condition and push subtree to element
                    element.append((current_node[i][2], rule_conditions + [f"{attribute_name} == {value}"]))

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """
        Visualizes a decision tree using Graphviz and generates a .dot and .pdf file.

        Args:
            dot_fname (str): The name of the .dot output file.
            pdf_fname (str): The name of the .pdf output file generated from the .dot file.
            attribute_names (list of str or None): A list of attribute names to use in the visualization
                (None if a list is not provided, in which case default attribute names based on
                indexes (e.g., "att0", "att1", ...) will be used).
        """
        def add_nodes_edges(dot, tree, parent_name=None):
            """Recursively adds nodes and edges to the Graphviz Digraph."""
            if tree[0] == "Leaf":
                # Leaf node
                node_name = f'leaf_{id(tree)}'  # Unique identifier for each node
                label = f"Class = {tree[1]}\n({tree[2]}/{tree[3]})"
                dot.node(node_name, label=label, shape="ellipse", style="filled", color="lightblue")
                if parent_name:
                    dot.edge(parent_name, node_name)
            else:
                # Attribute node
                attribute_name = (
                    attribute_names[int(tree[1][3:])] if attribute_names else tree[1]
                )
                node_name = f'node_{id(tree)}'  # Unique identifier for each node
                dot.node(node_name, label=attribute_name, shape="box", style="rounded,filled", color="lightyellow")
                if parent_name:
                    dot.edge(parent_name, node_name)

                # Recursively process children
                for i in range(2, len(tree)):
                    value = tree[i][1]
                    child_name = f'{node_name}_{value}'  # Unique edge name
                    add_nodes_edges(dot, tree[i][2], node_name)

        # Create a Graphviz Digraph
        dot = Digraph()

        # Add nodes and edges recursively
        add_nodes_edges(dot, self.tree)

        # Save the .dot file and render it as a .pdf
        dot.render(filename=dot_fname, format="pdf", cleanup=True)

        # Ensure the generated PDF is saved in the required directory
        os.makedirs("tree_vis", exist_ok=True)
        os.rename(f"{dot_fname}.pdf", f"tree_vis/{pdf_fname}.pdf")
