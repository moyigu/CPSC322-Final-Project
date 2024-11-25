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