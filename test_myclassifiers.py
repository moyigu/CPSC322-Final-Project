import numpy as np
from scipy import stats
import random

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.classifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier
from mysklearn.classifiers import MyNaiveBayesClassifier

from mysklearn.classifiers import MyDecisionTreeClassifier
from mysklearn.classifiers import MyRandomForestClassifier

# TODO: copy your test_myclassifiers.py solution from PA4-6 here

# interview dataset
header_NEO = ["est_diameter_min","est_diameter_max","relative_velocity","miss_distance","absolute_magnitude","hazardous"]
X_train_NEO = [
    [1.1982708007,2.6794149658,13569.2492241812,54839744.08284605,16.73],
    [0.2658,0.5943468684,73588.7266634981,61438126.52395093,20.0],
    [0.7220295577,1.6145071727,114258.6921290512,49798724.94045679,17.83],
    [0.096506147,0.2157943048,24764.3031380016,25434972.72075825,22.2],
    [0.2550086879,0.5702167609,42737.7337647264,46275567.00130072,20.09],
    [0.0363542322,0.0812905344,34297.5877783029,40585691.22792288,24.32],
    [0.1716148941,0.3837425691,27529.4723069673,29069121.41864897,20.95]
]
y_train_NEO = ["False","True","False","False","True","False","False"]

tree_NEO = []

header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]

y_train_iphone = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
    "yes"
]

tree_iphone = ['Attribute', 'att0', 
                    ['Value', 1, 
                        ['Attribute', 'att1', 
                            ['Value', 1, 
                                ['Leaf', 'yes', 1, 5]
                            ], 
                            ['Value', 2, 
                                ['Attribute', 'att2', 
                                    ['Value', 'fair', 
                                        ['Leaf', 'no', 1, 2]
                                    ], 
                                    ['Value', 'excellent', 
                                        ['Leaf', 'yes', 1, 2]
                                    ]
                                ]
                            ], 
                            ['Value', 3, 
                                ['Leaf', 'no', 2, 5]
                            ]
                        ]
                    ], 
                    ['Value', 2, 
                        ['Attribute', 'att2', 
                            ['Value', 'excellent', 
                                ['Attribute', 'att1', 
                                    ['Value', 1, 
                                        ['Leaf', 'no', 2, 4]
                                    ], 
                                    ['Value', 2, 
                                        ['Leaf', 'yes', 2, 4]
                                    ]
                                ]
                            ], 
                            ['Value', 'fair', 
                                ['Leaf', 'yes', 6, 10]
                            ]
                        ]
                    ]
                ]

# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# MA7 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain", "class"]
X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]

# TODO: copy your test_myclassifiers.py solution from PA4-5 here
# from in-class #1  (4 instances)
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

# from in-class #2 (8 instances)
# assume normalized
X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# from Bramer
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]

def discretizer(y_pred):
    return "high" if y_pred >=100 else "low"
# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)
    X_train = np.random.randint(1,50,size=(50,1))
    y_train = [(2 * x + np.random.normal(0,5)) for x in X_train]
    classifier = MySimpleLinearRegressionClassifier(discretizer=discretizer)
    classifier.fit(X_train,y_train)
    assert np.isclose(classifier.regressor.slope, 2, atol=0.5)
    assert np.isclose(classifier.regressor.intercept, 0, atol=1)

def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)
    X_train = np.random.randint(1,50,size=(50,1))
    y_train = [(2 * x + np.random.normal(0,5)) for x in X_train]
    classifier = MySimpleLinearRegressionClassifier(discretizer=discretizer)
    classifier.fit(X_train,y_train)
    x_test = np.random.randint(1,50,size=(2,1))
    y_test = [(classifier.regressor.slope * x + classifier.regressor.intercept) for x in X_train]
    dis_y = [discretizer(y) for y in y_test]
    y_pred = classifier.predict(x_test)
    for i,y in enumerate(y_pred):
        assert y == dis_y[i]

def test_kneighbors_classifier_kneighbors():
    classifier1 = MyKNeighborsClassifier(n_neighbors=3)
    classifier1.fit(X_train_class_example1,y_train_class_example1)
    dist, neighbor = classifier1.kneighbors([[0.33,1]])
    expect_dis = [[0.67,1,1.053]]
    expect_neighbor = [[0,2,3]]
    assert np.isclose(dist, expect_dis, atol= 0.01).all
    assert neighbor == expect_neighbor

    classifier2 = MyKNeighborsClassifier(n_neighbors=3)
    classifier2.fit(X_train_class_example2,y_train_class_example2)
    dist, neighbor = classifier2.kneighbors([[2,3]])
    expect_dis = [[1.41421356,1.41421356,2.0]]
    expect_neighbor = [[0, 4, 6]]
    assert np.isclose(dist, expect_dis, atol= 0.01).all
    assert neighbor == expect_neighbor

    classifier3 = MyKNeighborsClassifier(n_neighbors=5)
    classifier3.fit(X_train_bramer_example,y_train_bramer_example)
    dist, neighbor = classifier3.kneighbors([[9.1,11.0]])
    expect_dis = [[2.802, 1.237, 0.608, 2.202, 2.915]]
    expect_neighbor = [[6,5,7,4,8]]
    assert np.isclose(dist, expect_dis, atol= 0.01).all
    assert neighbor == expect_neighbor


def test_kneighbors_classifier_predict():
    classifier1 = MyKNeighborsClassifier(n_neighbors=3)
    classifier1.fit(X_train_class_example1,y_train_class_example1)
    y_pred = classifier1.predict(X_test=[[0.33,1]])
    assert y_pred == ['good']

    classifier2 = MyKNeighborsClassifier(n_neighbors=3)
    classifier2.fit(X_train_class_example2,y_train_class_example2)
    y_pred = classifier2.predict(X_test=[[2,3]])
    assert y_pred == ['yes']

    classifier3 = MyKNeighborsClassifier(n_neighbors=5)
    classifier3.fit(X_train_bramer_example,y_train_bramer_example)
    y_pred = classifier3.predict(X_test=[[9.1,11.0]])
    assert y_pred == ['+']
    

def test_dummy_classifier_fit():
    X_train = np.random.rand(100, 5)
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    classfier = MyDummyClassifier()
    classfier.fit(X_train,y_train)
    assert classfier.most_common_label == 'yes'

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    classfier.fit(None,y_train)
    assert classfier.most_common_label == 'no'

    y_train = list(np.random.choice(["rain", "sunny", "cloudy"], 100, replace=True, p=[0.1, 0.7, 0.2]))
    classfier.fit(None,y_train)
    assert classfier.most_common_label == 'sunny'

def test_dummy_classifier_predict():
    X_train = np.random.rand(100, 5)
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    classfier = MyDummyClassifier()
    classfier.fit(X_train,y_train)
    x_test = [6]
    assert classfier.predict(x_test) == ["yes" for _ in range(len(x_test))]

    X_train = np.random.rand(100, 5)
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    classfier.fit(X_train,y_train)
    assert classfier.predict(x_test) == ["no" for _ in range(len(x_test))]

    X_train = np.random.rand(100, 5)
    y_train = list(np.random.choice(["rain", "sunny", "cloudy"], 100, replace=True, p=[0.1, 0.7, 0.2]))
    classfier.fit(X_train,y_train)
    assert classfier.predict(x_test) == ["sunny" for _ in range(len(x_test))]


def test_naive_bayes_classifier_fit():
    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example,y_train_inclass_example)
    priors = {"yes": 5/8, "no": 3/8}
    posteriors = {"yes": [{1: 4 / 5, 2: 1 / 5}, {5: 2 / 5, 6: 3 / 5}], "no": [{1: 2 / 3, 2: 1 / 3}, {5: 2 / 3, 6: 1 / 3}]}
    assert classifier.priors == priors, "Failed priors test on in-class example"
    assert classifier.posteriors == posteriors, "Failed posteriors test on in-class example"

    classifier.fit(X_train_iphone, y_train_iphone)
    priors = {"yes": 10 / 15, "no": 5 / 15}
    posteriors = { "yes": [{1: 2 / 10, 2: 8 / 10},{3: 3 / 10, 2: 4 / 10, 1: 3 / 10},{"fair": 7 / 10, "excellent": 3 / 10}],"no": [{1: 3 / 5, 2: 2 / 5},{3: 2 / 5, 2: 2 / 5, 1: 1 / 5},{"fair": 2 / 5, "excellent": 3 / 5}]}
    assert classifier.priors == priors, "Failed priors test on MA7 dataset"
    assert classifier.posteriors == posteriors, "Failed posteriors test on MA7 dataset"

    classifier.fit(X_train_train,y_train_train)
    priors = {"on time": 15 / 20, "late": 2 / 20, "very late": 2 / 20, "cancelled": 1 / 20}
    posteriors = {"on time": [{"weekday": 12 / 15, "saturday": 2 / 15, "holiday": 1 / 15},{"spring": 4 / 15, "winter": 3 / 15, "summer": 4 / 15, "autumn": 4 / 15},{"none": 5 / 15, "normal": 8 / 15, "high": 2 / 15},{"none": 7 / 15, "slight": 5 / 15, "heavy": 3 / 15}],
    "late": [{"weekday": 1 / 2, "saturday": 1 / 2},{"winter": 1 / 2, "summer": 1 / 2},{"high": 1 / 2, "normal": 1 / 2},{"none": 1 / 2, "heavy": 1 / 2}],
    "very late": [{"weekday": 1 / 2, "saturday": 1 / 2},{"winter": 1 / 2, "autumn": 1 / 2},{"high": 1 / 2, "normal": 1 / 2},{"heavy": 1 / 2, "slight": 1 / 2}],
    "cancelled": [{"saturday": 1.0},{"spring": 1.0},{"high": 1.0},{"heavy": 1.0}]}
def test_naive_bayes_classifier_predict():
    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example,y_train_inclass_example)
    assert classifier.predict([[1,5],[2,6]]) == ["yes","yes"]
    classifier.fit(X_train_iphone,y_train_iphone)
    assert classifier.predict([[1,1,'excellent'],[2,2,'fair']]) == ["no","yes"]
    classifier.fit(X_train_train,y_train_train)
    assert classifier.predict([['weekday','winter','high','heavy']]) == ['very late']

def test_decision_tree_classifier_fit():
    classifier = MyDecisionTreeClassifier()
    classifier.fit(X_train_interview,y_train_interview)
    assert classifier.tree == tree_interview
    classifier.fit(X_train_iphone,y_train_iphone)
    assert classifier.tree == tree_iphone

def test_decision_tree_classifier_predict():
    classifier = MyDecisionTreeClassifier()
    classifier.fit(X_train_interview,y_train_interview)
    predict = classifier.predict([["Junior", "Java", "yes", "no"]])
    assert predict == ["True"]
    predict = classifier.predict([["Junior", "Java", "yes", "yes"]])
    assert predict == ["False"]
    classifier.fit(X_train_iphone,y_train_iphone)
    predict = classifier.predict([[2, 2, "fair"]])
    assert predict == ["yes"]
    predict = classifier.predict([[1, 1, "excellent"]])
    assert predict == ["yes"]

def test_random_tree_classifier_fit():
    classifier = MyRandomForestClassifier(n_trees=10, max_features=5)
    classifier.fit(X_train_NEO, y_train_NEO)
    
    assert len(classifier.trees) == 10, "Random forest should contain 10 trees."
    for i, (tree, feature_indices) in enumerate(classifier.trees):
        assert len(feature_indices) == 5, f"Tree {i + 1} should have 3 selected features."
        assert tree.tree is not None, f"Tree {i + 1} is not properly trained."  
        for index in feature_indices:
            assert 0 <= index <= len(header_NEO), f"Tree {i + 1} has invalid feature index {index}."

def test_random_tree_classifier_predict():
    random.seed(42)
    X_test = [
        [0.5, 1.0, 50000, 45000000, 19.0],
        [0.2, 0.4, 30000, 35000000, 21.0],
        [1.0, 2.5, 80000, 60000000, 18.0]
    ]
    expected_predictions = ["True", "False", "True"]
    classifier = MyRandomForestClassifier(n_trees=10, max_features=3)
    classifier.fit(X_train_NEO, y_train_NEO)
    y_pred = classifier.predict(X_test)
    assert len(y_pred) == len(X_test), "The number of predictions should match the number of test instances."
    assert y_pred == expected_predictions, f"Expected predictions: {expected_predictions}, but got: {y_pred}"

