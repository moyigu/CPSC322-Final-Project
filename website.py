import pickle
from flask import Flask, request, jsonify
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier
import mysklearn.myutils as myutils

# Function to load and prepare the dataset
def load_and_prepare_data(file_path):
    """
    Load and prepare the NEO dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - X (list of list of float): Normalized feature data.
            - y (list of int): Target labels (0 for False, 1 for True).
    """
    # Load data
    table = MyPyTable().load_from_file(file_path)
    
    # Sample data for balanced classes
    sampled_data = table.get_sample_data("hazardous", 5000, 5000)
    sampled_table = MyPyTable(column_names=table.column_names, data=sampled_data)
    sampled_table.save_to_file("dataset//neo_new.csv")
    df = MyPyTable().load_from_file("dataset//neo_new.csv")

    # Normalize specified columns
    df.normalize_columns([
        "est_diameter_min", 
        "est_diameter_max", 
        "relative_velocity", 
        "miss_distance", 
        "absolute_magnitude"
    ])

    # Extract normalized features
    est_diameter_min_nor = df.get_column("est_diameter_min")
    est_diameter_max_nor = df.get_column("est_diameter_max")
    relative_velocity_nor = df.get_column("relative_velocity")
    miss_distance_nor = df.get_column("miss_distance")
    absolute_magnitude_nor = df.get_column("absolute_magnitude")

    # Prepare X (features)
    X = [
        [
            est_diameter_min_nor[i],
            est_diameter_max_nor[i],
            miss_distance_nor[i],
            relative_velocity_nor[i],
            absolute_magnitude_nor[i]
        ]
        for i in range(len(est_diameter_min_nor))
    ]

    # Prepare y (target labels)
    hazardous = df.get_column("hazardous")
    y = [0 if val == "False" else 1 for val in hazardous]

    return X, y

# Train and save the KNN model
def train_and_save_model():
    """
    Train the KNN model and save it to a file.
    """
    # Load the data
    X, y = load_and_prepare_data("dataset//neo_copy.csv")

    # Train the KNN classifier
    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Save the model to a file
    with open("knn_model.pkl", "wb") as model_file:
        pickle.dump(knn, model_file)
    print("Model trained and saved successfully!")

# Create the Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return '''
    <h1>Welcome to the NEO Prediction App</h1>
    <form action="/predict" method="POST">
        <label>Estimated Diameter Min (normalized, 0-1):</label><br>
        <input type="text" name="est_diameter_min"><br>
        <label>Estimated Diameter Max (normalized, 0-1):</label><br>
        <input type="text" name="est_diameter_max"><br>
        <label>Miss Distance (normalized, 0-1):</label><br>
        <input type="text" name="miss_distance"><br>
        <label>Relative Velocity (normalized, 0-1):</label><br>
        <input type="text" name="relative_velocity"><br>
        <label>Absolute Magnitude (normalized, 0-1):</label><br>
        <input type="text" name="absolute_magnitude"><br>
        <button type="submit">Predict</button>
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    try:
        est_diameter_min = float(request.form["est_diameter_min"])
        est_diameter_max = float(request.form["est_diameter_max"])
        miss_distance = float(request.form["miss_distance"])
        relative_velocity = float(request.form["relative_velocity"])
        absolute_magnitude = float(request.form["absolute_magnitude"])
    except ValueError:
        return jsonify({"error": "Invalid input. Please ensure all fields are filled with numeric values."}), 400

    # Load the saved model
    with open("knn_model.pkl", "rb") as model_file:
        knn = pickle.load(model_file)

    # Make a prediction
    X_test = [[est_diameter_min, est_diameter_max, miss_distance, relative_velocity, absolute_magnitude]]
    prediction = knn.predict(X_test)

    # Return the result
    result = "Hazardous" if prediction[0] == 1 else "Not Hazardous"
    return jsonify({"prediction": result})

# Train and save the model if running this script directly
if __name__ == "__main__":
    train_and_save_model()  # Train the model and save it before starting the Flask app
    app.run(host="127.0.0.1", port=8080, debug=True)  # Change port to 8080