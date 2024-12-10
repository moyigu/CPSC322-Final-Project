import pickle
# we are going to use the Flask micro web framework
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the NEO Predict app</h1>", 200

