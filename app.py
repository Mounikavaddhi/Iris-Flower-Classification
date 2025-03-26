from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("iris_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get input values from the form
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        
        species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        return render_template("index.html", result=species[prediction[0]])
    
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
