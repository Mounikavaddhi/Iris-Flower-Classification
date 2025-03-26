# 🌸 Iris Flower Classification 🌸  

This project is a **machine learning model** that classifies iris flowers into three species: **Setosa, Versicolor, and Virginica**.  
It includes a **Flask web application** where users can input flower measurements and get predictions. 🚀  

---

## 📂 Project Structure  

Iris_Project/ │── iris_model.py # Machine learning model training │── iris_model.pkl # Saved trained model │── app.py # Flask web application │── templates/ │ ├── index.html # Web page for user input │── IRIS.csv # Dataset │── README.md # Project documentation


---

## 🔍 Dataset  
The dataset used is the **Iris Flower Dataset**, which contains:  
- 🌱 **Features**: `Sepal Length`, `Sepal Width`, `Petal Length`, `Petal Width`  
- 🎯 **Target (Species)**: `Setosa`, `Versicolor`, `Virginica`  

Dataset Source: [Kaggle - Iris Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)  

---

## ⚙️ How It Works  
1️⃣ **Train the Model:**  
   - The dataset is loaded and preprocessed in `iris_model.py`.  
   - A `RandomForestClassifier` is trained and saved as `iris_model.pkl`.  

2️⃣ **Run the Web App:**  
   ```bash
   python app.py
