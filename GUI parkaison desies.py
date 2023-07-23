import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to load the dataset from a CSV file
def load_dataset():
    file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        X = data.drop(columns=['Parkinsons'])
        y = data['Parkinsons']
        return X, y
    return None, None

# Function to train the model and display the results
def train_model():
    X, y = load_dataset()
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = SVC(kernel='linear')  # You can experiment with other models as well
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        result_label.config(text=f"Accuracy: {accuracy:.2f}\n\nConfusion Matrix:\n{conf_matrix}\n\nClassification Report:\n{classification_rep}")

# Create the main application window
app = tk.Tk()
app.title("Parkinson's Disease Detection")

# Button to load the dataset
load_button = tk.Button(app, text="Load Dataset", command=load_dataset)
load_button.pack(pady=10)

# Button to train the model and show results
train_button = tk.Button(app, text="Train Model", command=train_model)
train_button.pack(pady=10)

# Label to display the model evaluation results
result_label = tk.Label(app, text="")
result_label.pack(pady=10)

app.mainloop()
