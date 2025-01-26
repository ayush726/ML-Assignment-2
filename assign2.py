import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

input_file_x = r'C:\Users\YourPath\logistic_inputs.csv'
input_file_y = r'C:\Users\YourPath\logistic_outputs.csv'

features = pd.read_csv(input_file_x, header=None)
labels = pd.read_csv(input_file_y, header=None).values.ravel()

print("Feature Set (features):")
print(features.head())
print("\nLabel Set (labels):")
print(labels[:5])

log_reg_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg_model.fit(features, labels)

model_weights = log_reg_model.coef_
model_bias = log_reg_model.intercept_

print("Model Weights:", model_weights)
print("Model Bias:", model_bias)

def activation_function(z):
    return 1 / (1 + np.exp(-z))

def calculate_cost(features, labels, params):
    num_samples = len(labels)
    predictions = activation_function(np.dot(features, params))
    cost = (-1 / num_samples) * (np.dot(labels, np.log(predictions)) + np.dot((1 - labels), np.log(1 - predictions)))
    return cost

def optimize_params(features, labels, params, lr, epochs):
    num_samples = len(labels)
    cost_history = []
    for _ in range(epochs):
        predictions = activation_function(np.dot(features, params))
        gradients = (1 / num_samples) * np.dot(features.T, (predictions - labels))
        params -= lr * gradients
        cost_history.append(calculate_cost(features, labels, params))
    return params, cost_history

features_with_bias = np.c_[np.ones((features.shape[0], 1)), features]
params_init = np.zeros(features_with_bias.shape[1])

learning_rate = 0.1
num_iterations = 50
params, cost_values = optimize_params(features_with_bias, labels, params_init, learning_rate, num_iterations)

plt.plot(range(num_iterations), cost_values)
plt.xlabel("Iterations")
plt.ylabel("Cost Value")
plt.title("Training Loss Curve")
plt.show()

plt.scatter(features[0], features[1], c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data and Decision Boundary (Basic Model)")

x_vals = np.linspace(features[0].min(), features[0].max(), 100)
y_vals = -(params[1] * x_vals + params[0]) / params[2]
plt.plot(x_vals, y_vals, color='blue', label="Decision Boundary")
plt.legend()
plt.show()

poly_transform = PolynomialFeatures(degree=2, include_bias=False)
features_poly = poly_transform.fit_transform(features)

log_reg_model.fit(features_poly, labels)

plt.scatter(features[0], features[1], c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data and Decision Boundary (Polynomial Model)")

x1_range = np.linspace(features[0].min(), features[0].max(), 100)
x2_range = np.linspace(features[1].min(), features[1].max(), 100)
grid_x1, grid_x2 = np.meshgrid(x1_range, x2_range)

grid_predictions = log_reg_model.predict(poly_transform.transform(np.c_[grid_x1.ravel(), grid_x2.ravel()]))
grid_predictions = grid_predictions.reshape(grid_x1.shape)

plt.contour(grid_x1, grid_x2, grid_predictions, levels=[0.5], colors='red')
plt.show()

predicted_labels = log_reg_model.predict(features_poly)
conf_matrix = confusion_matrix(labels, predicted_labels)

acc = accuracy_score(labels, predicted_labels)
prec = precision_score(labels, predicted_labels)
rec = recall_score(labels, predicted_labels)
f1 = f1_score(labels, predicted_labels)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-Score: {f1:.2f}")
