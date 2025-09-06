Decision Tree Classifier Project
Project Overview

This project demonstrates the use of a Decision Tree Classifier using Scikit-learn to classify and predict outcomes on a dataset. The model is trained, evaluated, and visualized to understand decision-making steps.

Dataset

Dataset Used: Iris dataset (built-in in Scikit-learn)

Features: Sepal length, Sepal width, Petal length, Petal width

Target Classes: Setosa, Versicolor, Virginica

The dataset is clean and suitable for visualizing decision trees.

Project Steps

Data Loading & Exploration

Load dataset into a DataFrame

Check for missing values and explore feature statistics

Train-Test Split

Split dataset into training and testing sets (70%-30%)

Model Building

Build a Decision Tree Classifier using entropy criterion

Limit tree depth to avoid overfitting

Model Evaluation

Calculate accuracy, classification report, and confusion matrix

Evaluate model performance on test data

Visualization

Decision Tree Plot showing splits and leaf nodes

Feature Importance Plot highlighting key features

Results

The model achieves high accuracy (~95%+) on the test data

Petal length and petal width are the most important features

Tree visualization shows step-by-step decision-making

Technologies Used

Python 3

Jupyter Notebook

Pandas & NumPy

Matplotlib & Seaborn

Scikit-learn

How to Use

Clone the repository:

git clone <repository_url>


Open the Jupyter Notebook:

jupyter notebook Decision_Tree_Model.ipynb


Run all cells to explore data, train the model, and visualize the tree.

Conclusion

This project provides a hands-on understanding of Decision Trees, including data preprocessing, model training, evaluation, and visualization. It demonstrates how interpretable models can make accurate predictions.
