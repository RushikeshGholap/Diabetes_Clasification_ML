import numpy as np
from decision_tree import *


import numpy as np


def myDT(Xtrain, Ytrain, Xvalid):

    class DecisionTree:
        def __init__(self):
            self.tree = None

        def fit(self, X, Y):
            self.tree = self._build_tree(X, Y)

        def predict(self, X):
            predictions = []
            for sample in X:
                predictions.append(self._predict_sample(sample, self.tree))
            return np.array(predictions)

        def _entropy(self, Y):
            classes, counts = np.unique(Y, return_counts=True)
            probabilities = counts / len(Y)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy

        def _information_gain(self, X, Y, feature_index):
            unique_values = np.unique(X[:, feature_index])
            total_entropy = self._entropy(Y)

            weighted_entropy = 0
            for value in unique_values:
                subset_indices = X[:, feature_index] == value
                subset_Y = Y[subset_indices]
                weighted_entropy += len(subset_Y) / len(Y) * self._entropy(subset_Y)

            information_gain = total_entropy - weighted_entropy
            return information_gain

        def _build_tree(self, X, Y, depth=0):
            num_samples, num_features = X.shape
            unique_classes = np.unique(Y)

            # Base cases
            if len(unique_classes) == 1:
                # If all samples belong to the same class, return a leaf node
                return {'class': unique_classes[0]}
            # Limit the depth for simplicity
            if depth == 1000:  
                # If maximum depth is reached, return a leaf node with the majority class
                majority_class = np.argmax(np.bincount(Y))
                return {'class': majority_class}

            # Find the best split
            best_feature = None
            best_information_gain = -1
            for feature_index in range(num_features):
                information_gain = self._information_gain(X, Y, feature_index)
                if information_gain > best_information_gain:
                    best_feature = feature_index
                    best_information_gain = information_gain

            if best_information_gain == 0:
                # If no information gain, return a leaf node with the majority class
                majority_class = np.argmax(np.bincount(Y))
                return {'class': majority_class}

            # Split the data based on the best feature
            unique_values = np.unique(X[:, best_feature])
            node = {'feature': best_feature, 'children': {}}
            for value in unique_values:
                subset_indices = X[:, best_feature] == value
                subset_X = X[subset_indices]
                subset_Y = Y[subset_indices]
                node['children'][value] = self._build_tree(subset_X, subset_Y, depth + 1)

            return node

        def _predict_sample(self, sample, tree):
            if 'class' in tree:
                # If leaf node, return the class
                return tree['class']
            else:
                # Recursively traverse the tree
                feature_value = sample[tree['feature']]
                if feature_value not in tree['children']:
                    # If the feature value is not in the training data, return majority class
                    return np.argmax(np.bincount(Y))
                else:
                    return self._predict_sample(sample, tree['children'][feature_value])
    # Instantiate the DecisionTree classifier
    dt = DecisionTree()

    # Fit the model on the training data
    dt.fit(Xtrain, Ytrain)

    # Make predictions on the validation data
    predictions = dt.predict(Xvalid)

    return predictions

# Calculate accuracy and confusion matrix
# def evaluate_performance(Ytrue, Ypred): 
import numpy as np
import pandas as pd
from decision_tree import *

# Function to evaluate performance
def evaluate_performance(Ytrue, Ypred): 
    # Calculate accuracy and confusion matrix
    # (Code for calculating accuracy and confusion matrix goes here)
    pass

# Load data using Pandas
# Step 1: Reading in the data
df = pd.read_csv('diabetes_prediction_dataset.csv')
df = pd.get_dummies(df, columns=["gender", "smoking_history"])

diabetic_data = df[df['diabetes'] == 1]  
non_diabetic_data = df[df['diabetes'] == 0]

# Take 1/10th of non-diabetic data
# Take 1/10th of non-diabetic data
non_diabetic_data = non_diabetic_data.sample(frac=0.1)

# Concatenate the datasets to create the final dataset
data = pd.concat([diabetic_data,non_diabetic_data])


# Step 2: Shuffling the observations
df = data.sample(frac=1, random_state=42).reset_index(drop=True)
# df = data

cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_Female', 'gender_Male',
       'gender_Other', 'smoking_history_No Info', 'smoking_history_current',
       'smoking_history_ever', 'smoking_history_former',
       'smoking_history_never', 'smoking_history_not current']



np.random.seed(0)
data = data.sample(frac=1)  # Shuffle the entire dataset

# Splitting data into training and validation using iloc
train_size = int(np.ceil(2/3 * len(data)))
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:]

X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]
X_valid = val_data.iloc[:, :-1]
Y_valid = val_data.iloc[:, -1].astype(int)

# Pre-process the data (convert continuous features to binary)
# Ignore nan values

means = X_train.mean()
X_train_binary = (X_train > means).astype(int)
X_valid_binary = (X_valid > means).astype(int)

# Train the decision tree model and make predictions
Y_valid_pred = myDT(X_train_binary.values, Y_train.astype(int).values, X_valid_binary.values)


print(Y_valid_pred)
# Evaluate performance
accuracy, confusion_matrix = evaluate_performance(Y_valid.values, Y_valid_pred)

precision = confusion_matrix[1][1]/ (confusion_matrix[1][1] + confusion_matrix[0][1])
recall = confusion_matrix[1][1]/ (confusion_matrix[1][1] + confusion_matrix[1][0])

print(f'Validation Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(confusion_matrix)
print('Precision:', precision)
print('Recall:', recall)
