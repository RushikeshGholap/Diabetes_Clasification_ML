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
def evaluate_performance(Ytrue, Ypred): 
    accuracy = np.mean(Ytrue == Ypred)
    
    num_classes = len(np.unique(Ytrue))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    num_d = 0
    num_nd = 0
    for true_class, pred_class in zip(Ytrue, Ypred):
        if true_class == 1:
            num_d +=1
        else:
            num_nd +=1
        confusion_matrix[true_class, pred_class] += 1

    print("True Positives:", confusion_matrix[1, 1])
    print("True Negatives:", confusion_matrix[0, 0])
    print("False positives:", confusion_matrix[0, 1])
    print("False Negatives:", confusion_matrix[1, 0])
    print("total diabetic samples:", num_d)
    print("total non diabetic samples:", num_nd)
    
    return accuracy, confusion_matrix