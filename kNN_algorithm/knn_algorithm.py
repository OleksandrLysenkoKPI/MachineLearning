import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy import stats

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

class kNN:
    def __init__(self, k=3):
        self.k = k

    # Calculate Euclidean distance
    def euclidean_distance(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    # Store train set
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Get nearest neighbours
    def get_neighbours(self, test_row):
        distances = []

        # Calculate distance to all points in X_train
        for train_row, train_class in zip(self.X_train, self.y_train): # train_class is a class mark (index)
            distance = self.euclidean_distance(test_row, train_row)
            distances.append((distance, train_class))

        # Sort by first element in tuple (distance)
        distances.sort(key=lambda x: x[0])

        # Identify k classes of nearest neighbours
        neighbours = [distances[i][1] for i in range(self.k)]

        return neighbours

    def predict(self, X_test):
        predictions = []
        for test_row in X_test:
            nearest_neighbours = self.get_neighbours(test_row)
            mode = stats.mode(nearest_neighbours, keepdims=True).mode[0]
            predictions.append(mode)

        return np.array(predictions)

    # Function to calculate accuracy
    def accuracy(self, predictions, y_test):
        return 100 * np.mean(predictions == y_test)


if __name__ == "__main__":
    knn = kNN(k=3) # accuracy starts to drop beginning from k=44
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(f'Accuracy: {knn.accuracy(predictions, y_test):.3f} %')

    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction_new = knn.predict(X_new)
    print("Спрогнозована мітка: {}".format(iris['target_names'][prediction_new]))
