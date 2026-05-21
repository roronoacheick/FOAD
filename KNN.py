from dataloader import load_normalized_data


class KNN:

    def __init__(self, k):
        self.k = k
        self.training_points = None
        self.training_labels = None

    def fit(self, training_points, training_labels):
        self.training_points = training_points
        self.training_labels = training_labels

    def _euclidean_distance(self, point_a, point_b):
        sum_of_squares = 0
        for feature_index in range(len(point_a)):
            difference = point_a[feature_index] - point_b[feature_index]
            sum_of_squares += difference ** 2
        return sum_of_squares ** 0.5
    
    def _get_neighbors(self, test_point):
        distances = []
        for index in range(len(self.training_points)):
            distance = self._euclidean_distance(test_point, self.training_points[index])
            distances.append((distance, self.training_labels[index]))
        distances.sort(key=lambda pair: pair[0])
        neighbors = distances[:self.k]
        return neighbors