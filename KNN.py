from dataloader import load_normalized_data


class KNN:

    def __init__(self, k):
        self.k = k
        self.training_points = None
        self.training_labels = None

    def fit(self, training_points, training_labels):
        self.training_points = training_points
        self.training_labels = training_labels