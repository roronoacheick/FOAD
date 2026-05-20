from dataloader import load_normalized_data


class KNN:

    def __init__(self, k):
        self.k = k
        self.training_points = None
        self.training_labels = None