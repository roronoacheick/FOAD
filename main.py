from dataloader import load_normalized_data
from KNN import KNN

training_points, training_labels, scaler = load_normalized_data("bienetre.xlsx")


training_points = training_points.tolist()
training_labels = training_labels.tolist()

# Instanciation du modèle
model = KNN(k=3)
print("Modèle créé avec k =", model.k)
print("training_points avant fit :", model.training_points)
print("training_labels avant fit :", model.training_labels)

# Test du fit
model.fit(training_points, training_labels)
print("training_points après fit (5 premiers) :", model.training_points[:5])
print("training_labels après fit (5 premiers) :", model.training_labels[:5])