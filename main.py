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

# Test 
model.fit(training_points, training_labels)
print("training_points après fit (5 premiers) :", model.training_points[:5])
print("training_labels après fit (5 premiers) :", model.training_labels[:5])


point_a = training_points[0]
point_b = training_points[1]
distance = model._euclidean_distance(point_a, point_b)
print("Distance entre le point 0 et le point 1 :", distance)