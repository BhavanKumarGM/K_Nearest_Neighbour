from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- LOAD DATASET ----------------
X, y = load_iris(return_X_y=True)
target_names = load_iris().target_names  # for readable prediction

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- PIPELINE ----------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# ---------------- HYPERPARAMETER TUNING ----------------
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- NEW DATA PREDICTION ----------------
# Example: unseen flower measurements
new_sample = [[5.8, 2.7, 5.1, 1.9]]  # sepal & petal measurements

prediction = best_model.predict(new_sample)
prediction_label = target_names[prediction[0]]

print("\nNew Sample Prediction:", prediction_label)