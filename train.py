import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

from cardio_library.preprocessing.transformer import CardioTransformer
from cardio_library.models.logistic_model import LogisticModel
from cardio_library.models.decision_tree_model import DecisionTreeModel
from cardio_library.models.random_forest_model import RandomForestModel
from cardio_library.utils.save_load import save_model, save_transformer
from cardio_library.metrics.evaluation import evaluate_model

from cardio_library.model_selection.grid_search import perform_grid_search



# 🔹 1. Деректерді оқу
data = pd.read_csv("cardio_train.csv", sep=";")

# 🔹 2. Трансформер арқылы X, y бөліп алу
transformer = CardioTransformer()
transformer.fit(data)
X, y = transformer.transform(data)
column_names = X.columns.tolist()
# 🔹 3. Масштабтау
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 4. Train/Test бөлу
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔹 5. Модельдерді дайындау
models = {
    "Logistic Regression": LogisticModel(learning_rate=0.01, n_iter=1000, reg_lambda=0.0, class_weight='balanced'),
    "Decision Tree": DecisionTreeModel(max_depth=5, min_samples_split=10),
    "Random Forest": RandomForestModel(n_estimators=100, max_depth=10, min_samples_split=5)
}

# 🔹 6. Әр модель бойынша бағалау нәтижелерін жинау
results = []
roc_curves = {}

for name, model in models.items():
    print(f"\n📌 {name} моделін оқыту басталды...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Метрикаларды бағалау
    metrics = evaluate_model(y_test, y_pred, y_proba)
    metrics["Model"] = name
    results.append(metrics)

    # ROC қисығы
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, metrics["roc_auc"])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{name} — Қате матрицасы")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# 🔹 7. Нәтижелерді кестеге жинау
results_df = pd.DataFrame(results)
results_df = results_df[["Model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]

print("\n📊 Модельдердің салыстырмалы нәтижелері:")
print(results_df)

# 🔹 8. ROC қисықтарын бір графикке салу
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Кездейсоқ')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC қисықтарының салыстырмасы')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔹 9. Соңғы модельді сақтау
last_model_name = "Random Forest"
save_model(models[last_model_name], f"cardio_library/models/{last_model_name.replace(' ', '_').lower()}.pkl")
save_transformer(transformer, "cardio_library/preprocessing/transformer.pkl")
print(f"\n✅ {last_model_name} моделі мен трансформер сәтті сақталды.")

# 🔹 10. cardio бағанындағы класс теңгерімі
print("\n📦 cardio мәндерінің жиілігі:")
print(data['cardio'].value_counts())

import pickle

model_preds = {name: model.predict(X_test) for name, model in models.items()}
model_probas = {name: model.predict_proba(X_test) for name, model in models.items()}

with open("model_preds.pkl", "wb") as f:
    pickle.dump(model_preds, f)

with open("model_probas.pkl", "wb") as f:
    pickle.dump(model_probas, f)

with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Модель
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Параметр торы
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

# Grid SearchCV
grid_search = GridSearchCV(log_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("📌 Ең үздік параметрлер:", best_params)
print("🔍 Ең үздік кросс-валидация нәтижесі:", best_score)

# Сақтау
save_model(best_model, "cardio_library/models/sklearn_logistic_model.pkl")


