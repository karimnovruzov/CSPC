import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/concrete.csv")


print(f"The dataset contains {len(df)} rows.")
print(f"Missing value count: {df.isna().sum().sum()}")
print(f"Duplicate row count: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)
print(f"Row count after cleaning: {len(df)}")


X = df.drop(columns=["strength"])
y = df["strength"]


print("\n DATA DESCRIPTION ")
print(df.describe())


import os

os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))



axes[0].scatter(X["cement"], y, alpha=0.5, color="b")
axes[0].set_title("Strength vs Cement")
axes[0].set_xlabel("Cement (kg/m^3)")
axes[0].set_ylabel("Strength (MPa)")



axes[1].scatter(X["age"], y, alpha=0.5, color="g")
axes[1].set_title("Strength vs Age")
axes[1].set_xlabel("Age (days)")
axes[1].set_ylabel("Strength (MPa)")

plt.tight_layout()
plt.savefig("results/strength_vs_inputs.png")
print(
    "\nGraph saved successfully."
)


from sklearn.model_selection import train_test_split



X = df.drop(columns=["strength"])
y = df["strength"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Data Split Results ")
print(f"Training set (X_train) size: {X_train.shape}")
print(f"Test set (X_test) size: {X_test.shape}")
print(f"Training target (y_train) size: {y_train.shape}")
print(f"Test target (y_test) size: {y_test.shape}")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



df = pd.read_csv("data/concrete.csv")
df = df.drop_duplicates().reset_index(drop=True)



X = df.drop(columns=["strength"])
y = df["strength"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = LinearRegression()
model.fit(X_train, y_train)



y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)



train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

n = X_train.shape[0]
p = X_train.shape[1]
train_adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)



print("--- Model Metrics ---")
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)
print("Train R^2:", train_r2)
print("Train Adjusted R^2:", train_adj_r2)
print("Test R^2:", test_r2)



residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.scatter(X_test["age"], residuals, color="purple", alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xscale("log")
plt.title("Test Residuals vs Age")
plt.xlabel("Age (days)")
plt.ylabel("Residuals (True - Predicted)")
plt.savefig("results/linear_residuals_vs_age.png")
plt.close()

print("\nGraph saved successfully!")




from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


print("\n Polynomial Regression Metrics ")
print(f"{'Degree':<8} | {'Features (p)':<14} | {'Train RMSE':<12} | {'Test RMSE':<12} | {'Test R^2'}")
print("-" * 65)

for degree in [1, 2, 3, 4]:


 
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    
    poly_model.fit(X_train, y_train)


    
    y_tr_pred = poly_model.predict(X_train)
    y_te_pred = poly_model.predict(X_test)

    
    tr_rmse = np.sqrt(mean_squared_error(y_train, y_tr_pred))
    te_rmse = np.sqrt(mean_squared_error(y_test, y_te_pred))
    te_r2 = r2_score(y_test, y_te_pred)


    
    p_features = poly_model.named_steps['polynomialfeatures'].n_output_features_
    
    print(f"{degree:<8} | {p_features:<14} | {tr_rmse:<12.4f} | {te_rmse:<12.4f} | {te_r2:.4f}")









    from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


y_class = (df["strength"] >= 40).astype(int)

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print("\n CLASSIFICATION METRICS ")
k_list = [1, 3, 5, 7, 15, 25, 51, 101]

for k in k_list:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(Xc_train, c_train)

    train_acc = knn.score(Xc_train, c_train)
    test_acc = knn.score(Xc_test, c_test)

    print("K =", k, " | Train Accuracy:", train_acc, " | Test Accuracy:", test_acc)


print("\n Decision Tree Classification Metrics ")
depth_list = [1, 2, 3, 4, 5, 6, 8, None]

for depth in depth_list:
    tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=depth, random_state=0
    )
    tree.fit(Xc_train, c_train)

    train_acc = tree.score(Xc_train, c_train)
    test_acc = tree.score(Xc_test, c_test)
    leaves = tree.get_n_leaves()

    print(
        "Depth =",
        depth,
        " | Leaves =",
        leaves,
        " | Train Acc:",
        train_acc,
        " | Test Acc:",
        test_acc,
    )





    from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(Xc_train, c_train)
knn_pred = knn.predict(Xc_test)



tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0)
tree.fit(Xc_train, c_train)
tree_pred = tree.predict(Xc_test)

print("\nResults for KNN and Decision Tree Classifiers:")
tn, fp, fn, tp = confusion_matrix(c_test, knn_pred).ravel()
print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Negatives (TN):", tn)
print("Accuracy:", accuracy_score(c_test, knn_pred))
print("Precision:", precision_score(c_test, knn_pred))
print("Recall:", recall_score(c_test, knn_pred))
print("F1 Score:", f1_score(c_test, knn_pred))


print("\n Decision Tree Results ")
tn, fp, fn, tp = confusion_matrix(c_test, tree_pred).ravel()
print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Negatives (TN):", tn)
print("Accuracy:", accuracy_score(c_test, tree_pred))
print("Precision:", precision_score(c_test, tree_pred))
print("Recall:", recall_score(c_test, tree_pred))
print("F1 Score:", f1_score(c_test, tree_pred))




lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

print("\n LINEAR REGRESSION COEFFICIENTS ")
for name, coef in zip(X.columns, lin_reg_model.coef_):
    print(name, ":", coef)

tree_importance_model = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, random_state=0
)
tree_importance_model.fit(Xc_train, c_train)

print("\n--- DECISION TREE FEATURE IMPORTANCES ---")
for name, imp in zip(X.columns, tree_importance_model.feature_importances_):
    print(name, ":", imp)