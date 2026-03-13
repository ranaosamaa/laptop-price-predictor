import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score


df = pd.read_csv("data/laptop_data.csv")

X=df.drop(columns=['Price (Euro)'])
y=np.log(df['Price (Euro)'])

target_col = 'Price (Euro)'
categorical_features = ['Company','TypeName','GPU_Company','CPU','OS']
numeric_features = [c for c in df.columns
                    if c not in categorical_features + [target_col, 'id']]

degrees = [2,3,4,5]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_degree = None
best_score = -np.inf
best_poly_model = None

for d in degrees:
    preprocess = ColumnTransformer(
    transformers=[
        ("num_poly", Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=d, include_bias=False, interaction_only=True)),]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),])

    model = Pipeline([
    ("pre", preprocess),
    ("reg", LinearRegression())
    ])
    
    scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    mean_score = scores.mean()
    print(f"Degree {d}")
    print("R2 per fold:", scores)
    print("Mean R2:", scores.mean())
    print("======================")
    if mean_score > best_score:
        best_score = mean_score        
        best_degree = d
        best_poly_model = model

print("Best degree:", best_degree)
print("Best CV R2:", best_score)

best_poly_model.fit(X, y)

joblib.dump(best_poly_model, "pipeline.pkl")