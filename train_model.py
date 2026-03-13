import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import loguniform

df = pd.read_csv("data/laptop_data.csv")

X = df.drop("Price (Euro)", axis=1)
y = df["Price (Euro)"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=20000))
])

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions={"lasso__alpha": loguniform(1e-5, 1e1)},
    n_iter=20,
    cv=kfold,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

rs.fit(X, y)

print("Best alpha:", rs.best_params_["lasso__alpha"])
print("Best CV R2:", rs.best_score_)
best_model = rs.best_estimator_

joblib.dump(best_model, "pipeline.pkl")