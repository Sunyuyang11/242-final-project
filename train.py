import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("cleaned_shelter_data.csv")

y = df["is_adopted"]
cols_to_drop = ["is_adopted", "Outcome Type", "Outcome Date"]
X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

X = pd.get_dummies(X_raw, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print("AUC:", auc)

joblib.dump(model, "gb_model.pkl")
joblib.dump(list(X.columns), "gb_columns.pkl")
print("Saved: gb_model.pkl, gb_columns.pkl")
