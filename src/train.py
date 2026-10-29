# src/train.py
import json, os, joblib, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score

from src.config import ARTIFACTS_DIR, MODEL_PATH, METRICS_PATH, RANDOM_SEED, TEST_SIZE, CV_FOLDS
from src.data_loader import load_data
from src.preprocessor import get_preprocessor

def train_and_export():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_data()
    X = df.drop(columns=['y'])
    y = df['y']

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Detected {len(num_cols)} Numerical and {len(cat_cols)} Categorical features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = get_preprocessor(cat_cols, num_cols)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=120, class_weight='balanced', random_state=RANDOM_SEED)
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_auc, best_pipe, best_name = 0.0, None, ""
    leaderboard = []

    print("\nStarting Cross Validation...")
    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['roc_auc', 'f1'])
        mean_auc = float(scores['test_roc_auc'].mean())
        mean_f1 = float(scores['test_f1'].mean())
        leaderboard.append({"model": name, "cv_roc_auc": round(mean_auc, 4), "cv_f1": round(mean_f1, 4)})
        print(f"-> {name} | CV ROC-AUC: {mean_auc:.4f} | CV F1: {mean_f1:.4f}")
        if mean_auc > best_auc:
            best_auc, best_pipe, best_name = mean_auc, pipe, name

    # Train best model
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "best_model": best_name,
        "test_roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "test_pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "leaderboard": leaderboard,
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols
        }
    }

    joblib.dump(best_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*50)
    print(f"Training Complete! Champion Model: {best_name}")
    print(f"Holdout ROC-AUC: {metrics['test_roc_auc']}")
    print(f"Holdout PR-AUC: {metrics['test_pr_auc']}")
    print(f"Model saved to: {MODEL_PATH}")
    print("="*50)

if __name__ == "__main__":
    train_and_export()