import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
import shap

# === Load and preprocess data ===
df = pd.read_csv("thesis/data/processed/final_combined_dataset.csv")
df = df.dropna(subset=["credibility_score"])

non_feature_cols = ["url", "credibility_label", "credibility_score", "domain_type"]
X_raw = pd.get_dummies(df.drop(columns=non_feature_cols), drop_first=True)
y = df["credibility_score"]

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# === Split into train, validation, and test sets ===
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42
)  # 60% train, 20% val, 20% test

print("Any NaNs in X_train?", X_train.isnull().sum().sum())
X_train = X_train.fillna(0)

# === Apply SMOTE to training set ===
smote = SMOTE(sampling_strategy=0.9, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === Compute class weight for XGBoost (after SMOTE)
weight_ratio = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()

print("Train label distribution (after SMOTE):\n", y_train_resampled.value_counts())

# === Define Models ===
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf.fit(X_train_resampled, y_train_resampled)

# Determine appropriate eval_metric based on label uniqueness
if y_train.nunique() == 2:
    eval_metric = 'logloss'
else:
    eval_metric = 'mlogloss'

xgb_base = XGBClassifier(
    eval_metric=eval_metric,
    scale_pos_weight=weight_ratio,
    random_state=42
)
xgb_base.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_val, y_val)],
    #callbacks=[EarlyStopping(rounds=10)],
    verbose=False
)

# === Hyperparameter tuning for XGBoost ===
param_grid = {
    'max_depth': [6, 7, 8],
    'min_child_weight': [1, 2],
    'gamma': [0, 0.05, 0.1],
    'learning_rate': [0.05, 0.075, 0.1],
    'n_estimators': [250, 300, 350],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.9, 1.0],
    'reg_alpha': [0.05, 0.1],
    'reg_lambda': [0.5, 1, 2]
}

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=50,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_resampled, y_train_resampled)
xgb_best = random_search.best_estimator_
print(f"Best parameters: {random_search.best_params_}")
print(f"Best F1 Score: {random_search.best_score_}")

# Fit the best model on the full training set
xgb_best.fit(X_train_resampled, y_train_resampled)

# === Baseline (majority class) ===
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_resampled, y_train_resampled)

# === Model Evaluation ===

os.makedirs("balanced_results_changed_features", exist_ok=True)

models = [
    ("Baseline", baseline),
    ("Random Forest", rf),
    ("XGBoost (Base)", xgb_base),
    ("XGBoost (Tuned)", xgb_best)
]

results = []
thresholds = [0.3, 0.4, 0.5, 0.6]

for name, model in models:
    if hasattr(model, "predict_proba"):
        for thresh in thresholds if "XGBoost" in name else [0.5]:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= thresh).astype(int)

            # ROC for class 1 as positive (low credibility)
            fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
            roc_auc = auc(fpr, tpr)

            results.append({
                "Model": f"{name} @ {thresh}" if "XGBoost" in name else name,
                "Accuracy": (y_pred == y_test).mean(),
                "Precision": precision_score(y_test, y_pred, pos_label=1),
                "Recall": recall_score(y_test, y_pred, pos_label=1),
                "F1 Score": f1_score(y_test, y_pred, pos_label=1),
                "ROC-AUC": roc_auc
            })
    else:
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": model.score(X_test, y_test),
            "Precision": precision_score(y_test, y_pred, pos_label=1),
            "Recall": recall_score(y_test, y_pred, pos_label=1),
            "F1 Score": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": None,
        })


    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["Low", "High"]))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename_safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("@", "at")
    plt.tight_layout()
    plt.savefig(f"balanced_results_changed_features/confusion_{filename_safe}.png")
    plt.show()


# === Plot ROC Curves for All Probabilistic Models ===

plt.figure(figsize=(8, 6))

for name, model in models:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

# Add diagonal for random classifier
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

plt.xlabel("False Positive Rate (Falsely predicted low cred)")
plt.ylabel("True Positive Rate (Correctly predicted low cred)")
plt.title("ROC Curves for All Models (Low Credibility Focus)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("balanced_results_changed_features/roc_curve_all_models.png")
plt.show()


# === SHAP Analysis ===
explainer = shap.Explainer(xgb_best, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("balanced_results_changed_features/shap_summary_plot.png")
plt.show()

# Save top SHAP features
shap_sum = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({'feature': X_test.columns, 'shap_importance': shap_sum})
importance_df.sort_values(by='shap_importance', ascending=False).to_csv("balanced_results_changed_features/shap_feature_importance.csv", index=False)

# === Save and visualize ===
results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)

results_df.to_csv("balanced_results_changed_features/model_comparison.csv", index=False)

print("\n📊 Model Comparison Summary:")
print(tabulate(results_df, headers='keys', tablefmt='github', showindex=False))

# === Plot 1: F1 Score Comparison with highlight ===
best_f1 = results_df["F1 Score"].max()
best_model = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]

plt.figure(figsize=(9, 5))
sns.barplot(x="F1 Score", y="Model", data=results_df, palette="viridis")
plt.axvline(best_f1, color="red", linestyle="--", label=f"Best: {best_model}")
plt.title("🔍 F1 Score Comparison by Model")
plt.xlabel("F1 Score")
plt.ylabel("Model")
plt.legend()
plt.tight_layout()
plt.savefig("balanced_results_changed_features/f1_comparison_highlighted.png")
plt.show()

# === Plot 2: All Metrics Side-by-Side ===
melted = results_df.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"])
plt.figure(figsize=(10, 6))
sns.barplot(data=melted, x="value", y="Model", hue="variable", palette="Set2")
plt.title("📊 Metric Comparison by Model")
plt.xlabel("Score")
plt.ylabel("Model")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("balanced_results_changed_features/metric_comparison_all.png")
plt.show()

# === Identify Correctly Predicted Websites (XGBoost Best @ 0.5 Threshold) ===

# Get predicted probabilities and convert to binary predictions
y_proba_best = xgb_best.predict_proba(X_test)[:, 1]
y_pred_best = (y_proba_best >= 0.5).astype(int)

# Rebuild test subset with predictions
df_test = df.iloc[X_test.index].copy()
df_test["predicted"] = y_pred_best
df_test["actual"] = y_test.values

# === Identify Correctly Predicted Low/Medium Credibility Websites (Label 1) ===

# Filter: correct predictions where actual is 1 (low/medium)
correct_low_medium = df_test[
    (df_test["predicted"] == df_test["actual"]) &
    (df_test["actual"] == 1)
    ]

incorrect_low_medium = df_test[
    (df_test["predicted"] != df_test["actual"]) &
    (df_test["actual"] == 1)
    ]

# Save
incorrect_low_medium.to_csv("balanced_results_changed_features/incorrect_low_medium_predictions.csv", index=False)

print(f"\n✅ Saved {len(correct_low_medium)} correctly predicted LOW/MEDIUM credibility websites to 'balanced_results_changed_features/correct_low_medium_predictions.csv'")
print(f"\n✅ Saved {len(incorrect_low_medium)} incorrectly predicted LOW/MEDIUM credibility websites to 'balanced_results_changed_features/incorrect_low_medium_predictions.csv'")
