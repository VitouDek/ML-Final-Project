import glob
import os
import warnings

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def download_dataset():
    dataset_id = "sampadab17/network-intrusion-detection"
    try:
        path = kagglehub.dataset_download(dataset_id)
        print(f"Downloaded: {dataset_id}")
        return dataset_id, path
    except Exception as exc:
        raise RuntimeError(f"Could not download dataset. Error: {exc}")


def find_first_csv(dataset_path):
    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {dataset_path}")
    print("Found CSV files:")
    for csv_path in csv_files:
        print(" -", csv_path)
    return csv_files[0]


def pick_labeled_csv(csv_files):
    candidate_names = {"label", "class", "attack", "attack_cat", "target"}

    # First pass: exact candidate column names.
    for csv_path in csv_files:
        try:
            head = pd.read_csv(csv_path, nrows=200)
        except Exception:
            continue
        head.columns = [str(col).strip().lower() for col in head.columns]
        if any(col in candidate_names for col in head.columns):
            return csv_path

    # Second pass: substring hints in column names.
    for csv_path in csv_files:
        try:
            head = pd.read_csv(csv_path, nrows=200)
        except Exception:
            continue
        head.columns = [str(col).strip().lower() for col in head.columns]
        for col in head.columns:
            if any(token in col for token in ["label", "class", "attack", "target"]):
                return csv_path

    # Third pass: choose a file whose last column appears categorical.
    for csv_path in csv_files:
        try:
            head = pd.read_csv(csv_path, nrows=1000)
        except Exception:
            continue
        head.columns = [str(col).strip().lower() for col in head.columns]
        last_col = head.columns[-1]
        unique_count = head[last_col].nunique(dropna=False)
        if str(head[last_col].dtype) == "object" or unique_count <= 20:
            return csv_path

    return csv_files[0]


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def detect_target_column(df):
    candidates = ["label", "class", "attack", "attack_cat", "target"]
    for name in candidates:
        if name in df.columns:
            return name

    # Fallback: select a categorical or low-cardinality column.
    fallback_candidates = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        dtype_name = str(df[col].dtype)
        if dtype_name == "object" or unique_count <= 20:
            fallback_candidates.append((col, unique_count))

    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[1])
        return fallback_candidates[0][0]

    return df.columns[-1]


def save_class_distribution(y, file_path):
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(["Normal (0)", "Attack (1)"], counts.values, color=["#2ca25f", "#de2d26"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()


def save_top_features(feature_names, feature_scores, file_path):
    top_idx = np.argsort(feature_scores)[-15:]
    top_features = np.array(feature_names)[top_idx]
    top_scores = feature_scores[top_idx]

    order = np.argsort(top_scores)
    top_features = top_features[order]
    top_scores = top_scores[order]

    plt.figure(figsize=(10, 7))
    plt.barh(top_features, top_scores, color="#3182bd")
    plt.title("Top 15 Selected Features")
    plt.xlabel("Feature Score / Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()

    return list(top_features)


def save_confusion_matrix(y_true, y_pred, model_name, file_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()
    return cm


def save_model_comparison(summary_df, file_path):
    plot_df = summary_df.copy()
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(plot_df["Model"]))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, plot_df[metric], width=width, label=metric)

    plt.title("Model Metrics Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(x + width * 1.5, plot_df["Model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()


def encode_target(y_series):
    y_str = y_series.astype(str).str.strip().str.lower()
    return y_str.apply(lambda value: 0 if value in {"normal", "benign", "0"} else 1)


def get_models(train_size):
    use_linear_svc = train_size > 100000
    if use_linear_svc:
        svm_model = LinearSVC(random_state=RANDOM_STATE)
        svm_label = "LinearSVC"
    else:
        svm_model = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
        svm_label = "SVM"

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        svm_label: svm_model,
    }
    return models


def tuning_search_space(model_name):
    spaces = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "Decision Tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"],
        },
        "LinearSVC": {
            "C": [0.01, 0.1, 1, 10],
            "loss": ["hinge", "squared_hinge"],
        },
    }
    return spaces.get(model_name, {})


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    rows = []
    reports = {}
    confusion_data = {}
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }
        rows.append(metrics_row)

        report = classification_report(y_test, y_pred, digits=4)
        reports[model_name] = report

        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        confusion_data[model_name] = save_confusion_matrix(y_test, y_pred, model_name, cm_path)
        fitted_models[model_name] = model

    summary_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return summary_df, reports, confusion_data, fitted_models


def predict_threat(input_dict, model, scaler, feature_names):
    row = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
    row_scaled = scaler.transform(row)

    prediction = int(model.predict(row_scaled)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row_scaled)[0][1])
    else:
        # Approximate confidence for models without predict_proba.
        decision = model.decision_function(row_scaled)
        probability = float(1 / (1 + np.exp(-decision[0])))

    label = "THREAT DETECTED" if prediction == 1 else "NORMAL"
    return f"{label} (confidence: {probability:.2%})"


def main():
    print("=== STEP 1/2: Download and load data ===")
    dataset_name, dataset_path = download_dataset()
    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {dataset_path}")
    print("Found CSV files:")
    for csv_path in csv_files:
        print(" -", csv_path)

    csv_path = pick_labeled_csv(csv_files)
    print("Using CSV file:", csv_path)

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    target_col = detect_target_column(df)
    print("\nDetected target column:", target_col)

    print("\nRaw class counts:\n", df[target_col].value_counts(dropna=False))

    # Constraint handling for inf values.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Handle missing values by dropping rows with nulls for simplicity and robustness.
    df = df.dropna(axis=0).reset_index(drop=True)

    # Drop constant columns except target.
    nunique = df.nunique(dropna=False)
    constant_cols = [col for col in df.columns if col != target_col and nunique[col] <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print("Dropped constant columns:", constant_cols)

    y = encode_target(df[target_col])
    if y.nunique() < 2:
        raise ValueError(
            f"Target column '{target_col}' produced a single class after encoding. "
            "Please verify the selected CSV contains labels."
        )

    X = df.drop(columns=[target_col])

    print("\nEncoded class counts:\n", y.value_counts())

    class_dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    save_class_distribution(y, class_dist_path)

    # Handle categorical columns
    cat_cols = ["protocol_type", "service", "flag"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    categorical_info = {col: X[col].unique().tolist() for col in cat_cols}
    joblib.dump(categorical_info, os.path.join(ARTIFACTS_DIR, "categorical_info.pkl"))

    # Ensure no data leakage: split FIRST
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # To do feature selection on original features, we temporarily encode ONLY the training set
    X_train_oe = X_train_raw.copy()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_oe[cat_cols] = oe.fit_transform(X_train_oe[cat_cols])

    print("\n=== STEP 4: Feature selection ===")
    selector_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    selector_model.fit(X_train_oe, y_train)

    k = min(15, X_train_oe.shape[1])
    importances = selector_model.feature_importances_
    top_idx = np.argsort(importances)[-k:]

    selected_features = X_train_oe.columns[top_idx].tolist()
    selected_features = [str(c) for c in selected_features]  # Convert from numpy strings
    feature_scores = importances[top_idx]

    selected_features = save_top_features(
        selected_features,
        feature_scores,
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )
    selected_features = [str(c) for c in selected_features]

    # Build ColumnTransformer based on selected features
    sel_cat_cols = [str(c) for c in selected_features if c in cat_cols]
    sel_num_cols = [str(c) for c in selected_features if c not in cat_cols]

    # Important: output dense arrays to work smoothly with SMOTE/RandomForest
    deploy_scaler = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), sel_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), sel_cat_cols),
        ]
    )

    X_train_selected = X_train_raw[selected_features]
    X_test_selected = X_test_raw[selected_features]

    X_train_scaled_array = deploy_scaler.fit_transform(X_train_selected)
    X_test_scaled_array = deploy_scaler.transform(X_test_selected)

    X_train_scaled = pd.DataFrame(X_train_scaled_array, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, index=X_test_raw.index)

    # Class imbalance check with SMOTE if minority < 20%.
    class_ratio = y_train.value_counts(normalize=True)
    minority_ratio = class_ratio.min()

    if minority_ratio < 0.20:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE applied. Minority ratio before: {minority_ratio:.2%}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        print(f"SMOTE not required. Minority ratio: {minority_ratio:.2%}")
        
    # We set X_train_selected to X_train_balanced for model training to expect numerical arrays
    X_train_selected = X_train_balanced
    X_test_selected = X_test_scaled

    print("Selected features (top 15):")
    print(selected_features)

    print("\n=== STEP 5/6: Model training and evaluation ===")
    models = get_models(len(X_train_selected))

    summary_df, reports, confusion_data, fitted_models = train_and_evaluate(
        models,
        X_train_selected,
        X_test_selected,
        y_train_balanced,
        y_test,
    )

    for model_name, report in reports.items():
        print(f"\nClassification Report - {model_name}\n{report}")

    save_model_comparison(summary_df, os.path.join(OUTPUT_DIR, "model_comparison.png"))

    best_row = summary_df.iloc[0]
    best_model_name = best_row["Model"]
    best_base_model = fitted_models[best_model_name]
    print("Best base model by F1:", best_model_name)

    # Save dedicated best confusion matrix.
    best_predictions = best_base_model.predict(X_test_selected)
    best_cm = save_confusion_matrix(
        y_test,
        best_predictions,
        f"Best - {best_model_name}",
        os.path.join(OUTPUT_DIR, "confusion_matrix_best.png"),
    )

    print("\n=== STEP 7: Hyperparameter tuning (best model) ===")
    param_dist = tuning_search_space(best_model_name)
    tuned_model = best_base_model
    best_params = {}

    if param_dist:
        search = RandomizedSearchCV(
            estimator=best_base_model,
            param_distributions=param_dist,
            n_iter=min(10, max(1, np.prod([len(v) for v in param_dist.values()]))),
            cv=3,
            scoring="f1",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train_selected, y_train_balanced)
        tuned_model = search.best_estimator_
        best_params = search.best_params_

    tuned_model.fit(X_train_selected, y_train_balanced)
    y_tuned_pred = tuned_model.predict(X_test_selected)

    tuned_acc = accuracy_score(y_test, y_tuned_pred)
    tuned_prec = precision_score(y_test, y_tuned_pred, zero_division=0)
    tuned_rec = recall_score(y_test, y_tuned_pred, zero_division=0)
    tuned_f1 = f1_score(y_test, y_tuned_pred, zero_division=0)

    print("Best params:", best_params)
    print("\nTuned model report:\n", classification_report(y_test, y_tuned_pred, digits=4))

    # Save model artifacts.
    joblib.dump(tuned_model, os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    joblib.dump(deploy_scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(selected_features, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
    joblib.dump({
        "Accuracy": f"{tuned_acc:.2%}",
        "Precision": f"{tuned_prec:.2%}",
        "Recall": f"{tuned_rec:.2%}",
        "F1-Score": f"{tuned_f1:.2%}",
    }, os.path.join(ARTIFACTS_DIR, "metrics.pkl"))

    # CLI prediction demo using one raw encoded test row.
    sample = X_test_raw.iloc[0][selected_features].to_dict()
    prediction_message = predict_threat(
        sample,
        tuned_model,
        deploy_scaler,
        selected_features,
    )
    print("\nDeployment demo:", prediction_message)

    # Final summary report.
    tn, fp, fn, tp = confusion_matrix(y_test, y_tuned_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    class_balance = y.value_counts(normalize=True).sort_index()
    normal_pct = class_balance.get(0, 0.0) * 100
    attack_pct = class_balance.get(1, 0.0) * 100

    print("\n" + "=" * 45)
    print("  ITM-390 | AI-Based Threat Detection Tool")
    print("  American University of Phnom Penh")
    print("=" * 45)
    print(f"Dataset:         {dataset_name}")
    print(f"Total Samples:   {len(df)}")
    print(f"Features Used:   {selected_features}")
    print(f"Class Balance:   Normal={normal_pct:.2f}% | Attack={attack_pct:.2f}%")

    print("\n--- Model Comparison (ranked by F1) ---")
    print(f"{'Model':24} {'Acc':6} {'Prec':6} {'Rec':6} {'F1':6}")
    print("-" * 55)
    for _, row in summary_df.iterrows():
        marker = " <- BEST" if row["Model"] == best_model_name else ""
        print(
            f"{row['Model'][:24]:24} {row['Accuracy']:.3f}  {row['Precision']:.3f}  {row['Recall']:.3f}  {row['F1']:.3f}{marker}"
        )

    print(f"\nBest Model:      {best_model_name}")
    print(f"Best F1-Score:   {tuned_f1:.3f}")
    print(f"False Positive Rate: {false_positive_rate:.3%}")
    print("Deployment:      predict_threat() function ready")
    print("=" * 45)

    print("\nSaved artifacts:")
    print(" -", os.path.join(OUTPUT_DIR, "class_distribution.png"))
    print(" -", os.path.join(OUTPUT_DIR, "feature_importance.png"))
    print(" -", os.path.join(OUTPUT_DIR, "model_comparison.png"))
    print(" -", os.path.join(OUTPUT_DIR, "confusion_matrix_best.png"))
    print(" -", os.path.join("artifacts", "best_model.pkl"))
    print(" -", os.path.join("artifacts", "scaler.pkl"))
    print(" -", os.path.join("artifacts", "feature_names.pkl"))
    print(" -", os.path.join("artifacts", "categorical_info.pkl"))


if __name__ == "__main__":
    main()
