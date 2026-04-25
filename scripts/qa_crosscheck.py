import ast
import importlib
import os
from datetime import datetime

import joblib
import pandas as pd
from PIL import Image


def main() -> int:
    results = {}

    print("=" * 50)
    print("FILE EXISTENCE CHECK")
    print("=" * 50)

    required_files = {
        "threat_detection.ipynb": ["threat_detection.ipynb", "notebooks/threat_detection.ipynb"],
        "app.py": ["app.py"],
        "best_model.pkl": ["best_model.pkl", "artifacts/best_model.pkl"],
        "scaler.pkl": ["scaler.pkl", "artifacts/scaler.pkl"],
        "feature_names.pkl": ["feature_names.pkl", "artifacts/feature_names.pkl"],
        "categorical_info.pkl": ["categorical_info.pkl", "artifacts/categorical_info.pkl"],
        "output/class_distribution.png": ["output/class_distribution.png"],
        "output/feature_importance.png": ["output/feature_importance.png"],
        "output/model_comparison.png": ["output/model_comparison.png"],
        "output/confusion_matrix_best.png": ["output/confusion_matrix_best.png"],
    }

    all_present = True
    resolved = {}

    for name, paths in required_files.items():
        found = None
        for path in paths:
            if os.path.exists(path):
                found = path
                break
        resolved[name] = found
        exists = found is not None
        all_present = all_present and exists
        status = "✅ FOUND" if exists else "❌ MISSING"
        suffix = f" (using {found})" if exists and found != name else ""
        print(f"{status}  ->  {name}{suffix}")

    print()
    print("✅ All required assets found." if all_present else "❌ Missing required assets.")

    print("\n" + "=" * 50)
    print("CHECK 1: PKL FILE INTEGRITY")
    print("=" * 50)

    model = None
    scaler = None
    features = None

    try:
        model = joblib.load(resolved["best_model.pkl"])
        print(f"✅ best_model.pkl loaded -> {type(model).__name__}")
        c1_model = True
    except Exception as e:
        print(f"❌ best_model.pkl failed -> {e}")
        c1_model = False

    try:
        scaler = joblib.load(resolved["scaler.pkl"])
        print(f"✅ scaler.pkl loaded -> {type(scaler).__name__}")
        c1_scaler = True
    except Exception as e:
        print(f"❌ scaler.pkl failed -> {e}")
        c1_scaler = False

    try:
        cat_info = joblib.load(resolved.get("categorical_info.pkl", "artifacts/categorical_info.pkl"))
    except:
        cat_info = {}

    try:
        features = joblib.load(resolved["feature_names.pkl"])
        print(f"✅ feature_names.pkl loaded -> {len(features)} features")
        c1_features = True
    except Exception as e:
        print(f"❌ feature_names.pkl failed -> {e}")
        c1_features = False

    results[1] = c1_model and c1_scaler and c1_features

    print("\n" + "=" * 50)
    print("CHECK 2: MODEL PREDICTION SHAPE")
    print("=" * 50)
    try:
        dummy_input = pd.DataFrame([{f: (cat_info[f][0] if f in cat_info else 0.0) for f in features}])[features]
        dummy_scaled = scaler.transform(dummy_input)

        print(f"✅ Scaler output shape correct -> {dummy_scaled.shape}")

        dummy_scaled_df = dummy_scaled
        pred = model.predict(dummy_scaled_df)
        assert int(pred[0]) in [0, 1]
        print(f"✅ Model prediction returned -> {int(pred[0])} (0=Normal, 1=Threat)")

        proba = model.predict_proba(dummy_scaled_df)
        assert proba.shape == (1, 2)
        assert abs(float(proba[0].sum()) - 1.0) < 1e-6
        print(
            f"✅ Probabilities correct -> Normal: {proba[0][0]:.2%} | Threat: {proba[0][1]:.2%}"
        )
        results[2] = True
    except Exception as e:
        print(f"❌ Prediction check failed -> {e}")
        results[2] = False

    print("\n" + "=" * 50)
    print("CHECK 3: NORMAL vs THREAT DIFFERENTIATION")
    print("=" * 50)

    normal_input = {f: (cat_info[f][0] if f in cat_info else 0.0) for f in features}
    normal_overrides = {
        "duration": 0,
        "src_bytes": 491,
        "dst_bytes": 0,
        "logged_in": 1,
        "count": 511,
        "srv_count": 511,
        "serror_rate": 0.0,
        "rerror_rate": 0.0,
        "same_srv_rate": 1.0,
    }
    if 'protocol_type' in cat_info: normal_overrides['protocol_type'] = 'tcp'
    if 'service' in cat_info: normal_overrides['service'] = 'http'
    if 'flag' in cat_info: normal_overrides['flag'] = 'SF'
    for key, value in normal_overrides.items():
        if key in normal_input:
            normal_input[key] = value

    threat_input = {f: (cat_info[f][0] if f in cat_info else 0.0) for f in features}
    threat_overrides = {
        "duration": 299,
        "src_bytes": 0,
        "dst_bytes": 0,
        "num_failed_logins": 5,
        "logged_in": 0,
        "num_compromised": 1,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 1.0,
        "rerror_rate": 0.5,
        "same_srv_rate": 0.06,
    }
    if 'protocol_type' in cat_info: threat_overrides['protocol_type'] = 'tcp'
    if 'service' in cat_info: threat_overrides['service'] = 'private'
    if 'flag' in cat_info: threat_overrides['flag'] = 'S0'
    for key, value in threat_overrides.items():
        if key in threat_input:
            threat_input[key] = value

    try:
        def run_prediction(input_dict):
            row = pd.DataFrame([input_dict])[features]
            scaled = scaler.transform(row)
            scaled_df = scaled
            pred = int(model.predict(scaled_df)[0])
            proba = model.predict_proba(scaled_df)[0]
            return pred, proba

        n_pred, n_proba = run_prediction(normal_input)
        t_pred, t_proba = run_prediction(threat_input)

        print(
            f"Normal input -> Prediction: {'🟢 NORMAL' if n_pred == 0 else '🔴 THREAT'} | Threat prob: {n_proba[1]:.2%}"
        )
        print(
            f"Threat input -> Prediction: {'🟢 NORMAL' if t_pred == 0 else '🔴 THREAT'} | Threat prob: {t_proba[1]:.2%}"
        )

        if n_pred == 0 and t_pred == 1:
            print("✅ Model correctly distinguishes Normal from Threat")
            results[3] = True
        else:
            print("❌ Model did not clearly separate normal/threat in this sanity case")
            results[3] = False
    except Exception as e:
        print(f"❌ Differentiation check failed -> {e}")
        results[3] = False

    print("\n" + "=" * 50)
    print("CHECK 4: OUTPUT IMAGE VALIDITY")
    print("=" * 50)

    image_files = [
        "output/class_distribution.png",
        "output/feature_importance.png",
        "output/model_comparison.png",
        "output/confusion_matrix_best.png",
    ]

    c4_all = True
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img.verify()
            print(f"✅ Valid image -> {img_path}")
        except FileNotFoundError:
            print(f"❌ MISSING -> {img_path}")
            c4_all = False
        except Exception as e:
            print(f"❌ Corrupt image -> {img_path} | {e}")
            c4_all = False
    results[4] = c4_all

    print("\n" + "=" * 50)
    print("CHECK 5: app.py SYNTAX CHECK")
    print("=" * 50)

    try:
        with open("app.py", "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print("✅ app.py has no syntax errors")
        results[5] = True
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py -> Line {e.lineno}: {e.msg}")
        results[5] = False
    except FileNotFoundError:
        print("❌ app.py not found")
        results[5] = False

    print("\n" + "=" * 50)
    print("CHECK 6: STREAMLIT APP DEPENDENCIES")
    print("=" * 50)

    required_packages = ["streamlit", "joblib", "pandas", "numpy", "matplotlib", "sklearn", "PIL"]
    c6_all = True
    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg} installed")
        except ImportError:
            print(f"❌ {pkg} NOT installed")
            c6_all = False
    results[6] = c6_all

    print("\n" + "=" * 50)
    print("CHECK 7: FEATURE COUNT CONSISTENCY")
    print("=" * 50)

    try:
        n_scaler_features = int(scaler.n_features_in_)
        n_saved_features = int(len(features))
        print(f"Scaler expects:         {n_scaler_features} features")
        print(f"feature_names.pkl has:  {n_saved_features} features")
        
        c7_scaler = n_scaler_features == n_saved_features
        print("✅ Feature counts match" if c7_scaler else "❌ Scaler/features mismatch")
        
        c7_model = True
        # Model operates on transformed features, so n_features_in_ won't match n_saved_features
        results[7] = c7_scaler
    except Exception as e:
        print(f"❌ Consistency check failed -> {e}")
        results[7] = False

    print("\n" + "=" * 50)
    print("CHECK 8: END-TO-END PIPELINE TEST")
    print("=" * 50)

    normal_case_input = {f: 0.0 for f in features}
    if 'protocol_type' in cat_info: normal_overrides['protocol_type'] = 'tcp'
    if 'service' in cat_info: normal_overrides['service'] = 'http'
    if 'flag' in cat_info: normal_overrides['flag'] = 'SF'
    for key, value in normal_overrides.items():
        if key in normal_case_input:
            normal_case_input[key] = value

    test_cases = [
        {"label": "Normal Traffic Sample", "input": normal_case_input, "expected": 0},
        {
            "label": "Threat Traffic Sample",
            "input": {**{f: 0.0 for f in features}, **{k: v for k, v in threat_overrides.items() if k in features}},
            "expected": 1,
        },
    ]

    c8_all = True
    for case in test_cases:
        try:
            row = pd.DataFrame([case["input"]])[features]
            scaled = scaler.transform(row)
            scaled_df = scaled
            pred = int(model.predict(scaled_df)[0])
            proba = model.predict_proba(scaled_df)[0]

            result = "🟢 NORMAL" if pred == 0 else "🔴 THREAT"
            match = "✅ PASS" if pred == case["expected"] else "❌ FAIL"
            if pred != case["expected"]:
                c8_all = False

            print(f"{match} | {case['label']} -> {result} | Confidence: {max(proba):.2%}")
        except Exception as e:
            print(f"❌ FAIL | {case['label']} -> {e}")
            c8_all = False

    print("\n✅ End-to-end pipeline test passed" if c8_all else "\n❌ End-to-end pipeline test failed")
    results[8] = c8_all

    print("\n" + "=" * 55)
    print("  ITM-390 | FULL QA CROSSCHECK - FINAL REPORT")
    print("=" * 55)
    print(f"  Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:         {type(model).__name__ if model is not None else 'N/A'}")
    print(f"  Features:      {len(features) if features is not None else 'N/A'}")
    print(f"  Scaler:        {type(scaler).__name__ if scaler is not None else 'N/A'}")

    print("\n  Checklist:")
    for i in range(1, 9):
        mark = "✅" if results.get(i, False) else "❌"
        print(f"  [{mark}] CHECK {i}")

    all_ok = all(results.get(i, False) for i in range(1, 9))
    print("\n  READY FOR SUBMISSION" if all_ok else "\n  ACTION REQUIRED")
    print("=" * 55)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
