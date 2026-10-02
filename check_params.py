from joblib import load
model = load("models/best_model.joblib")
print(type(model).__name__)
for k, v in sorted(model.get_params().items()):
    print(f"{k}: {v}")