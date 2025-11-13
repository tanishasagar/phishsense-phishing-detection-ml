import joblib, traceback, sys
p = r"C:\Users\HP\phish-sense\models\phishsense_pipeline.joblib"
print("Trying to load:", p)
try:
    m = joblib.load(p)
    print("Loaded model type:", type(m))
    if hasattr(m, "named_steps"):
        print("Pipeline steps:", list(m.named_steps.keys()))
except Exception:
    traceback.print_exc()
    sys.exit(1)
