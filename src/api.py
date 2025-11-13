# api.py (PhishSense) — consolidated, robust version
"""
PhishSense API (explainable).
- Expects models/phishsense_pipeline.joblib (trained pipeline).
- Optional: models/feature_names.json (if present used for SHAP background alignment).
- Optional: data/processed/all_features.csv (used for SHAP background sample).
- Optional: whitelist.txt / blacklist.txt in project root (one entry per line).
"""

from flask import Flask, request, jsonify
import joblib, traceback, pandas as pd, numpy as np, os, json, math, logging
from pathlib import Path
from urllib.parse import urlparse
from feature_extractor import extract_features

# --- helper transformer class for unpickling (if pipeline used it) ---
from sklearn.base import BaseEstimator, TransformerMixin
import re as _re

class URLNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = []
        for u in X:
            s = str(u)
            length = len(s)
            digits = sum(ch.isdigit() for ch in s)
            dots = s.count('.')
            hyphens = s.count('-')
            slashes = s.count('/')
            at_sign = 1 if '@' in s else 0
            has_ip = 1 if _re.search(r'://\d+\.\d+\.\d+\.\d+', s) else 0
            out.append([length, digits, dots, hyphens, slashes, at_sign, has_ip])
        return np.array(out)

# --- app & paths ---
app = Flask(__name__)
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishsense_pipeline.joblib"
FEATURES_PATH = ROOT / "models" / "feature_names.json"
BACKGROUND_CSV = ROOT / "data" / "processed" / "all_features.csv"
WHITELIST_PATH = ROOT / "whitelist.txt"
BLACKLIST_PATH = ROOT / "blacklist.txt"
LOG_PATH = ROOT / "logs"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# defaults
PHISH_THRESHOLD_DEFAULT = 0.60

# logging
logging.basicConfig(
    filename=str(LOG_PATH / "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --- whitelist load ---
WHITELIST = set()
if WHITELIST_PATH.exists():
    try:
        with open(WHITELIST_PATH, "r", encoding="utf-8") as fh:
            WHITELIST = set(line.strip().lower() for line in fh if line.strip() and not line.strip().startswith("#"))
        print(f"[api] whitelist loaded from {WHITELIST_PATH}: {len(WHITELIST)} entries")
    except Exception as e:
        print("[api] failed loading whitelist:", str(e))

BUILTIN_WHITELIST = {
    "whatsapp.com", "web.whatsapp.com", "support.whatsapp.com",
    "youtube.com", "google.com", "accounts.google.com",
    "facebook.com", "twitter.com", "github.com", "microsoft.com",
}

# --- blacklist load ---
BLACKLIST = set()
BLACKLIST_RAW = []
if BLACKLIST_PATH.exists():
    try:
        with open(BLACKLIST_PATH, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip() and not line.strip().startswith("#")]
            BLACKLIST_RAW = lines[:]
            for line in lines:
                BLACKLIST.add(line.lower())
        print(f"[api] blacklist loaded from {BLACKLIST_PATH}: {len(BLACKLIST)} entries")
    except Exception as e:
        print("[api] failed loading blacklist:", str(e))
else:
    print("[api] no blacklist file found (ok).")

# --- shap availability ---
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# --- global model objects ---
model = None
FEATURE_COLUMNS = []
SHAP_EXPLAINER = None

# --- helpers ---
def _sanitize_for_json(x):
    if x is None:
        return None
    if isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _sanitize_for_json(v) for k,v in x.items()}
    try:
        import pandas as _pd
        if isinstance(x, _pd.Series):
            return _sanitize_for_json(x.to_dict())
        if isinstance(x, _pd.DataFrame):
            return _sanitize_for_json(x.to_dict(orient="records"))
    except Exception:
        pass
    try:
        if hasattr(x, "tolist"):
            return _sanitize_for_json(x.tolist())
    except Exception:
        pass
    try:
        return str(x)
    except Exception:
        return None

def canonicalize_url(u):
    if u is None:
        return ""
    s = str(u).strip()
    s = s.split('#')[0]
    try:
        s = _re.sub(r'(\?|&)(utm_[^=]+|fbclid|gclid)=[^&]*', '', s)
        s = _re.sub(r'\?&|&&', '?', s)
    except Exception:
        pass
    return s

def get_domain_simple(url):
    try:
        parsed = urlparse(url if '://' in url else ('http://' + url))
        host = parsed.netloc or parsed.path
        host = host.split(':')[0]
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""

# --- model & shap loader ---
def load_model_and_prepare():
    """
    Load the serialized model, feature names and (optionally) SHAP explainer.
    This function is idempotent and can be called at startup or lazily at request time.
    """
    global model, FEATURE_COLUMNS, SHAP_EXPLAINER

    # load model
    if not MODEL_PATH.exists():
        print(f"[api] model file not found at {MODEL_PATH}")
        model = None
        return

    try:
        model = joblib.load(MODEL_PATH)
        print("[api] model loaded from", MODEL_PATH)
    except Exception as e:
        print("[api] failed to load model:", str(e))
        traceback.print_exc()
        model = None
        return

    # load feature names (optional)
    FEATURE_COLUMNS = []
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as fh:
                FEATURE_COLUMNS = json.load(fh)
            print(f"[api] loaded {len(FEATURE_COLUMNS)} feature names from {FEATURES_PATH}")
        except Exception as e:
            print("[api] failed reading feature names:", str(e))
            FEATURE_COLUMNS = []
    else:
        # try to infer a sample from feature_extractor
        try:
            sample = extract_features("https://example.com")
            if isinstance(sample, dict):
                FEATURE_COLUMNS = list(sample.keys())
                print(f"[api] inferred {len(FEATURE_COLUMNS)} feature names from feature_extractor")
        except Exception:
            FEATURE_COLUMNS = []

    # prepare SHAP explainer if possible
    SHAP_EXPLAINER = None
    if SHAP_AVAILABLE:
        try:
            # build a sensible background (if CSV present, use it)
            if BACKGROUND_CSV.exists():
                df_bg = pd.read_csv(BACKGROUND_CSV, low_memory=False)
                if FEATURE_COLUMNS:
                    for c in FEATURE_COLUMNS:
                        if c not in df_bg.columns:
                            df_bg[c] = -1
                    Xbg = df_bg[FEATURE_COLUMNS].fillna(-1).select_dtypes(include=[np.number])
                else:
                    Xbg = df_bg.fillna(-1).select_dtypes(include=[np.number])
                if len(Xbg) > 200:
                    Xbg = Xbg.sample(200, random_state=42)
            else:
                if FEATURE_COLUMNS:
                    Xbg = pd.DataFrame(np.zeros((10, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
                else:
                    Xbg = pd.DataFrame(np.zeros((10, 1)))

            # extract a final estimator if model is a pipeline
            final_model = model
            try:
                if hasattr(model, "named_steps"):
                    if "clf" in model.named_steps:
                        final_model = model.named_steps["clf"]
                    else:
                        final_model = list(model.named_steps.values())[-1]
            except Exception:
                final_model = model

            use_explainer = None
            # tree explainer if possible
            if hasattr(final_model, "predict_proba") and hasattr(final_model, "feature_importances_"):
                try:
                    SHAP_EXPLAINER = shap.TreeExplainer(final_model, Xbg)
                    use_explainer = "tree"
                except Exception:
                    SHAP_EXPLAINER = None

            # callable fallback
            if SHAP_EXPLAINER is None:
                try:
                    if hasattr(final_model, "predict_proba"):
                        model_callable = lambda x: final_model.predict_proba(x)[:,1]
                    else:
                        model_callable = lambda x: final_model.predict(x)
                    try:
                        masker = shap.maskers.Independent(Xbg)
                    except Exception:
                        masker = None
                    SHAP_EXPLAINER = shap.Explainer(model_callable, masker if masker is not None else Xbg)
                    use_explainer = "callable"
                except Exception:
                    SHAP_EXPLAINER = None

            if SHAP_EXPLAINER is not None:
                print(f"[api] SHAP explainer prepared (mode={use_explainer}).")
            else:
                print("[api] SHAP explainer not prepared (fallback).")
        except Exception as e:
            SHAP_EXPLAINER = None
            print("[api] SHAP initialisation failed:", str(e))
    else:
        print("[api] shap not installed — SHAP explanations disabled.")

# attempt to load at startup (best-effort)
try:
    load_model_and_prepare()
except Exception:
    traceback.print_exc()

# ---------------- importance & shap helpers ----------------
def get_global_importances(model_obj, feature_names):
    try:
        clf = model_obj
        if hasattr(model_obj, "named_steps"):
            if "clf" in model_obj.named_steps:
                clf = model_obj.named_steps["clf"]
            else:
                clf = list(model_obj.named_steps.values())[-1]
        if hasattr(clf, "feature_importances_"):
            imps = np.array(clf.feature_importances_, dtype=float)
            if len(imps) == len(feature_names):
                return imps
            return np.pad(imps, (0, max(0, len(feature_names)-len(imps))), 'constant')
        try:
            if callable(getattr(clf, "get_booster", None)):
                b = clf.get_booster()
                sc = b.get_score(importance_type="gain")
                imps = [float(sc.get(f"f{idx}", 0.0)) for idx in range(len(feature_names))]
                return np.array(imps, dtype=float)
        except Exception:
            pass
    except Exception:
        pass
    return np.zeros(len(feature_names), dtype=float)

def top_k_features_from_importance(model_obj, feature_names, x_row, k=3):
    imps = get_global_importances(model_obj, feature_names)
    if len(imps) != len(feature_names):
        imps = np.zeros(len(feature_names), dtype=float)
    idx = np.argsort(imps)[::-1][:k]
    out = []
    for i in idx:
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        importance = float(imps[i]) if i < len(imps) else 0.0
        try:
            val = x_row.get(fname, None)
        except Exception:
            val = None
        out.append({"feature": fname, "importance": importance, "value": _sanitize_for_json(val)})
    return out

def shap_local_explanation(X_row):
    if SHAP_EXPLAINER is None:
        return {"error": "shap_unavailable"}
    try:
        try:
            sv = SHAP_EXPLAINER.shap_values(X_row)
        except Exception:
            sv = SHAP_EXPLAINER(X_row)
        if isinstance(sv, list) and len(sv) == 2:
            arr = np.array(sv[1]).flatten()
        else:
            try:
                arr = np.array(sv.values).flatten()
            except Exception:
                arr = np.array(sv).flatten()
        pairs = []
        for i, fname in enumerate(FEATURE_COLUMNS):
            val = X_row.iloc[0].get(fname, None) if fname in X_row.columns else None
            shapv = float(arr[i]) if i < len(arr) else 0.0
            pairs.append((fname, val, shapv))
        pos = sorted([p for p in pairs if p[2] > 0], key=lambda x: x[2], reverse=True)[:3]
        neg = sorted([p for p in pairs if p[2] < 0], key=lambda x: x[2])[:3]
        def serial(lst):
            out = []
            for a,b,c in lst:
                out.append({"feature": a, "value": _sanitize_for_json(b), "shap_value": float(c)})
            return out
        return {"positive": serial(pos), "negative": serial(neg)}
    except Exception as e:
        return {"error": "shap_failed", "message": str(e)}

# ---------------- Demo page ----------------
@app.route("/", methods=["GET"])
def index():
    # Modern, clean demo UI — keeps the same JS IDs and behavior
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>PhishSense — Demo</title>
      <style>
        :root{
          --bg:#0b0d0f;
          --panel:#0f1113;
          --muted:#9aa4ae;
          --accent:#10b981; /* green */
          --danger:#ef4444; /* red */
          --card:#0b0c0d;
          --card-border: rgba(255,255,255,0.04);
          --glass: rgba(255,255,255,0.02);
          --accent-2: #4b5563;
          --radius:14px;
          --maxw:1000px;
        }
        html,body{height:100%;margin:0;background:linear-gradient(180deg,#070708 0%, #0b0d0f 60%);font-family:Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;color:#e6eef6}
        .site{max-width:var(--maxw);margin:28px auto;padding:28px}
        header{display:flex;align-items:center;gap:18px;margin-bottom:20px}
        .logo{
          width:72px;height:72px;border-radius:12px;background:linear-gradient(135deg,#0ea5a4,#065f46);
          display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px;color:white;box-shadow:0 6px 24px rgba(2,6,23,0.5)
        }
        h1{margin:0;font-size:28px;letter-spacing:-0.5px}
        p.lead{color:var(--muted);margin-top:6px;margin-bottom:18px}

        .card{background:var(--card);border:1px solid var(--card-border);border-radius:var(--radius);padding:18px;box-shadow:0 6px 28px rgba(2,6,23,0.45)}
        .controls{display:grid;grid-template-columns:1fr 260px;gap:12px;align-items:center}
        .controls .small{font-size:13px;color:var(--muted)}
        input[type="text"], input[type="url"]{
          width:100%;padding:12px 14px;border-radius:10px;border:1px solid rgba(255,255,255,0.04);background:var(--glass);color:inherit;outline:none;font-size:15px;box-sizing:border-box;transition:box-shadow .15s,transform .06s
        }
        input[type="text"]:focus, input[type="url"]:focus{box-shadow:0 8px 30px rgba(2,6,23,0.6);transform:translateY(-1px);border-color:rgba(255,255,255,0.06)}
        .btn{
          display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:10px;background:var(--accent-2);color:white;border:none;cursor:pointer;font-weight:600;box-shadow:0 6px 18px rgba(2,6,23,0.45);transition:transform .08s,background .12s
        }
        .btn:hover{transform:translateY(-2px)}
        .row{display:flex;gap:12px;align-items:center;margin-top:12px}

        .result-wrap{margin-top:18px}
        .label-pill{display:inline-block;padding:16px 22px;border-radius:10px;font-weight:900;font-size:34px}
        .phish{background:linear-gradient(90deg,#7f1d1d,#b91c1c);color:white}
        .legit{background:linear-gradient(90deg,#14532d,#059669);color:white}

        .meta{margin-top:10px;color:var(--muted);font-size:14px}
        pre#out{background:#050607;padding:12px;border-radius:10px;overflow:auto;color:#dbeafe;margin-top:12px;border:1px solid rgba(255,255,255,0.03);min-height:36px}

        footer{margin-top:18px;color:var(--muted);font-size:13px}

        /* responsive */
        @media (max-width:780px){
          .controls{grid-template-columns:1fr;gap:10px}
          .controls .small{display:none}
          .label-pill{font-size:28px;padding:12px 16px}
        }
      </style>
    </head>
    <body>
      <div class="site">
        <header>
          <div class="logo">PS</div>
          <div>
            <h1>PhishSense — Demo</h1>
            <p class="lead">Quickly check a URL. Results are explainable and include SHAP (if available).</p>
          </div>
        </header>

        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
            <div style="flex:1;min-width:0">
              <div class="controls">
                <input id="url" type="url" placeholder="https://example.com/path/login" />
                <div style="display:flex;flex-direction:column;gap:8px;">
                  <input id="threshold" type="text" placeholder="threshold (0-1, default {PHISH_THRESHOLD_DEFAULT})" />
                  <div style="display:flex;gap:8px;">
                    <button class="btn" onclick="send()">Check</button>
                    <button class="btn" onclick="document.getElementById('url').value='https://web.whatsapp.com'">Demo: WhatsApp</button>
                  </div>
                </div>
              </div>
              <div class="row">
                <div id="status" class="small" style="color:var(--muted)"></div>
              </div>
            </div>
          </div>

          <div class="result-wrap">
            <div id="label" style="margin-bottom:12px"></div>
            <div class="meta" id="meta"></div>
            <pre id="out"></pre>
          </div>
        </div>

        <footer>
          <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
            <div>Tip: use the threshold box to increase/decrease sensitivity. Built-in whitelist & blacklist are applied on the server.</div>
            <div>PhishSense • Demo</div>
          </div>
        </footer>
      </div>

      <script>
      async function send(){
        const u = document.getElementById('url').value.trim();
        let t = document.getElementById('threshold').value.trim();
        if(!u){ alert('Enter a URL'); return; }
        const payload = {url: u};
        let uri = '/predict';
        if(t){
          const v = parseFloat(t);
          if(!isNaN(v)) uri = uri + '?threshold=' + v;
        }
        document.getElementById('status').textContent = 'Querying…';
        document.getElementById('meta').textContent = '';
        try{
          const res = await fetch(uri, {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload)
          });
          const j = await res.json();
          document.getElementById('out').textContent = JSON.stringify(j, null, 2);
          document.getElementById('status').textContent = 'Status: ' + res.status;

          // update big label
          const labelEl = document.getElementById('label');
          labelEl.innerHTML = '';
          if(res.status === 200){
            const flag = j.phishing_flag;
            const prob = j.phishing_probability;
            const threshold = j.threshold_used ?? {PHISH_THRESHOLD_DEFAULT};
            const span = document.createElement('span');
            span.className = 'label-pill ' + (flag ? 'phish' : 'legit');
            span.textContent = flag ? 'PHISHING' : 'LEGIT';
            labelEl.appendChild(span);

            const m = document.getElementById('meta');
            m.textContent = `prob=${(prob||0).toFixed(3)}   threshold=${threshold}`;
            // if shap present show small hint (we show full JSON in the box already)
            if(j.shap_explanation && !j.shap_explanation.error){
              // optional: display top positive contributors
              const pos = j.shap_explanation.positive || [];
              const neg = j.shap_explanation.negative || [];
              let hint = '';
              if(pos.length) hint += '↑ ' + pos.map(x => x.feature).join(', ');
              if(neg.length) hint += (hint ? '  •  ' : '') + '↓ ' + neg.map(x => x.feature).join(', ');
              if(hint) m.textContent += '   •   top:' + hint;
            }
          } else {
            // error or non-200
            const span = document.createElement('span');
            span.className = 'label-pill legit';
            span.textContent = 'RESULT';
            labelEl.appendChild(span);
          }
        } catch(err){
          document.getElementById('out').textContent = String(err);
          document.getElementById('status').textContent = 'Request failed';
        }
      }
      // allow Enter to submit from the URL field
      document.getElementById('url').addEventListener('keydown', function(e){
        if(e.key === 'Enter') send();
      });
      </script>
    </body>
    </html>
    """
    return html.replace("{PHISH_THRESHOLD_DEFAULT}", str(PHISH_THRESHOLD_DEFAULT))

# ---------------- /predict route ----------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON { "url": "https://..." }
    Optional query param: ?threshold=0.60
    """
    try:
        # threshold
        try:
            thr = float(request.args.get("threshold", PHISH_THRESHOLD_DEFAULT))
            if thr < 0 or thr > 1:
                thr = PHISH_THRESHOLD_DEFAULT
        except Exception:
            thr = PHISH_THRESHOLD_DEFAULT

        # lazy model load if missing
        global model, FEATURE_COLUMNS, SHAP_EXPLAINER
        if model is None:
            load_model_and_prepare()
            if model is None:
                return jsonify({"error": "model not loaded"}), 500

        # accept input
        data = request.get_json(force=True) if request.is_json else (request.form.to_dict() or request.args.to_dict())
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "JSON must contain 'url'"}), 400

        url_c = canonicalize_url(url)
        host = get_domain_simple(url_c)

        # whitelist check
        combined_whitelist = set(WHITELIST) | set(BUILTIN_WHITELIST)
        if any(host == w or host.endswith("." + w) for w in combined_whitelist):
            resp = {
                "url": url,
                "phishing_probability": 0.0,
                "phishing_flag": 0,
                "threshold_used": float(thr),
                "top_features": [],
                "shap_explanation": None,
                "raw_features": {"whitelist_hit": host},
                "note": "Domain found in whitelist -> forced LEGIT for demo."
            }
            logging.info(f"WHITELIST HIT: {host} url={url}")
            return jsonify(_sanitize_for_json(resp))

        # blacklist check (host & substring)
        host_l = host.lower()
        if any(host_l == b or host_l.endswith(b) for b in BLACKLIST):
            resp = {
                "url": url,
                "phishing_probability": 1.0,
                "phishing_flag": 1,
                "threshold_used": float(thr),
                "top_features": [],
                "shap_explanation": None,
                "raw_features": {"blacklist_hit": host_l},
                "note": "Domain found in blacklist -> forced PHISHING for demo."
            }
            logging.warning(f"BLACKLIST HIT (host): {host} url={url}")
            return jsonify(_sanitize_for_json(resp))
        url_l = url_c.lower()
        for bl in BLACKLIST_RAW:
            if bl.lower() in url_l:
                resp = {
                    "url": url,
                    "phishing_probability": 1.0,
                    "phishing_flag": 1,
                    "threshold_used": float(thr),
                    "top_features": [],
                    "shap_explanation": None,
                    "raw_features": {"blacklist_hit": bl},
                    "note": "Blacklist substring matched -> forced PHISHING for demo."
                }
                logging.warning(f"BLACKLIST HIT (substr): {bl} url={url}")
                return jsonify(_sanitize_for_json(resp))

        # feature extraction
        try:
            feats = extract_features(url)
            raw_feats = dict(feats) if isinstance(feats, dict) else {}
        except Exception as e:
            raw_feats = {"extract_error": str(e)}
            feats = {}
        X = pd.DataFrame([feats])
        if FEATURE_COLUMNS:
            for c in FEATURE_COLUMNS:
                if c not in X.columns:
                    X[c] = -1
            X_aligned = X[FEATURE_COLUMNS].fillna(-1)
        else:
            X_aligned = X.fillna(-1)

        # predict
        try:
            if hasattr(model, "predict_proba"):
                try:
                    proba = float(model.predict_proba(X_aligned)[:,1][0])
                except Exception:
                    try:
                        proba = float(model.predict_proba([url_c])[0][1])
                    except Exception:
                        proba = float(model.predict(X_aligned)[0])
            else:
                pred = int(model.predict(X_aligned)[0])
                proba = float(pred)
        except Exception as e:
            logging.exception("Prediction failed")
            return jsonify({"error": "prediction failed", "message": str(e)}), 500

        label = 1 if proba >= thr else 0

        # explain
        top_feats = top_k_features_from_importance(model, FEATURE_COLUMNS or list(X_aligned.columns), X_aligned.iloc[0], k=3)
        shap_info = None
        if SHAP_EXPLAINER is not None:
            try:
                shap_info = shap_local_explanation(X_aligned)
            except Exception as e:
                shap_info = {"error": "shap_runtime_error", "message": str(e)}

        resp = {
            "url": url,
            "phishing_probability": float(proba),
            "phishing_flag": int(label),
            "threshold_used": float(thr),
            "top_features": top_feats,
            "shap_explanation": shap_info,
            "raw_features": raw_feats,
            "note": "top_features = global importances. shap_explanation = local attribution (if available)."
        }

        logging.info(json.dumps({
            "url": url, "host": host, "prob": round(float(proba),4), "label": int(label), "threshold": float(thr)
        }))

        return jsonify(_sanitize_for_json(resp))

    except Exception as e:
        traceback.print_exc()
        logging.exception("Predict error")
        return jsonify({"error": str(e)}), 500

# run
if __name__ == "__main__":
    import socket
    def _get_local_ip():
        """Return the machine's LAN IP that other devices on the same network can use.
        Falls back to 127.0.0.1 if detection fails.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't actually send packets; used to discover outbound IP
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            try:
                s.close()
            except Exception:
                pass
        return ip

    host = "127.0.0.1"      # local-only (safe)
    port = 5000
    local_ip = _get_local_ip()

    print("Starting PhishSense API.")
    print("Model path:", MODEL_PATH)
    print("Feature_names:", FEATURES_PATH)
    print()
    print(f" * Local browser URL: http://{host}:{port}/")
    # show LAN-accessible URL too (if you want to access from phone/other PC on same Wi-Fi)
    if local_ip and local_ip != "127.0.0.1":
        print(f" * LAN URL (open from other devices on same network): http://{local_ip}:{port}/")
    else:
        print(" * LAN URL: not detected (you'll still be able to use http://127.0.0.1:5000 locally)")

    print()
    print("Note: The Flask dev server is for testing/demo only. To allow other devices to connect,")
    print("you may use host='0.0.0.0' (app.run(host='0.0.0.0', port=port)) and ensure your firewall allows the port.")
    print()

    # start server (keep debug=True if you want hot-reload)
    app.run(host=host, port=port, debug=True)
