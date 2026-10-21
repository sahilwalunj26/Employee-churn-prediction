import math, os, pickle
import pandas as pd
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "naive_bayes.pkl"), "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Category -> integer codes. ASSUMPTION: alphabetical LabelEncoder order used in training.
# If your notebook encoded differently, edit these maps only.
CATEGORIES = {
    "Education": {"Bachelors": 0, "Masters": 1, "PHD": 2},
    "City": {"Bangalore": 0, "New Delhi": 1, "Pune": 2},
    "Gender": {"Female": 0, "Male": 1},
    "EverBenched": {"No": 0, "Yes": 1},
}
RANGES = {"JoiningYear": (2000, 2030), "PaymentTier": (1, 3),
          "Age": (16, 70), "ExperienceInCurrentDomain": (0, 50)}
LABELS = {"ExperienceInCurrentDomain": "Experience", "JoiningYear": "Joining year",
          "PaymentTier": "Payment tier", "EverBenched": "Ever benched"}

# Training-set mean/variance per feature (for "unusual value" hints)
w = model.class_count_ / model.class_count_.sum()
MEAN = (model.theta_ * w[:, None]).sum(0)
STD = ((model.var_ + model.theta_ ** 2) * w[:, None]).sum(0) - MEAN ** 2
STD = STD.clip(min=1e-9) ** 0.5

app = Flask(__name__)

# ---------------- Frontend (HTML + CSS + JS in one string) ----------------
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Employee Attrition Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f7fc;--card:#fff;--ink:#0f1b3d;--mute:#5b6b8c;--line:#dfe7f5;--brand:#2563eb;--stay:#10b981;--warn:#f59e0b;--leave:#ef4444;--soft:#eef3fd}
@media(prefers-color-scheme:dark){:root{--bg:#0a1122;--card:#111a33;--ink:#eaf0ff;--mute:#93a3c7;--line:#223055;--soft:#18244a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 Inter,system-ui,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 48px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:24px}
h1{font-size:clamp(26px,4vw,38px);margin:0;letter-spacing:-.02em;font-weight:800}
h1 span{color:var(--brand)}.sub{color:var(--mute);margin:6px 0 0}
.tools{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;font-weight:600;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:9px 14px;border-radius:10px;cursor:pointer;transition:.15s}
button:hover{border-color:var(--brand);color:var(--brand)}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:22px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 8px 30px rgba(30,60,140,.06)}
.card h2{font-size:13px;letter-spacing:.09em;text-transform:uppercase;color:var(--brand);margin:0 0 16px}
.card h2:not(:first-child){margin-top:26px}
.field{margin-bottom:18px}.field>label,.lab{display:flex;justify-content:space-between;font-weight:600;font-size:14px;margin-bottom:8px}
.lab output{color:var(--brand);font-weight:700}
.seg{display:flex;background:var(--soft);border-radius:12px;padding:4px;gap:4px}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg label{flex:1;text-align:center;padding:9px 6px;border-radius:9px;cursor:pointer;font-weight:600;font-size:14px;color:var(--mute);transition:.15s}
.seg input:checked+label{background:var(--card);color:var(--brand);box-shadow:0 2px 8px rgba(30,60,140,.15)}
.seg input:focus-visible+label{outline:2px solid var(--brand)}
input[type=range]{-webkit-appearance:none;width:100%;height:8px;border-radius:99px;outline:none;background:linear-gradient(90deg,var(--brand) var(--p,50%),var(--soft) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;border-radius:50%;background:#fff;border:4px solid var(--brand);cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25)}
input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:#fff;border:4px solid var(--brand);cursor:pointer}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:520px){.two{grid-template-columns:1fr}}
.result{position:sticky;top:20px;text-align:center}
.gauge{position:relative;width:220px;height:220px;margin:4px auto 10px}
.gauge svg{transform:rotate(-90deg)}.gauge circle{fill:none;stroke-width:14}
.track{stroke:var(--soft)}.bar{stroke:var(--c,var(--brand));stroke-linecap:round;transition:stroke-dashoffset .7s cubic-bezier(.2,.8,.2,1),stroke .4s}
.gtxt{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.pct{font-size:46px;font-weight:800;letter-spacing:-.03em;color:var(--c,var(--ink))}.gtxt small{color:var(--mute);font-weight:600}
.verdict{display:inline-block;padding:8px 18px;border-radius:99px;font-weight:700;background:color-mix(in srgb,var(--c) 15%,transparent);color:var(--c);transition:.3s}
.hint{color:var(--mute);font-size:14px;margin:12px 0 0}
.fac{text-align:left;margin-top:22px;border-top:1px solid var(--line);padding-top:18px}
.fac h3{font-size:14px;margin:0 0 4px}.fac p{font-size:12.5px;color:var(--mute);margin:0 0 12px}
.row{display:grid;grid-template-columns:128px 1fr;gap:10px;align-items:center;margin-bottom:9px;font-size:13px}
.row b{display:block;font-weight:600}.row em{font-style:normal;color:var(--mute);font-size:12px}
.dv{position:relative;height:10px;background:var(--soft);border-radius:99px}
.dv:after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:var(--line)}
.dv i{position:absolute;top:0;height:100%;border-radius:99px;transition:.5s}
.legend{display:flex;justify-content:space-between;font-size:11.5px;color:var(--mute);margin:2px 0 10px 138px}
.warn{background:color-mix(in srgb,var(--warn) 14%,transparent);color:var(--ink);border-radius:10px;padding:8px 12px;font-size:13px;margin-top:10px;text-align:left}
.err{color:var(--leave);font-size:14px}
.foot{text-align:center;color:var(--mute);font-size:12.5px;margin-top:26px}
</style>
</head>
<body>
<div class="wrap">
<header>
 <div><h1>Employee <span>Attrition</span> Predictor</h1><p class="sub">Adjust the profile — the prediction updates instantly.</p></div>
 <div class="tools"><button id="ex1" type="button">✅ Example: Stays</button><button id="ex2" type="button">⚠️ Example: Leaves</button><button id="reset" type="button">↺ Reset</button></div>
</header>
<div class="grid">
<form class="card" id="form" onsubmit="return false">
 <h2>Personal</h2>
 <div class="field"><label>Gender</label><div class="seg">
  <input type="radio" name="Gender" id="g0" value="Female"><label for="g0">Female</label>
  <input type="radio" name="Gender" id="g1" value="Male" checked><label for="g1">Male</label></div></div>
 <div class="field"><div class="lab"><span>Age</span><output id="o_Age"></output></div><input type="range" name="Age" min="20" max="60" value="29"></div>
 <div class="field"><label>Education</label><div class="seg">
  <input type="radio" name="Education" id="e0" value="Bachelors" checked><label for="e0">Bachelors</label>
  <input type="radio" name="Education" id="e1" value="Masters"><label for="e1">Masters</label>
  <input type="radio" name="Education" id="e2" value="PHD"><label for="e2">PhD</label></div></div>
 <h2>Job</h2>
 <div class="field"><label>City</label><div class="seg">
  <input type="radio" name="City" id="c0" value="Bangalore" checked><label for="c0">Bangalore</label>
  <input type="radio" name="City" id="c1" value="New Delhi"><label for="c1">New Delhi</label>
  <input type="radio" name="City" id="c2" value="Pune"><label for="c2">Pune</label></div></div>
 <div class="field"><label>Payment tier <small style="font-weight:400;color:var(--mute)">(1 = highest)</small></label><div class="seg">
  <input type="radio" name="PaymentTier" id="p1" value="1"><label for="p1">Tier 1</label>
  <input type="radio" name="PaymentTier" id="p2" value="2"><label for="p2">Tier 2</label>
  <input type="radio" name="PaymentTier" id="p3" value="3" checked><label for="p3">Tier 3</label></div></div>
 <div class="two">
  <div class="field"><div class="lab"><span>Joining year</span><output id="o_JoiningYear"></output></div><input type="range" name="JoiningYear" min="2012" max="2018" value="2015"></div>
  <div class="field"><div class="lab"><span>Years in current domain</span><output id="o_ExperienceInCurrentDomain"></output></div><input type="range" name="ExperienceInCurrentDomain" min="0" max="10" value="3"></div>
 </div>
 <div class="field" style="margin-bottom:0"><label>Ever benched (kept without a project)?</label><div class="seg">
  <input type="radio" name="EverBenched" id="b0" value="No" checked><label for="b0">No</label>
  <input type="radio" name="EverBenched" id="b1" value="Yes"><label for="b1">Yes</label></div></div>
</form>

<aside class="card result" id="res">
 <h2>Prediction</h2>
 <div class="gauge"><svg width="220" height="220" viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="54"/><circle class="bar" id="bar" cx="60" cy="60" r="54" stroke-dasharray="339.29" stroke-dashoffset="339.29"/></svg>
  <div class="gtxt"><div class="pct" id="pct">–</div><small>chance of leaving</small></div></div>
 <div class="verdict" id="verdict">Calculating…</div>
 <p class="hint" id="hint"></p><div id="warns"></div>
 <div class="fac"><h3>What's driving this?</h3><p>Each bar shows how much a value shifts the model toward “leaves” (right) or “stays” (left).</p>
  <div class="legend"><span>← Stays</span><span>Leaves →</span></div><div id="facs"></div></div>
</aside>
</div>
<p class="foot">Gaussian Naive Bayes model · For decision support only, not a final verdict on any individual.</p>
</div>
<script>
const $=s=>document.querySelector(s), form=$('#form');
const DEF={Gender:'Male',Age:29,Education:'Bachelors',City:'Bangalore',PaymentTier:'3',JoiningYear:2015,ExperienceInCurrentDomain:3,EverBenched:'No'};
const EX1={Gender:'Male',Age:34,Education:'Masters',City:'Bangalore',PaymentTier:'3',JoiningYear:2013,ExperienceInCurrentDomain:5,EverBenched:'No'};
const EX2={Gender:'Female',Age:26,Education:'Masters',City:'Pune',PaymentTier:'2',JoiningYear:2017,ExperienceInCurrentDomain:2,EverBenched:'Yes'};
function read(){const d={};for(const n of Object.keys(DEF)){const el=form.elements[n];d[n]=el.value!==undefined&&el.type==='range'?+el.value:(form.querySelector(`input[name=${n}]:checked`)||{}).value}return d}
function write(d){for(const [n,v] of Object.entries(d)){const els=form.querySelectorAll(`[name=${n}]`);els.forEach(e=>{if(e.type==='range')e.value=v;else e.checked=e.value==String(v)})}sync();run()}
function sync(){form.querySelectorAll('input[type=range]').forEach(r=>{const p=(r.value-r.min)/(r.max-r.min)*100;r.style.setProperty('--p',p+'%');$('#o_'+r.name).textContent=r.value})}
let t,seq=0,shown=0;
function run(){clearTimeout(t);t=setTimeout(async()=>{const my=++seq;try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(read())});const j=await r.json();if(my!==seq)return;if(!r.ok)throw new Error(j.error);render(j)}catch(e){$('#verdict').textContent='Error';$('#hint').innerHTML='<span class="err">'+e.message+'</span>'}},180)}
function render(j){const p=j.p_leave,c=p<.4?'var(--stay)':p<.6?'var(--warn)':'var(--leave)';
 $('#res').style.setProperty('--c',c);$('#bar').style.strokeDashoffset=339.29*(1-p);
 const from=shown,to=Math.round(p*100),t0=performance.now();(function f(n){const k=Math.min(1,(n-t0)/600);$('#pct').textContent=Math.round(from+(to-from)*k)+'%';if(k<1)requestAnimationFrame(f)})(t0);shown=to;
 $('#verdict').textContent=p<.4?'Likely to stay':p<.6?'Uncertain — could go either way':'Likely to leave';
 $('#hint').textContent=`Stay ${(j.p_stay*100).toFixed(1)}%  ·  Leave ${(p*100).toFixed(1)}%`;
 $('#warns').innerHTML=j.warnings.map(w=>`<div class="warn">⚠️ ${w}</div>`).join('');
 const m=Math.max(...j.factors.map(f=>Math.abs(f.score)),.01);
 $('#facs').innerHTML=j.factors.map(f=>{const w=Math.abs(f.score)/m*50,pos=f.score>0;return `<div class="row"><div><b>${f.name}</b><em>${f.value}</em></div><div class="dv"><i style="${pos?'left:50%':'right:50%'};width:${w}%;background:${pos?'var(--leave)':'var(--stay)'}"></i></div></div>`}).join('')}
form.addEventListener('input',()=>{sync();run()});
$('#ex1').onclick=()=>write(EX1);$('#ex2').onclick=()=>write(EX2);$('#reset').onclick=()=>write(DEF);
write(DEF);
</script>
</body>
</html>
"""
# ---------------------------------------------------------------------------


def loglik(x, cls, i):
    t, v = model.theta_[cls][i], model.var_[cls][i]
    return -0.5 * math.log(2 * math.pi * v) - (x - t) ** 2 / (2 * v)


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row, shown = [], {}
    try:
        for name in FEATURES:
            raw = data[name]
            if name in CATEGORIES:
                row.append(CATEGORIES[name][str(raw)]); shown[name] = str(raw)
            else:
                val = float(raw); lo, hi = RANGES[name]
                if not lo <= val <= hi:
                    raise ValueError(f"{name} must be between {lo} and {hi}")
                row.append(val); shown[name] = str(int(val))
    except KeyError as e:
        return jsonify(error=f"Missing or invalid value: {e}"), 400
    except ValueError as e:
        return jsonify(error=str(e)), 400

    X = pd.DataFrame([row], columns=FEATURES)
    p = model.predict_proba(X)[0]
    p_stay, p_leave = float(p[0]), float(p[1])

    factors, warnings = [], []
    for i, name in enumerate(FEATURES):
        c = loglik(row[i], 1, i) - loglik(row[i], 0, i)  # >0 pushes toward "leave"
        factors.append({"name": LABELS.get(name, name), "value": shown[name], "score": round(c, 3)})
        if abs(row[i] - MEAN[i]) > 3 * STD[i] and name not in CATEGORIES:
            warnings.append(f"{LABELS.get(name, name)} = {shown[name]} is unusual compared with the training data.")
    factors.sort(key=lambda f: -abs(f["score"]))
    return jsonify(leave=p_leave >= 0.5, p_leave=p_leave, p_stay=p_stay,
                   factors=factors, warnings=warnings)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
