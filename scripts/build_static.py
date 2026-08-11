from pathlib import Path
import base64, gzip, io, json, shutil, struct

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"
TEMPLATE = ROOT / "static" / "index.html"

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "model").mkdir(parents=True)

# Reconstruct the original CSV from the existing repository chunks.
encoded = "".join(
    p.read_text(encoding="ascii").strip()
    for p in sorted((ROOT / "data_chunks").glob("chunk_*.txt"))
)
raw_csv = gzip.decompress(base64.b64decode(encoded))
df = pd.read_csv(io.BytesIO(raw_csv))

# Exactly the same cleaning logic used by the deployed Streamlit version.
df["Area"] = (
    df["Area"].astype(str).str.replace(",", "", regex=False)
)
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df = df.dropna(subset=["Area", "Address"]).copy()
df = df[(df["Area"] > 0) & (df["Area"] <= 1000)].copy()

for col in ["Parking", "Warehouse", "Elevator"]:
    df[col] = df[col].astype(int)

addresses = sorted(df["Address"].astype(str).unique())
dataset_size = len(df)

model_df = df.drop(columns=["Price(USD)"]).copy()
model_df = pd.get_dummies(
    model_df,
    columns=["Address"],
    drop_first=True,
    dtype=int,
)

X = model_df.drop(columns=["Price"])
y = model_df["Price"]

# Evaluation is kept separate from the final production model.
# These metrics are from an 80/20 hold-out split and are not used for inference.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

eval_rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)
eval_rf.fit(X_train, y_train)
pred = eval_rf.predict(X_test)
metrics = {
    "r2": float(r2_score(y_test, pred)),
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
}

# Final production model: EXACTLY the same configuration as app.py.
# It is trained on all 3,451 cleaned rows, just like the Streamlit deployment.
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y)

header = json.dumps(
    {
        "columns": X.columns.tolist(),
        "addresses": addresses,
        "dataset_size": dataset_size,
        "metrics": metrics,
        "production_model": {
            "n_estimators": 300,
            "random_state": 42,
            "trained_rows": dataset_size,
        },
    },
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")

# RFP3 stores thresholds and leaf values as float64.
# JavaScript Number is also IEEE-754 float64, so browser inference keeps
# the same tree values/thresholds instead of the old float32 approximation.
buf = bytearray()
buf += b"RFP3"
buf += struct.pack("<I", len(header))
buf += header
buf += struct.pack("<I", len(rf.estimators_))

for estimator in rf.estimators_:
    tree = estimator.tree_
    buf += struct.pack("<I", tree.node_count)
    for i in range(tree.node_count):
        buf += struct.pack(
            "<iihdd",
            int(tree.children_left[i]),
            int(tree.children_right[i]),
            int(tree.feature[i]),
            float(tree.threshold[i]),
            float(tree.value[i][0][0]),
        )

compressed = gzip.compress(bytes(buf), compresslevel=9)
encoded_model = base64.b64encode(compressed).decode("ascii")
chunk_size = 80000
chunks = [
    encoded_model[i:i + chunk_size]
    for i in range(0, len(encoded_model), chunk_size)
]

chunk_files = []
for i, chunk in enumerate(chunks, 1):
    name = f"model/chunk_{i:02d}.txt"
    (OUT / name).write_text(chunk, encoding="ascii")
    chunk_files.append(name)

(OUT / "manifest.json").write_text(
    json.dumps(
        {
            "format": "RFP3",
            "encoding": "base64+gzip",
            "chunks": chunk_files,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# The checked-in template originally parsed the lightweight RFP2/float32 model.
# Patch only the parser in the generated site so the source UI stays unchanged.
site_html = TEMPLATE.read_text(encoding="utf-8")

old_magic = 'if(magic!=="RFP2")'
new_magic = 'if(magic!=="RFP3")'
if old_magic not in site_html:
    raise RuntimeError("Static template magic parser was not found.")
site_html = site_html.replace(old_magic, new_magic, 1)

old_arrays = 'threshold=new Float32Array(nodeCount),value=new Float32Array(nodeCount)'
new_arrays = 'threshold=new Float64Array(nodeCount),value=new Float64Array(nodeCount)'
if old_arrays not in site_html:
    raise RuntimeError("Static template float32 arrays were not found.")
site_html = site_html.replace(old_arrays, new_arrays, 1)

old_read = 'threshold[i]=view.getFloat32(p,true);p+=4;value[i]=view.getFloat32(p,true);p+=4'
new_read = 'threshold[i]=view.getFloat64(p,true);p+=8;value[i]=view.getFloat64(p,true);p+=8'
if old_read not in site_html:
    raise RuntimeError("Static template float32 reader was not found.")
site_html = site_html.replace(old_read, new_read, 1)

(OUT / "index.html").write_text(site_html, encoding="utf-8")
(OUT / ".nojekyll").write_text("", encoding="utf-8")

print(f"Built static site with {dataset_size} rows")
print("Production model: 300 trees, full cleaned dataset, float64 export")
print(f"Hold-out R2={metrics['r2']:.4f}")
print(f"Hold-out MAE={metrics['mae']:,.0f}")
print(f"Hold-out RMSE={metrics['rmse']:,.0f}")
print(f"Compressed exact browser model={len(compressed)/1024:.1f} KB")
print(f"Model chunks={len(chunks)}")
