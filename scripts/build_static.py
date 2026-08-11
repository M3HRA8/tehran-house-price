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

# Same cleaning logic used in the project.
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Browser-friendly RF: keeps quality close to the main project while
# making the static site small and fast on mobile.
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=14,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
metrics = {
    "r2": float(r2_score(y_test, pred)),
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
}

header = json.dumps(
    {
        "columns": X.columns.tolist(),
        "addresses": addresses,
        "dataset_size": dataset_size,
        "metrics": metrics,
    },
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")

buf = bytearray()
buf += b"RFP2"
buf += struct.pack("<I", len(header))
buf += header
buf += struct.pack("<I", len(rf.estimators_))

for estimator in rf.estimators_:
    tree = estimator.tree_
    buf += struct.pack("<I", tree.node_count)
    for i in range(tree.node_count):
        buf += struct.pack(
            "<iihff",
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
            "format": "RFP2",
            "encoding": "base64+gzip",
            "chunks": chunk_files,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

shutil.copy2(TEMPLATE, OUT / "index.html")
(OUT / ".nojekyll").write_text("", encoding="utf-8")

print(f"Built static site with {dataset_size} rows")
print(f"R2={metrics['r2']:.4f}")
print(f"MAE={metrics['mae']:,.0f}")
print(f"RMSE={metrics['rmse']:,.0f}")
print(f"Compressed browser model={len(compressed)/1024:.1f} KB")
