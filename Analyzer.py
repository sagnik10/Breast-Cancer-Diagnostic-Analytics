import os
import re
import csv
import html
import time
import zipfile
import warnings
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")

start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_BASE = BASE_DIR
OUT = os.path.join(BASE_DIR, "Output")
CHART = os.path.join(OUT, "charts")
os.makedirs(CHART, exist_ok=True)

def clean_name(x):
    x = re.sub(r"[^0-9a-zA-Z]+", "_", str(x).strip().lower())
    return re.sub(r"_+", "_", x).strip("_")

def E(x):
    if pd.isna(x):
        return "N/A"
    return html.escape(str(x))

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except:
        return "N/A"

def fmt_num(x, d=3):
    try:
        return f"{float(x):,.{d}f}"
    except:
        return "N/A"

def fmt_pct(x):
    try:
        x = float(x)
        if abs(x) <= 1:
            x *= 100
        return f"{x:.1f}%"
    except:
        return "N/A"

def discover_files():
    files = []
    extract_root = os.path.join(BASE_DIR, "unzipped_input")
    os.makedirs(extract_root, exist_ok=True)
    for root, dirs, names in os.walk(INPUT_BASE):
        for name in names:
            path = os.path.join(root, name)
            low = name.lower()
            if low.endswith((".csv", ".xlsx", ".xls", ".sqlite", ".db")):
                files.append(path)
            elif low.endswith(".zip"):
                target = os.path.join(extract_root, clean_name(os.path.splitext(name)[0]))
                os.makedirs(target, exist_ok=True)
                try:
                    with zipfile.ZipFile(path, "r") as z:
                        z.extractall(target)
                except:
                    pass
    for root, dirs, names in os.walk(extract_root):
        for name in names:
            path = os.path.join(root, name)
            if name.lower().endswith((".csv", ".xlsx", ".xls", ".sqlite", ".db")):
                files.append(path)
    return sorted(set(files))

def read_csv_flexible(path):
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except pd.errors.ParserError:
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return pd.DataFrame()
            header = rows[0]
            width = len(header)
            fixed = []
            for row in rows[1:]:
                if len(row) < width:
                    row = row + [""] * (width - len(row))
                elif len(row) > width:
                    row = row[:width - 1] + [",".join(row[width - 1:])]
                fixed.append(row)
            return pd.DataFrame(fixed, columns=header)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, engine="python", on_bad_lines="skip")

def load_file(path):
    low = path.lower()
    if low.endswith(".csv"):
        return read_csv_flexible(path)
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if low.endswith((".sqlite", ".db")):
        conn = sqlite3.connect(path)
        tables = pd.read_sql("select name from sqlite_master where type='table'", conn)["name"].tolist()
        best = pd.DataFrame()
        for t in tables:
            temp = pd.read_sql(f'select * from "{t}"', conn)
            if len(temp) > len(best):
                best = temp
        conn.close()
        return best
    return pd.DataFrame()

def find_dataset():
    files = discover_files()
    if not files:
        raise ValueError("No input files found under /kaggle/input.")
    loaded = []
    for path in files:
        try:
            df = load_file(path)
            if len(df) > 0:
                loaded.append((path, df))
        except:
            pass
    if not loaded:
        raise ValueError("No readable input dataset found.")
    for path, df in loaded:
        cols = [clean_name(c) for c in df.columns]
        if "diagnosis" in cols and any("radius" in c for c in cols):
            return path, df
    return max(loaded, key=lambda x: len(x[1]))

data_path, df = find_dataset()
df.columns = [clean_name(c) for c in df.columns]

for c in df.columns:
    if c != "diagnosis":
        df[c] = pd.to_numeric(df[c], errors="ignore")

if "diagnosis" not in df.columns:
    raise ValueError("The dataset must contain a diagnosis column.")

df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
df["target"] = df["diagnosis"].map({"M": 1, "B": 0, "MALIGNANT": 1, "BENIGN": 0})
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ["target"] and c != "id"]
if "malignancy_risk_score" in feature_cols:
    risk_col = "malignancy_risk_score"
else:
    risk_col = None

title_style = ParagraphStyle("title", fontSize=22, leading=28, alignment=TA_CENTER, textColor=HexColor("#9F1239"), spaceAfter=18)
subtitle_style = ParagraphStyle("subtitle", fontSize=12, leading=16, alignment=TA_CENTER, textColor=HexColor("#334155"), spaceAfter=22)
heading_style = ParagraphStyle("heading", fontSize=16, leading=21, textColor=HexColor("#1D4ED8"), spaceAfter=10)
subheading_style = ParagraphStyle("subheading", fontSize=12.5, leading=17, textColor=HexColor("#111827"), spaceAfter=7)
body_style = ParagraphStyle("body", fontSize=9.4, leading=13.5, textColor=HexColor("#1F2937"), spaceAfter=8)
small_style = ParagraphStyle("small", fontSize=7.4, leading=9.1, textColor=HexColor("#1F2937"))
header_style = ParagraphStyle("header", fontSize=7.4, leading=9.1, textColor=white)

report_path = os.path.join(OUT, "Breast_Cancer_Enhanced_Dataset_Analytics_Report.pdf")
doc = SimpleDocTemplate(report_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=38, bottomMargin=38)
elements = []
chart_no = 0

def add_heading(x):
    elements.append(Paragraph(E(x), heading_style))

def add_body(x):
    elements.append(Paragraph(x, body_style))

def add_table(title, rows, headers, widths=None, max_rows=20):
    if rows is None or len(rows) == 0:
        return
    elements.append(Paragraph(E(title), subheading_style))
    rows = rows[:max_rows]
    data = [[Paragraph(f"<b>{E(h)}</b>", header_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(E(v), small_style) for v in row])
    if widths is None:
        widths = [6.9 * inch / len(headers)] * len(headers)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#9F1239")),
        ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFF1F2"), white])
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.1 * inch))

def add_chart(title, fig, insight):
    global chart_no
    chart_no += 1
    path = os.path.join(CHART, f"{chart_no:02d}_{clean_name(title)[:70]}.png")
    try:
        fig.tight_layout()
    except:
        pass
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    elements.append(Paragraph(E(title), subheading_style))
    elements.append(Image(path, width=6.55 * inch, height=3.7 * inch))
    elements.append(Paragraph(insight, body_style))
    elements.append(Spacer(1, 0.09 * inch))

elements.append(Paragraph("Breast Cancer Enhanced Dataset Analytics Report", title_style))
elements.append(Paragraph("Automated Kaggle analysis for diagnosis patterns, tumor feature distributions, risk scoring, predictive modeling, clustering and anomaly detection", subtitle_style))
add_body(f"This report analyzes the uploaded dataset <b>{E(os.path.basename(data_path))}</b>. It contains <b>{fmt_int(len(df))}</b> observations and <b>{fmt_int(len(df.columns))}</b> columns after preprocessing.")

diagnosis_counts = df["diagnosis"].value_counts()
malignant_rate = df["target"].mean()
summary_rows = [
    ["Rows", fmt_int(len(df)), "Total usable observations after diagnosis cleaning"],
    ["Columns", fmt_int(len(df.columns)), "Total available fields including engineered features"],
    ["Numeric predictors", fmt_int(len(feature_cols)), "Candidate modeling variables"],
    ["Benign cases", fmt_int((df["target"] == 0).sum()), "Diagnosis coded as benign"],
    ["Malignant cases", fmt_int((df["target"] == 1).sum()), "Diagnosis coded as malignant"],
    ["Malignancy rate", fmt_pct(malignant_rate), "Share of malignant diagnosis records"]
]
if risk_col:
    summary_rows.append(["Average malignancy risk score", fmt_num(df[risk_col].mean(), 3), "Mean engineered risk-score value"])
add_table("Executive Summary Metrics", summary_rows, ["Metric", "Value", "Interpretation"], [2.0 * inch, 1.5 * inch, 3.4 * inch], 12)

missing_rows = [[c, fmt_int(df[c].isna().sum()), fmt_pct(df[c].isna().mean())] for c in df.columns if df[c].isna().sum() > 0]
add_table("Missing Value Summary", missing_rows, ["Column", "Missing Count", "Missing Percent"], [2.8 * inch, 1.5 * inch, 1.5 * inch], 20)

elements.append(PageBreak())
add_heading("Diagnosis Distribution and Core Tumor Measurements")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(diagnosis_counts.index.astype(str), diagnosis_counts.values, color="#9F1239")
ax.set_title("Diagnosis Class Distribution")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Records")
add_chart("Diagnosis Class Distribution", fig, f"The dataset contains <b>{fmt_int((df['target'] == 0).sum())}</b> benign and <b>{fmt_int((df['target'] == 1).sum())}</b> malignant cases.")

main_features = [c for c in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "malignancy_risk_score"] if c in df.columns]
for c in main_features[:6]:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    vals_b = df.loc[df["target"] == 0, c].dropna()
    vals_m = df.loc[df["target"] == 1, c].dropna()
    ax.hist(vals_b, bins=35, alpha=0.55, label="Benign")
    ax.hist(vals_m, bins=35, alpha=0.55, label="Malignant")
    ax.set_title(f"{c} Distribution by Diagnosis")
    ax.set_xlabel(c)
    ax.set_ylabel("Count")
    ax.legend()
    add_chart(f"{c} Distribution by Diagnosis", fig, f"Mean <b>{E(c)}</b> is <b>{fmt_num(vals_b.mean(), 3)}</b> for benign cases and <b>{fmt_num(vals_m.mean(), 3)}</b> for malignant cases.")

group_rows = []
for c in feature_cols:
    b = df.loc[df["target"] == 0, c]
    m = df.loc[df["target"] == 1, c]
    diff = m.mean() - b.mean()
    group_rows.append([c, fmt_num(b.mean(), 3), fmt_num(m.mean(), 3), fmt_num(diff, 3), fmt_num(abs(diff) / (df[c].std() if df[c].std() else np.nan), 3)])
group_rows = sorted(group_rows, key=lambda x: float(x[4].replace(",", "")) if x[4] != "N/A" else -1, reverse=True)
add_table("Largest Diagnosis-Level Feature Differences", group_rows, ["Feature", "Benign Mean", "Malignant Mean", "Difference", "Std Difference"], [2.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch], 20)

elements.append(PageBreak())
add_heading("Correlation and Feature Relationship Analysis")

corr_cols = [c for c in feature_cols if df[c].notna().sum() > 10]
corr = df[corr_cols + ["target"]].corr(numeric_only=True)
target_corr = corr["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)
add_table("Top Correlations with Malignant Diagnosis", [[k, fmt_num(v, 4)] for k, v in target_corr.head(20).items()], ["Feature", "Correlation with Target"], [3.8 * inch, 2.2 * inch], 20)

fig, ax = plt.subplots(figsize=(9, 5))
top_corr = target_corr.head(15).sort_values()
ax.barh(top_corr.index, top_corr.values, color="#BE123C")
ax.set_title("Top Feature Correlations with Malignancy")
ax.set_xlabel("Correlation")
ax.tick_params(axis="y", labelsize=7.5)
add_chart("Top Feature Correlations with Malignancy", fig, f"The strongest absolute linear association with malignancy is <b>{E(target_corr.index[0])}</b> with correlation <b>{fmt_num(target_corr.iloc[0], 4)}</b>.")

if len(corr_cols) >= 2:
    heat_cols = target_corr.head(min(12, len(target_corr))).index.tolist()
    mat = df[heat_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(len(heat_cols)))
    ax.set_yticks(range(len(heat_cols)))
    ax.set_xticklabels(heat_cols, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(heat_cols, fontsize=7)
    ax.set_title("Correlation Matrix of Top Predictive Features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_chart("Top Feature Correlation Matrix", fig, "The heatmap highlights redundant or strongly related tumor measurements that may carry overlapping diagnostic information.")

scatter_pairs = [("radius_mean", "texture_mean"), ("radius_mean", "concavity_mean"), ("area_mean", "perimeter_mean"), ("radius_concavity_interaction", "malignancy_risk_score")]
for x, y in scatter_pairs:
    if x in df.columns and y in df.columns:
        fig, ax = plt.subplots(figsize=(8.6, 5))
        ax.scatter(df.loc[df["target"] == 0, x], df.loc[df["target"] == 0, y], alpha=0.35, label="Benign")
        ax.scatter(df.loc[df["target"] == 1, x], df.loc[df["target"] == 1, y], alpha=0.35, label="Malignant")
        ax.set_title(f"{y} vs {x}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        add_chart(f"{y} vs {x}", fig, f"This scatter plot compares <b>{E(x)}</b> and <b>{E(y)}</b> across benign and malignant records.")

elements.append(PageBreak())
add_heading("Predictive Modeling")

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
y = df["target"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
Xte_scaled = scaler.transform(Xte)

classifiers = [
    ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42))
]

model_rows = []
best_model = None
best_auc = -1
best_name = ""
best_pred = None
best_prob = None

for name, model in classifiers:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, prob)
    model_rows.append([name, fmt_num(acc, 4), fmt_num(auc, 4)])
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name
        best_pred = pred
        best_prob = prob

add_table("Diagnosis Classification Model Performance", model_rows, ["Model", "Accuracy", "ROC AUC"], [2.6 * inch, 1.5 * inch, 1.5 * inch], 5)

cm = confusion_matrix(yte, best_pred)
cm_rows = [["Actual Benign", fmt_int(cm[0, 0]), fmt_int(cm[0, 1])], ["Actual Malignant", fmt_int(cm[1, 0]), fmt_int(cm[1, 1])]]
add_table(f"{best_name} Confusion Matrix", cm_rows, ["", "Predicted Benign", "Predicted Malignant"], [2.0 * inch, 1.7 * inch, 1.9 * inch], 4)

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#9F1239")
    ax.set_title(f"{best_name} Feature Importance")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=7.5)
    add_chart("Diagnosis Classification Feature Importance", fig, f"The strongest model feature is <b>{E(imp.index[0])}</b>. The best classifier is <b>{E(best_name)}</b> with ROC AUC <b>{fmt_num(best_auc, 4)}</b>.")

if risk_col:
    reg_features = [c for c in feature_cols if c != risk_col]
    Xr = df[reg_features].replace([np.inf, -np.inf], np.nan).fillna(df[reg_features].median(numeric_only=True))
    yr = pd.to_numeric(df[risk_col], errors="coerce")
    ok = yr.notna()
    Xr = Xr.loc[ok]
    yr = yr.loc[ok]
    Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.22, random_state=42)
    regressors = [
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1)),
        ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42))
    ]
    reg_rows = []
    best_reg = None
    best_rmse = None
    best_r2 = None
    best_reg_name = ""
    for name, model in regressors:
        model.fit(Xrtr, yrtr)
        pred = model.predict(Xrte)
        rmse = np.sqrt(mean_squared_error(yrte, pred))
        r2 = r2_score(yrte, pred)
        reg_rows.append([name, fmt_num(rmse, 4), fmt_num(r2, 4)])
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_reg = model
            best_reg_name = name
    add_table("Malignancy Risk Score Regression Performance", reg_rows, ["Model", "RMSE", "R2"], [2.8 * inch, 1.4 * inch, 1.4 * inch], 5)
    if hasattr(best_reg, "feature_importances_"):
        imp = pd.Series(best_reg.feature_importances_, index=Xr.columns).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.barh(imp.index[::-1], imp.values[::-1], color="#1D4ED8")
        ax.set_title("Risk Score Regression Feature Importance")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=7.5)
        add_chart("Risk Score Feature Importance", fig, f"The best risk-score model is <b>{E(best_reg_name)}</b> with RMSE <b>{fmt_num(best_rmse, 4)}</b> and R2 <b>{fmt_num(best_r2, 4)}</b>.")

elements.append(PageBreak())
add_heading("Clustering and Anomaly Detection")

cluster_features = target_corr.head(min(10, len(target_corr))).index.tolist()
Xc = df[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(df[cluster_features].median(numeric_only=True))
Xcs = StandardScaler().fit_transform(Xc)

k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=20)
df["cluster"] = km.fit_predict(Xcs)
cluster_summary = df.groupby("cluster").agg(records=("target", "size"), malignancy_rate=("target", "mean")).reset_index()
add_table("Tumor Feature Cluster Summary", cluster_summary.values.tolist(), ["Cluster", "Records", "Malignancy Rate"], [1.2 * inch, 1.5 * inch, 1.8 * inch], 10)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
cluster_counts = df["cluster"].value_counts().sort_index()
ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color="#1D4ED8")
ax.set_title("Cluster Size Distribution")
ax.set_xlabel("Cluster")
ax.set_ylabel("Records")
add_chart("Tumor Cluster Size Distribution", fig, f"KMeans grouped records into <b>{k}</b> tumor-feature clusters.")

if len(cluster_features) >= 2:
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.scatter(Xc.iloc[:, 0], Xc.iloc[:, 1], c=df["cluster"], alpha=0.45)
    ax.set_title("Cluster Separation Using Two Strong Features")
    ax.set_xlabel(cluster_features[0])
    ax.set_ylabel(cluster_features[1])
    add_chart("Two-Feature Cluster Separation", fig, f"The chart shows cluster structure using <b>{E(cluster_features[0])}</b> and <b>{E(cluster_features[1])}</b>.")

contamination = min(0.05, max(0.01, 80 / len(df)))
iso = IsolationForest(contamination=contamination, random_state=42)
df["anomaly_flag"] = iso.fit_predict(Xcs)
anomaly_count = int((df["anomaly_flag"] == -1).sum())

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(["Regular", "Anomalous"], [len(df) - anomaly_count, anomaly_count], color=["#16A34A", "#DC2626"])
ax.set_title("Isolation Forest Anomaly Detection")
ax.set_ylabel("Records")
add_chart("Anomaly Detection Summary", fig, f"Isolation Forest flagged <b>{fmt_int(anomaly_count)}</b> unusual records based on top diagnostic features.")

anomaly_rows = df.loc[df["anomaly_flag"] == -1, ["diagnosis"] + cluster_features[:6]].head(15).values.tolist()
add_table("Sample Potentially Anomalous Records", anomaly_rows, ["Diagnosis"] + cluster_features[:6], None, 15)

elements.append(PageBreak())
add_heading("Final Analytical Summary")

final_lines = [
    f"The dataset contains <b>{fmt_int(len(df))}</b> valid records with a malignant diagnosis rate of <b>{fmt_pct(malignant_rate)}</b>.",
    f"The strongest feature association with malignancy is <b>{E(target_corr.index[0])}</b> with correlation <b>{fmt_num(target_corr.iloc[0], 4)}</b>.",
    f"The best classification model is <b>{E(best_name)}</b> with ROC AUC <b>{fmt_num(best_auc, 4)}</b>.",
    f"The clustering section separates tumor profiles into <b>{k}</b> groups and reports malignancy concentration by cluster.",
    f"The anomaly-detection section flags <b>{fmt_int(anomaly_count)}</b> records for closer review.",
    f"The report generated <b>{fmt_int(chart_no)}</b> charts and saved all outputs in <b>{E(OUT)}</b>."
]
add_body("<br/><br/>".join(final_lines))

doc.build(elements)

df.to_csv(os.path.join(OUT, "processed_breast_cancer_dataset.csv"), index=False)

print("Input File:", data_path)
print("Rows Processed:", len(df))
print("Charts Generated:", chart_no)
print("PDF Report Generated:", report_path)
print("Processed CSV Generated:", os.path.join(OUT, "processed_breast_cancer_dataset.csv"))
print("Execution Time:", round(time.time() - start, 2), "seconds")