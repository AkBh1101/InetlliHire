# ⚡ HR TalentScope — ML Analytics Platform

An end-to-end Machine Learning platform for HR job-change prediction.
Train 5 models, explore data visually, and generate competition submissions.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open in browser
http://localhost:5050
```

---

## 📁 Project Structure

```
hr_analytics/
├── app.py                  ← Flask application & API routes
├── requirements.txt
├── aug_train.csv           ← Built-in training data (19,158 rows)
├── aug_test.csv            ← Built-in test data (2,129 rows)
├── sample_submission.csv   ← Submission format reference
├── pipeline/
│   ├── data_loader.py      ← CSV loading & profiling
│   ├── eda.py              ← 10 EDA chart generators
│   ├── preprocessor.py     ← Feature engineering & scaling
│   ├── trainer.py          ← 5 ML models + evaluation charts
│   └── predictor.py        ← Inference & submission file
├── templates/
│   └── index.html          ← Full SPA dashboard
├── uploads/                ← Uploaded CSVs land here
└── outputs/                ← submission.csv saved here
```

---

## 🎯 5-Step Pipeline

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Data Loading** | Built-in or drag-and-drop CSV upload |
| 2 | **EDA** | 10 interactive chart categories |
| 3 | **Preprocessing** | Missing values, encoding, 4 new features |
| 4 | **Training** | LR · DT · RF · SVM · KNN + CV metrics |
| 5 | **Predict** | Submission CSV download |

---

## 🤖 Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable, max_depth=8 |
| Random Forest | 50 trees, balanced class weights |
| SVM | RBF kernel, trained on 4K sample |
| KNN | k=7, trained on 4K sample |

> SVM and KNN use a 4,000-row stratified subsample for speed.
> All models use `class_weight='balanced'` to handle the ~25% positive class.

---

## 📊 EDA Charts

- Target class distribution (bar + donut)
- Missing value analysis
- Experience distribution
- Education level breakdown
- Company size & type
- Gender vs. job-change rate
- Training hours distribution + box plot
- City Development Index analysis
- Feature correlation matrix
- Feature vs. target seek-rate grid

---

## 📤 Submission Format

```
enrollee_id,target
32403,0.22
9858,0.42
...
```
`target` is the predicted probability of job-seeking (0–1).

---

## 🔌 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/load-builtin` | Load bundled train + test CSVs |
| POST | `/api/upload-train` | Upload custom training CSV |
| POST | `/api/upload-test` | Upload custom test CSV |
| POST | `/api/eda` | Run EDA, return stats + 10 charts |
| POST | `/api/preprocess` | Feature engineering pipeline |
| POST | `/api/train` | Train all 5 models |
| POST | `/api/predict` | Generate predictions |
| GET  | `/api/download-submission` | Download submission.csv |
| GET  | `/api/status` | Pipeline step completion status |
| GET  | `/api/model-comparison` | Model metrics JSON |
