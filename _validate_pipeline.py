"""
run_pipeline.py — End-to-end ML pipeline entry point
Loads data, preprocesses, trains, and generates submission output.
"""

from pipeline.data_loader  import load_and_profile
from pipeline.preprocessor import Preprocessor
from pipeline.trainer      import ModelTrainer
from pipeline.predictor    import generate_submission

# ── Data ingestion ─────────────────────────────────────────────────────────────
train_df, _ = load_and_profile('aug_train.csv')
print('train', len(train_df), train_df.shape[1])

# ── Feature engineering ────────────────────────────────────────────────────────
pre = Preprocessor()
X, y, summary, distributions, boxplots = pre.fit_transform(train_df)
print('processed', X.shape, y.shape)
print('distributions:', len(distributions), 'boxplots:', len(boxplots))

# ── Model training ─────────────────────────────────────────────────────────────
trainer = ModelTrainer()
trainer.feature_names = summary.get('feature_names', [])
res = trainer.train_all(X, y)
print('trained', list(res.keys())[:5])

# ── Inference & submission ─────────────────────────────────────────────────────
test_df, _ = load_and_profile('aug_test.csv', is_test=True)
sub, meta  = generate_submission(test_df, pre, trainer.models, res)
print('submission rows', len(sub))
print('meta model_used', meta.get('model_used'), 'predicted_seekers', meta.get('predicted_seekers'))
