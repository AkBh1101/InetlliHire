"""
IntelliHiree web service (fixed datasets, no upload)
"""

import logging
import os
import traceback
import warnings

warnings.filterwarnings('ignore')

from flask import Flask, jsonify, render_template, request, send_file  # type: ignore

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
_log = logging.getLogger(__name__)

# ── App bootstrap ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hr_talentscope_2024'

# ── Shared pipeline state ──────────────────────────────────────────────────────
_ctx = {
    'train_data':       None,
    'test_data':        None,
    'preprocessor':     None,
    'processed_train':  None,
    'models':           {},
    'results':          {},
    'feature_names':    [],
    'steps': {
        'data_loaded':        False,
        'eda_done':           False,
        'preprocessing_done': False,
        'training_done':      False,
        'prediction_done':    False,
    },
}


# ── Guard helpers ──────────────────────────────────────────────────────────────

def _require(condition: bool, message: str):
    """Return a 400 error response tuple when *condition* is False."""
    if not condition:
        return jsonify({'success': False, 'error': message}), 400
    return None


def _abort500(exc: Exception):
    _log.error(traceback.format_exc())
    return jsonify({'success': False, 'error': str(exc)}), 500


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load-data', methods=['POST'])
def load_data():
    try:
        from pipeline.data_loader import load_and_profile
        train_df, train_profile = load_and_profile(os.path.join(_ROOT, 'aug_train.csv'))
        test_df,  test_profile  = load_and_profile(os.path.join(_ROOT, 'aug_test.csv'), is_test=True)

        _ctx['train_data'] = train_df
        _ctx['test_data']  = test_df
        _ctx['steps']['data_loaded'] = True

        return jsonify({
            'success': True,
            'train':   train_profile,
            'test':    test_profile,
            'message': f'{len(train_df):,} train rows · {len(test_df):,} test rows loaded',
        })
    except Exception as exc:
        return _abort500(exc)


@app.route('/api/eda', methods=['POST'])
def run_eda():
    try:
        guard = _require(_ctx['train_data'] is not None, 'Load data first')
        if guard:
            return guard

        from pipeline.eda import perform_eda
        result = perform_eda(_ctx['train_data'])
        _ctx['steps']['eda_done'] = True
        return jsonify({'success': True, **result})
    except Exception as exc:
        return _abort500(exc)


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    try:
        guard = _require(_ctx['train_data'] is not None, 'Load data first')
        if guard:
            return guard

        from pipeline.preprocessor import Preprocessor
        prep = Preprocessor()
        X, y, summary, distributions, boxplots = prep.fit_transform(_ctx['train_data'])

        _ctx['preprocessor']    = prep
        _ctx['processed_train'] = (X, y)
        _ctx['feature_names']   = prep.feature_names
        _ctx['steps']['preprocessing_done'] = True

        return jsonify({
            'success':       True,
            'summary':       summary,
            'distributions': distributions,
            'boxplots':      boxplots,
        })
    except Exception as exc:
        return _abort500(exc)


@app.route('/api/train', methods=['POST'])
def train_models():
    try:
        guard = _require(_ctx['steps']['preprocessing_done'], 'Run preprocessing first')
        if guard:
            return guard

        from pipeline.trainer import ModelTrainer
        X, y = _ctx['processed_train']

        trainer = ModelTrainer()
        trainer.feature_names = _ctx['feature_names']
        results = trainer.train_all(X, y)

        _ctx['models']  = trainer.models
        _ctx['results'] = results
        _ctx['steps']['training_done'] = True

        _CURVE_KEYS = {'roc_fpr', 'roc_tpr', 'pr_prec', 'pr_rec',
                       'confusion_matrix', 'feature_importances'}

        out = {
            k: {mk: mv for mk, mv in v.items() if mk not in _CURVE_KEYS}
            for k, v in results.items()
            if not k.startswith('_')
        }
        out['_charts']        = results.get('_charts', {})
        out['_meta']          = results.get('_meta', {})
        out['_best_params']   = {
            m: results[m].get('best_params', {})
            for m in results if not m.startswith('_')
        }
        out['_interpretability'] = results.get('_interpretability', {})

        return jsonify({'success': True, 'results': out})
    except Exception as exc:
        return _abort500(exc)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        guard = _require(_ctx['steps']['training_done'], 'Train models first')
        if guard:
            return guard
        guard = _require(_ctx['test_data'] is not None, 'Test data not loaded')
        if guard:
            return guard

        from pipeline.predictor import generate_submission
        body       = request.get_json(silent=True) or {}
        sub, meta  = generate_submission(
            _ctx['test_data'], _ctx['preprocessor'],
            _ctx['models'],    _ctx['results'],
            body.get('model', 'best'),
        )

        out_path = os.path.join(_ROOT, 'outputs', 'submission.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sub.to_csv(out_path, index=False)
        _ctx['steps']['prediction_done'] = True

        return jsonify({
            'success': True,
            'meta':    meta,
            'preview': sub.head(10).to_dict(orient='records'),
        })
    except Exception as exc:
        return _abort500(exc)


@app.route('/api/download-submission')
def download_submission():
    path = os.path.join(_ROOT, 'outputs', 'submission.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'No submission yet'}), 404
    return send_file(path, as_attachment=True, download_name='submission.csv')


@app.route('/api/status')
def status():
    return jsonify(_ctx['steps'])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(os.path.join(_ROOT, 'outputs'), exist_ok=True)
    app.run(host='0.0.0.0', port=5050)
