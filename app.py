"""
HR TalentScope – Flask Application (fixed datasets, no upload)
"""
import os, logging, warnings, traceback
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, jsonify, send_file # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hr_talentscope_2024'

STATE = {
    'train_data': None, 'test_data': None,
    'preprocessor': None,
    'processed_train': None,
    'models': {}, 'results': {},
    'feature_names': [],
    'pipeline_steps': {
        'data_loaded': False, 'eda_done': False,
        'preprocessing_done': False, 'training_done': False, 'prediction_done': False
    }
}
BASE = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load-data', methods=['POST'])
def load_data():
    try:
        from pipeline.data_loader import load_and_profile
        tdf, tp = load_and_profile(os.path.join(BASE, 'aug_train.csv'))
        xdf, xp = load_and_profile(os.path.join(BASE, 'aug_test.csv'), is_test=True)
        STATE['train_data'] = tdf
        STATE['test_data']  = xdf
        STATE['pipeline_steps']['data_loaded'] = True
        return jsonify({'success': True, 'train': tp, 'test': xp,
                        'message': f'{len(tdf):,} train rows · {len(xdf):,} test rows loaded'})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/eda', methods=['POST'])
def run_eda():
    try:
        if STATE['train_data'] is None:
            return jsonify({'success': False, 'error': 'Load data first'}), 400
        from pipeline.eda import perform_eda
        result = perform_eda(STATE['train_data'])
        STATE['pipeline_steps']['eda_done'] = True
        return jsonify({'success': True, **result})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    try:
        if STATE['train_data'] is None:
            return jsonify({'success': False, 'error': 'Load data first'}), 400
        from pipeline.preprocessor import Preprocessor
        prep = Preprocessor()
        X, y, summary, distributions, boxplots = prep.fit_transform(STATE['train_data'])
        STATE['preprocessor']    = prep
        STATE['processed_train'] = (X, y)
        STATE['feature_names']   = prep.feature_names
        STATE['pipeline_steps']['preprocessing_done'] = True
        return jsonify({
            'success': True,
            'summary': summary,
            'distributions': distributions,
            'boxplots': boxplots
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_models():
    try:
        if not STATE['pipeline_steps']['preprocessing_done']:
            return jsonify({'success': False, 'error': 'Run preprocessing first'}), 400
        from pipeline.trainer import ModelTrainer
        X, y = STATE['processed_train']
        trainer = ModelTrainer()
        trainer.feature_names = STATE['feature_names']
        results = trainer.train_all(X, y)
        STATE['models']  = trainer.models
        STATE['results'] = results
        STATE['pipeline_steps']['training_done'] = True
        out = {k: {mk: mv for mk, mv in v.items()
                   if mk not in ('roc_fpr','roc_tpr','pr_prec','pr_rec',
                                 'confusion_matrix','feature_importances')}
               for k, v in results.items() if not k.startswith('_')}
        out['_charts'] = results.get('_charts', {})
        out['_meta'] = results.get('_meta', {})
        out['_best_params'] = {
            model: results[model].get('best_params', {})
            for model in results if not model.startswith('_')
        }
        out['_interpretability'] = results.get('_interpretability', {})
        return jsonify({'success': True, 'results': out})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not STATE['pipeline_steps']['training_done']:
            return jsonify({'success': False, 'error': 'Train models first'}), 400
        if STATE['test_data'] is None:
            return jsonify({'success': False, 'error': 'Test data not loaded'}), 400
        from pipeline.predictor import generate_submission
        body = request.get_json(silent=True) or {}
        sub, meta = generate_submission(
            STATE['test_data'], STATE['preprocessor'],
            STATE['models'], STATE['results'], body.get('model', 'best'))
        out_path = os.path.join(BASE, 'outputs', 'submission.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sub.to_csv(out_path, index=False)
        STATE['pipeline_steps']['prediction_done'] = True
        return jsonify({'success': True, 'meta': meta,
                        'preview': sub.head(10).to_dict(orient='records')})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-submission')
def download_submission():
    p = os.path.join(BASE, 'outputs', 'submission.csv')
    if not os.path.exists(p):
        return jsonify({'error': 'No submission yet'}), 404
    return send_file(p, as_attachment=True, download_name='submission.csv')

@app.route('/api/status')
def status():
    return jsonify(STATE['pipeline_steps'])

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE, 'outputs'), exist_ok=True)
    app.run(host="0.0.0.0", port=5050)
