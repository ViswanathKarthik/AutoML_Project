import json
import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from sklearn.metrics import accuracy_score, r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in os.sys.path:
    os.sys.path.insert(0, SRC_ROOT)

from modules.preprocessing import DataPreprocessor
from modules.feature_selection import FeatureSelector
from modules.model_training import ModelTrainer
from modules.reporting import ReportGenerator
from modules.utils import detect_problem_type
import config


def slugify(name):
    return ''.join(c.lower() if c.isalnum() else '_' for c in name)


def make_session_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_session_dir(session_id):
    return os.path.join(settings.MEDIA_ROOT, 'web', 'sessions', session_id)


def save_artifact(session_dir, artifact):
    with open(os.path.join(session_dir, 'artifact.pkl'), 'wb') as f:
        pickle.dump(artifact, f)


def load_artifact(session_id):
    path = os.path.join(get_session_dir(session_id), 'artifact.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Session artifact not found: {session_id}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def url_for(path):
    rel = os.path.relpath(path, settings.MEDIA_ROOT).replace(os.sep, '/')
    return f"{settings.MEDIA_URL.rstrip('/')}/{rel}"


def build_histograms(X, max_features=None):
    histograms = {}
    cols = X.columns.tolist()

    if max_features is not None:
        cols = cols[:max_features]

    for col in cols:
        values = X[col].dropna()
        if len(values) == 0:
            continue

        if pd.api.types.is_numeric_dtype(values):
            counts, edges = np.histogram(values, bins=10)
            histograms[col] = {
                'type': 'numeric',
                'bins': [float(x) for x in edges.tolist()],
                'counts': [int(x) for x in counts.tolist()]
            }
        else:
            counts = values.astype(str).value_counts().sort_index()
            histograms[col] = {
                'type': 'categorical',
                'bins': [str(x) for x in counts.index.tolist()],
                'counts': [int(x) for x in counts.tolist()]
            }

    return histograms


@csrf_exempt
def health_check(request):
    return JsonResponse({'status': 'ok'})


@csrf_exempt
def preview_csv(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    uploaded_file = request.FILES.get('file')

    if not uploaded_file:
        return JsonResponse({'error': 'Please upload a CSV file.'}, status=400)

    if not uploaded_file.name.endswith('.csv'):
        return JsonResponse({'error': 'Only CSV files are supported.'}, status=400)

    try:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        preview = df.head(10).fillna('').to_dict(orient='records')
        histograms = build_histograms(df)

        return JsonResponse({
            'status': 'success',
            'columns': columns,
            'preview': preview,
            'histograms': histograms,
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


@csrf_exempt
def run_pipeline(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    uploaded_file = request.FILES.get('file')
    target = request.POST.get('target', '').strip()

    if not uploaded_file or not target:
        return JsonResponse({'error': 'Please upload a CSV file and provide a target column.'}, status=400)

    if not uploaded_file.name.endswith('.csv'):
        return JsonResponse({'error': 'Only CSV files are supported.'}, status=400)

    session_id = make_session_id()
    session_dir = get_session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    try:
        upload_path = default_storage.save(f'uploads/{uploaded_file.name}', ContentFile(uploaded_file.read()))
        full_path = default_storage.path(upload_path)

        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(full_path)
        X, y = preprocessor.preprocess(df, target)

        problem_type = detect_problem_type(y)
        selector = FeatureSelector(problem_type=problem_type)
        X_selected, selection_log = selector.select_features(X, y)

        trainer = ModelTrainer(problem_type=problem_type)
        X_train, X_test, y_train, y_test = trainer.split_data(X_selected, y)
        trainer.train_models(X_train, y_train)
        trainer.evaluate_models(X_test, y_test)
        best_model_name, best_model, best_metrics, _ = trainer.get_best_model()

        model_files = {}
        for model_name, model in trainer.models.items():
            filename = f"{slugify(model_name)}.pkl"
            file_path = os.path.join(session_dir, filename)
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            model_files[model_name] = url_for(file_path)

        feature_importance_df = trainer.get_feature_importance_df()
        feature_importance = []
        if feature_importance_df is not None:
            feature_importance = feature_importance_df.to_dict(orient='records')

        histograms = build_histograms(X_selected)

        artifact = {
            'preprocessor': preprocessor,
            'selector': selector,
            'models': trainer.models,
            'model_files': model_files,
            'selected_features': X_selected.columns.tolist(),
            'target': target,
            'problem_type': problem_type,
            'best_model_name': best_model_name,
        }
        save_artifact(session_dir, artifact)

        report_root = os.path.join(settings.MEDIA_ROOT, 'web', 'sessions', session_id)
        os.makedirs(report_root, exist_ok=True)
        reporter = ReportGenerator(output_dir=report_root)
        json_file = reporter.generate_json_report(trainer.results)
        csv_file = reporter.generate_csv_report(trainer.results)

        dataset_summary = {
            'Rows': len(df),
            'Columns': len(df.columns),
            'Target': target,
            'Problem Type': problem_type,
            'Features Selected': X_selected.shape[1],
            'Training Samples': len(X_train),
            'Test Samples': len(X_test),
            'Best Model': best_model_name,
        }

        try:
            correlation_matrix = X_selected.corr(numeric_only=True)
        except Exception:
            correlation_matrix = X_selected.corr()

        html_file = reporter.generate_html_report(
            trainer.results,
            {
                'Problem Type': problem_type,
                'Best Model': best_model_name,
                'Features Selected': X_selected.shape[1],
                'Rows': len(df),
                'Target': target,
            },
            selected_features=X_selected.columns.tolist(),
            feature_importance=feature_importance,
            histograms=histograms,
            preprocessing_summary=preprocessor.preprocessing_summary,
            selection_log=getattr(selector, 'selection_log', {}),
            correlation_matrix=correlation_matrix,
            dataset_summary=dataset_summary,
        )

        return JsonResponse({
            'status': 'success',
            'session_id': session_id,
            'problem_type': problem_type,
            'best_model': best_model_name,
            'metrics': best_metrics,
            'results': trainer.results,
            'selected_features': X_selected.columns.tolist(),
            'feature_importance': feature_importance,
            'histograms': histograms,
            'pickle_files': model_files,
            'files': {
                'json': url_for(json_file),
                'csv': url_for(csv_file),
                'html': url_for(html_file),
            },
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


@csrf_exempt
def predict_test(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    uploaded_file = request.FILES.get('file')
    session_id = request.POST.get('session_id', '').strip()

    if not uploaded_file or not session_id:
        return JsonResponse({'error': 'Please upload a test CSV file and provide a session id.'}, status=400)

    if not uploaded_file.name.endswith('.csv'):
        return JsonResponse({'error': 'Only CSV files are supported.'}, status=400)

    try:
        artifact = load_artifact(session_id)
        df = pd.read_csv(uploaded_file)

        target = artifact.get('target')
        problem_type = artifact.get('problem_type')
        best_model_name = artifact.get('best_model_name')
        best_model = artifact['models'][best_model_name]

        actual = None
        if target in df.columns:
            actual = df[target].copy()
            X_test = df.drop(columns=[target])
        else:
            X_test = df.copy()

        preprocessor = artifact['preprocessor']
        selector = artifact['selector']

        X_test = preprocessor.handle_missing(X_test)
        X_test = preprocessor.handle_outliers(X_test)
        X_test = selector.transform(X_test)

        predictions = best_model.predict(X_test)
        prediction_list = [pred for pred in predictions]

        table = []
        for idx, pred in enumerate(prediction_list):
            row = {
                'index': idx,
                'prediction': pred,
            }
            if actual is not None:
                row['actual'] = actual.iloc[idx]
            table.append(row)

        score = None
        if actual is not None:
            if problem_type == 'classification':
                score = float(accuracy_score(actual, predictions))
            else:
                score = float(r2_score(actual, predictions))

        return JsonResponse({
            'status': 'success',
            'session_id': session_id,
            'model': best_model_name,
            'predictions': table,
            'columns': ['index', 'prediction'] + (['actual'] if actual is not None else []),
            'score': score,
            'has_actual': actual is not None,
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)
