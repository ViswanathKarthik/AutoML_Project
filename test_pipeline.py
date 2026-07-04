"""Quick test script to verify the AutoML pipeline"""

import os
import pickle
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import io
import contextlib

from modules.preprocessing import DataPreprocessor
from modules.feature_selection import FeatureSelector
from modules.model_training import ModelTrainer
from modules.utils import detect_problem_type, print_section
from modules.report_generator import AutoMLReport


if __name__ == '__main__':
    log_buffer = io.StringIO()

    with contextlib.redirect_stdout(log_buffer):
        print_section('Testing AutoML Pipeline')
        
        # Test with sample data
        data_path = 'data/train_sample_titanic.csv'
        target = 'Transported'
        
        if not os.path.exists(data_path):
            print(f'✗ Sample data not found at {data_path}')
            sys.exit(1)
        
        # Phase 1: Preprocessing
        print('\n[1/4] Preprocessing...')
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        X, y = preprocessor.preprocess(df, target)
        
        X_preprocessed = X.copy()
        X_preprocessed.to_csv('output/preprocessed_data.csv', index=False)  # Save preprocessed data for report
        
        # Phase 2: Detect problem type
        print('\n[2/4] Detecting problem type...')
        problem_type = detect_problem_type(y)
        print(f'✓ Problem Type: {problem_type}')
        
        # Phase 3: Feature Selection
        print('\n[3/4] Selecting features...')
        selector = FeatureSelector(problem_type=problem_type)
        X_selected, log = selector.select_features(X, y)
        print(f'✓ Features retained: {X_selected.shape[1]}/{X.shape[1]}')
        
        # Phase 4: Model Training
        print('\n[4/4] Training models...')
        trainer = ModelTrainer(problem_type=problem_type)
        X_train, X_test, y_train, y_test = trainer.split_data(X_selected, y)
        trainer.train_models(X_train, y_train)
        trainer.evaluate_models(X_test, y_test)
        best_model, best_model_obj, best_metrics,importance_df = trainer.get_best_model()
        # ---- SAVE ARTIFACT ----
        artifact = {
        "preprocessing_params": {
            "means": preprocessor.means,
            "fill_values": preprocessor.fill_values,
            "outlier_bounds": preprocessor.outlier_bounds,
            "drop_cols": preprocessor.drop_cols
        },
        "selected_features": selector.selected_features,
        "encoders": selector.encoders,   # ✅ use THIS
        "models": trainer.models,
        "features": X_selected.columns.tolist(),

    }

        with open("automl_full.pkl", "wb") as f:
            pickle.dump(artifact, f)

        print("✅ Model + pipeline saved as automl_full.pkl")
        
        print(f'\n✅ Test completed successfully!')
        print(f'✓ Best Model: {best_model}')
        print(f'✓ Sample Metric: {list(best_metrics.values())[0]:.4f}')
        print("\n[5/5] Generating AutoML report...")
    pipeline_logs = log_buffer.getvalue()
    report = AutoMLReport()

    report.generate_pdf(
        filepath="output/automl_full_report.pdf",
        dataset=df,
        target=target,
        models=trainer.models,
        results=trainer.results,
        best_model=trainer.best_model_name,
        preprocessing_log=preprocessor.preprocessing_summary,
        feature_selection_log=selector.selection_log,
        feature_importance=trainer.importance_df,
        X_preprocessed=X_preprocessed,
        X_test=X_test,
        y_test=y_test,
        pipeline_logs=pipeline_logs
    )