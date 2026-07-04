"""Main CLI interface for AutoML project"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.preprocessing import DataPreprocessor
from modules.feature_selection import FeatureSelector
from modules.model_training import ModelTrainer
from modules.reporting import ReportGenerator
from modules.utils import detect_problem_type, print_section, print_subsection
import config


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='AutoML Project - Automated Machine Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data.csv --target target_column
  python main.py -i data.csv -t price --mi-threshold 0.05
  python main.py -i data.csv -t target --correlation 0.85
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '-t', '--target',
        type=str,
        required=True,
        help='Name of target variable column'
    )
    
    parser.add_argument(
        '--mi-threshold',
        type=float,
        default=config.MI_THRESHOLD,
        help=f'Mutual Information threshold for feature selection (default: {config.MI_THRESHOLD})'
    )
    
    parser.add_argument(
        '--correlation',
        type=float,
        default=config.CORRELATION_THRESHOLD,
        help=f'Correlation threshold for dropping features (default: {config.CORRELATION_THRESHOLD})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for reports (default: output)'
    )
    
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['json', 'csv', 'html'],
        choices=['json', 'csv', 'html', 'pdf'],
        help='Output report formats (default: json csv html)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f'✗ Error: Input file "{args.input}" not found')
        sys.exit(1)
    
    print_section('🤖 AutoML Project - Automated Machine Learning Pipeline')
    
    try:
        # ==================== DATA LOADING ====================
        print_section('PHASE 1: DATA LOADING & PREPROCESSING')
        
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(args.input)
        
        # ==================== PREPROCESSING ====================
        X, y = preprocessor.preprocess(df, args.target)
        
        # ==================== PROBLEM TYPE DETECTION ====================
        problem_type = detect_problem_type(y)
        print_subsection('Problem Type Detection')
        print(f'✓ Detected as: {problem_type.upper()}')
        
        # ==================== FEATURE SELECTION ====================
        print_section('PHASE 2: FEATURE SELECTION')
        
        selector = FeatureSelector(
            mi_threshold=args.mi_threshold,
            correlation_threshold=args.correlation,
            problem_type=problem_type
        )
        
        X_selected, selection_log = selector.select_features(X, y)
        
        # ==================== MODEL TRAINING ====================
        print_section('PHASE 3: MODEL TRAINING & EVALUATION')
        
        trainer = ModelTrainer(problem_type=problem_type)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X_selected, y)
        trainer.train_models(X_train, y_train)
        trainer.evaluate_models(X_test, y_test)
        best_model_name, best_model, best_metrics = trainer.get_best_model()
        cv_results = trainer.cross_validate_best_model(X_selected, y)
        
        # ==================== REPORT GENERATION ====================
        print_section('PHASE 4: REPORT GENERATION')
        
        reporter = ReportGenerator(output_dir=args.output)
        
        model_info = {
            'Problem Type': problem_type,
            'Best Model': best_model_name,
            'Total Samples': len(df),
            'Total Features (Initial)': len(df.columns) - 1,
            'Total Features (Final)': X_selected.shape[1],
            'Training Samples': len(X_train),
            'Test Samples': len(X_test),
            'MI Threshold': args.mi_threshold,
            'Correlation Threshold': args.correlation,
            'CV Folds': config.CV_FOLDS
        }
        
        # Generate reports in requested formats
        generated_files = []
        
        if 'json' in args.formats:
            json_file = reporter.generate_json_report(trainer.results)
            generated_files.append(json_file)
        
        if 'csv' in args.formats:
            csv_file = reporter.generate_csv_report(trainer.results)
            generated_files.append(csv_file)
        
        if 'html' in args.formats:
            html_file = reporter.generate_html_report(trainer.results, model_info)
            generated_files.append(html_file)
        
        if 'pdf' in args.formats:
            if 'html' in args.formats:
                # Use the HTML file we just created
                html_file = [f for f in generated_files if f.endswith('.html')][0]
                pdf_file = reporter.generate_pdf_report(html_file)
                if pdf_file:
                    generated_files.append(pdf_file)
            else:
                # Generate temporary HTML for PDF
                html_file = reporter.generate_html_report(trainer.results, model_info)
                pdf_file = reporter.generate_pdf_report(html_file)
                if pdf_file:
                    generated_files.append(pdf_file)
                os.remove(html_file)
        
        # ==================== FINAL SUMMARY ====================
        print_section('COMPLETION SUMMARY')
        
        summary = reporter.generate_summary(trainer.results, best_model_name, best_metrics)
        print(summary)
        
        print(f'\n📁 Generated Files:')
        for file in generated_files:
            print(f'   ✓ {file}')
        
        print(f'\n✅ AutoML Pipeline Complete!')
        
    except KeyboardInterrupt:
        print('\n\n⏹️  Pipeline interrupted by user')
        sys.exit(0)
    except Exception as e:
        print(f'\n✗ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
