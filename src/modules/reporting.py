"""Report generation module"""

import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .utils import print_subsection


class ReportGenerator:
    """Generate reports in multiple formats."""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.reports_dir = os.path.join(output_dir, 'reports')
        self.metrics_dir = os.path.join(output_dir, 'metrics')

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _escape(self, value):
        if value is None:
            return 'N/A'
        text = str(value)
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _format_metric(self, value):
        if isinstance(value, (int, float)):
            return f"{float(value):.4f}"
        return self._escape(value)

    def _render_score_bars(self, results: Dict) -> str:
        if not results:
            return '<p>No model results available.</p>'

        max_score = 1.0
        for metrics in results.values():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key.lower() in {'accuracy', 'precision', 'recall', 'f1-score', 'roc-auc', 'r² score', 'r2 score'}:
                    max_score = max(max_score, float(value))

        bars = []
        for model_name, metrics in results.items():
            primary_metric = 'F1-Score' if 'F1-Score' in metrics else ('R² Score' if 'R² Score' in metrics else 'Accuracy' if 'Accuracy' in metrics else sorted(metrics.keys(), key=lambda x: x.lower())[-1])
            score = float(metrics.get(primary_metric, metrics.get('Accuracy', 0) if 'Accuracy' in metrics else 0))
            width = max(6, (score / max_score) * 100) if max_score else 0
            bars.append(
                f"<div class='bar-row'><div class='bar-label'>{self._escape(model_name)}</div><div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div><div class='bar-value'>{self._format_metric(score)}</div></div>"
            )
        return ''.join(bars)

    def _render_svg_bar_chart(self, labels: List[str], values: List[float], width: int = 760, height: int = 220) -> str:
        if not labels or not values:
            return '<p>No chart data available.</p>'

        cleaned_labels = [self._escape(str(label)) for label in labels]
        numeric_values = [float(value) for value in values]
        max_value = max(numeric_values) if numeric_values else 1
        left = 50
        right = 20
        top = 18
        bottom = 46
        plot_width = width - left - right
        plot_height = height - top - bottom
        step = plot_width / max(len(cleaned_labels), 1)
        bar_width = max(18, min(48, step * 0.7))
        chart_lines = []

        for i in range(5):
            y = top + (plot_height * i / 4)
            chart_lines.append(f"<line x1='{left}' y1='{y}' x2='{width - right}' y2='{y}' stroke='rgba(255,255,255,0.12)' stroke-width='1' />")

        bars = []
        for idx, (label, value) in enumerate(zip(cleaned_labels, numeric_values)):
            bar_height = 0 if max_value == 0 else (value / max_value) * plot_height
            x = left + idx * step + (step - bar_width) / 2
            y = top + plot_height - bar_height
            bars.append(
                f"<g><rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' rx='6' fill='rgba(139, 156, 255, 0.95)' />"
                f"<text x='{x + bar_width / 2:.2f}' y='{height - 18}' text-anchor='middle' fill='#c8d2ef' font-size='10'>{label[:10]}</text>"
                f"<text x='{x + bar_width / 2:.2f}' y='{y - 6:.2f}' text-anchor='middle' fill='#e8f0ff' font-size='9'>{value:.1f}</text></g>"
            )

        return (
            f"<svg viewBox='0 0 {width} {height}' width='100%' height='220' role='img' aria-label='Bar chart'>"
            f"{''.join(chart_lines)}{''.join(bars)}</svg>"
        )

    def _render_histograms(self, histograms: Dict) -> str:
        if not histograms:
            return '<p>No pre-run histograms were generated for this dataset.</p>'

        blocks = []
        for feature_name, histogram in histograms.items():
            bins = histogram.get('bins', [])
            counts = histogram.get('counts', [])
            if not bins or not counts:
                continue

            if histogram.get('type') == 'categorical':
                labels = [str(label) for label in bins]
                values = [int(value) for value in counts]
            else:
                labels = []
                values = []
                for idx, count in enumerate(counts[:10]):
                    start = bins[idx]
                    end = bins[idx + 1] if idx + 1 < len(bins) else bins[-1]
                    labels.append(f"{float(start):.1f}-{float(end):.1f}")
                    values.append(int(count))

            blocks.append(f"<div class='histogram-card'><h4>{self._escape(feature_name)}</h4>{self._render_svg_bar_chart(labels, values)}</div>")
        return ''.join(blocks)

    def _render_feature_importance(self, feature_importance: List[Dict]) -> str:
        if not feature_importance:
            return '<p>No feature importance was available for this model.</p>'

        labels = [str(item.get('feature', 'unknown')) for item in feature_importance[:15]]
        values = [float(item.get('importance', 0) or 0) for item in feature_importance[:15]]
        return self._render_svg_bar_chart(labels, values)

    def _render_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> str:
        if correlation_matrix is None or correlation_matrix.empty:
            return '<p>No correlation matrix was generated.</p>'

        feature_names = list(correlation_matrix.columns)
        cells = []
        for row_name in feature_names:
            row_cells = []
            for col_name in feature_names:
                value = float(correlation_matrix.loc[row_name, col_name]) if row_name in correlation_matrix.index and col_name in correlation_matrix.columns else 0.0
                intensity = abs(value)
                color = f"hsl({210 - intensity * 180}, 76%, {55 + (1 - intensity) * 15}%)"
                row_cells.append(f"<div class='corr-cell' style='background:{color}'>{value:.2f}</div>")
            cells.append(f"<div class='corr-row'>{''.join(row_cells)}</div>")

        headers = ''.join(f"<div class='corr-header'>{self._escape(name)}</div>" for name in feature_names)
        return f"<div class='corr-grid'><div class='corr-header-row'><div class='corr-empty'></div>{headers}</div>{''.join(cells)}</div>"

    def _render_preprocessing_summary(self, preprocessing_summary: Dict) -> str:
        if not preprocessing_summary:
            return '<p>No preprocessing steps were recorded.</p>'

        summary_items = []
        for key, value in preprocessing_summary.items():
            if key == 'missing_values' and value:
                rows = [f"<li>{self._escape(feature)}: {details.get('count', 0)} missing values filled using {details.get('strategy', 'default')}</li>" for feature, details in value.items()]
                summary_items.append(f"<div class='summary-block'><h4>Missing value handling</h4><ul>{''.join(rows)}</ul></div>")
            elif key == 'dropped_columns' and value:
                rows = [f"<li>{self._escape(feature)} ({float(percent):.1f}% missing)</li>" for feature, percent in value.items()]
                summary_items.append(f"<div class='summary-block'><h4>Dropped columns</h4><ul>{''.join(rows)}</ul></div>")
            elif key == 'duplicates_removed' and value:
                summary_items.append(f"<div class='summary-block'><h4>Duplicates</h4><p>{int(value)} duplicate rows removed.</p></div>")
            elif key == 'outliers' and value:
                rows = [f"<li>{self._escape(feature)}: clipped {int(details.get('count', 0))} outliers to [{float(details.get('lower', 0)):.3f}, {float(details.get('upper', 0)):.3f}]</li>" for feature, details in value.items()]
                summary_items.append(f"<div class='summary-block'><h4>Outlier handling</h4><ul>{''.join(rows)}</ul></div>")

        return ''.join(summary_items) or '<p>No preprocessing details were detected.</p>'

    def _render_selection_summary(self, selection_log: Dict) -> str:
        if not selection_log:
            return '<p>No feature selection details were recorded.</p>'

        cards = []
        for key in ['initial_features', 'id_removed', 'high_cardinality_removed', 'constant_removed', 'mi_removed', 'correlation_removed', 'model_selected', 'final_features']:
            items = selection_log.get(key, [])
            if not items:
                continue
            safe_items = [self._escape(item) for item in items]
            cards.append(f"<div class='summary-block'><h4>{key.replace('_', ' ').title()}</h4><p>{', '.join(safe_items[:20])}{'...' if len(safe_items) > 20 else ''}</p></div>")

        if not cards:
            return '<p>Feature selection was completed without extra logged removals.</p>'
        return ''.join(cards)

    def _render_model_table(self, results: Dict) -> str:
        all_metrics = sorted({metric for metrics in results.values() for metric in metrics.keys()})
        rows = []
        for model_name, metrics in results.items():
            cells = [f'<td><strong>{self._escape(model_name)}</strong></td>']
            for metric in all_metrics:
                cells.append(f'<td>{self._format_metric(metrics.get(metric, "N/A"))}</td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')
        return ''.join(rows)

    def generate_json_report(self, results: Dict, filename: str = None) -> str:
        """Generate JSON report."""
        print_subsection('Generating JSON Report')

        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.metrics_dir, filename)

        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {metric: float(value) for metric, value in metrics.items()}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)

        print(f' JSON report saved: {filepath}')
        return filepath

    def generate_csv_report(self, results: Dict, filename: str = None) -> str:
        """Generate CSV report."""
        print_subsection('Generating CSV Report')

        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(self.metrics_dir, filename)

        df = pd.DataFrame(results).T
        df.index.name = 'Model'
        df.to_csv(filepath)

        print(f' CSV report saved: {filepath}')
        return filepath

    def generate_html_report(
        self,
        results: Dict,
        model_info: Dict = None,
        filename: str = None,
        selected_features: List[str] = None,
        feature_importance: List[Dict] = None,
        histograms: Dict = None,
        preprocessing_summary: Dict = None,
        selection_log: Dict = None,
        correlation_matrix: pd.DataFrame = None,
        dataset_summary: Dict = None,
    ) -> str:
        """Generate a detailed HTML report for the AutoML session."""
        print_subsection('Generating HTML Report')

        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        filepath = os.path.join(self.reports_dir, filename)
        model_info = model_info or {}

        summary_items = []
        if dataset_summary:
            for key, value in dataset_summary.items():
                summary_items.append(f"<div class='stat-card'><span>{self._escape(key.replace('_', ' ').title())}</span><strong>{self._escape(value)}</strong></div>")
        elif model_info:
            for key, value in model_info.items():
                summary_items.append(f"<div class='stat-card'><span>{self._escape(key)}</span><strong>{self._escape(value)}</strong></div>")

        best_model = model_info.get('Best Model', 'AutoML')
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        feature_list = ', '.join(self._escape(feature) for feature in (selected_features or []))
        all_metrics = sorted({metric for metrics in results.values() for metric in metrics.keys()})

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoML Deep Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: #07111f;
                    color: #eaf2ff;
                    margin: 0;
                    padding: 28px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: #0d1b2a;
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 18px;
                    padding: 28px;
                    box-shadow: 0 20px 40px rgba(2,6,12,0.35);
                }}
                h1 {{ margin: 0 0 12px; font-size: 2.1rem; letter-spacing: -0.04em; }}
                h2 {{ margin: 26px 0 14px; font-size: 1.4rem; color: #cfe1ff; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; }}
                h3 {{ margin: 0 0 12px; font-size: 1.08rem; color: #dfe9ff; }}
                h4 {{ margin: 0 0 10px; font-size: 0.95rem; color: #dfe9ff; }}
                p, li {{ color: #c3cfe8; line-height: 1.7; }}
                .meta {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; color: #a7b9d9; font-size: 0.9rem; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 16px 0 20px; }}
                .stat-card {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; }}
                .stat-card span {{ display: block; color: #9aaecb; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
                .stat-card strong {{ font-size: 1.2rem; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 12px; }}
                .summary-block {{ background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; }}
                .summary-block ul {{ margin: 8px 0 0; padding-left: 18px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: rgba(255,255,255,0.02); }}
                th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }}
                th {{ background: rgba(124,106,244,0.14); color: #dfe7ff; }}
                .bar-row {{ display: grid; grid-template-columns: 170px 1fr 70px; gap: 12px; align-items: center; margin-bottom: 10px; }}
                .bar-label {{ font-size: 0.88rem; color: #dfe9ff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                .bar-track {{ position: relative; height: 12px; background: rgba(255,255,255,0.08); border-radius: 999px; overflow: hidden; }}
                .bar-track.small {{ height: 8px; }}
                .bar-fill {{ position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg, #22c55e, #86efac); border-radius: 999px; }}
                .bar-value {{ font-size: 0.8rem; color: #dfe9ff; text-align: right; }}
                .histogram-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 12px; }}
                .histogram-card {{ background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; }}
                .mini-bar {{ display: grid; grid-template-columns: 90px 1fr 36px; gap: 8px; align-items: center; margin-top: 8px; font-size: 0.72rem; color: #dfe9ff; }}
                .corr-grid {{ margin-top: 12px; display: grid; gap: 6px; }}
                .corr-header-row, .corr-row {{ display: grid; grid-template-columns: 120px repeat(auto-fit, minmax(72px, 1fr)); gap: 6px; }}
                .corr-header, .corr-cell, .corr-empty {{ padding: 8px 6px; text-align: center; border-radius: 6px; font-size: 0.7rem; font-weight: 700; }}
                .corr-header {{ background: rgba(255,255,255,0.08); color: #dfe9ff; }}
                .corr-empty {{ background: transparent; color: #dfe9ff; }}
            </style>
        </head>
        <body>
            <div class='container'>
                <h1>AutoML Project Deep Report</h1>
                <div class='meta'>
                    <span>Generated: {generated_at}</span>
                    <span>Best model: {self._escape(best_model)}</span>
                    <span>Feature importance method: SHAP + Mutual Information</span>
                </div>
                <div class='stats'>
                    {''.join(summary_items)}
                </div>

                <h2>1. Executive summary</h2>
                <p>This report summarizes the complete machine learning lifecycle: data cleaning, feature engineering, feature selection, model training and comparison, evaluation metrics, feature importance interpretation, and distribution analysis. It is intended to be a detailed project-level record of what was done and why.</p>

                <h2>2. Data preprocessing and feature engineering</h2>
                <div class='summary-grid'>
                    {self._render_preprocessing_summary(preprocessing_summary)}
                    {self._render_selection_summary(selection_log)}
                </div>

                <h2>3. Selected features</h2>
                <p>{feature_list if feature_list else 'No selected features were returned by the feature selection stage.'}</p>

                <h2>4. Model performance comparison</h2>
                <div class='summary-block'>
                    {self._render_score_bars(results)}
                </div>
                <table>
                    <tr>
                        <th>Model</th>
                        {''.join(f'<th>{self._escape(metric)}</th>' for metric in all_metrics)}
                    </tr>
                    {self._render_model_table(results)}
                </table>

                <h2>5. Feature importance</h2>
                <div class='summary-block'>
                    {self._render_feature_importance(feature_importance or [])}
                </div>

                <h2>6. Feature distribution and histograms</h2>
                <div class='histogram-grid'>
                    {self._render_histograms(histograms or {})}
                </div>

                <h2>7. Correlation overview</h2>
                {self._render_correlation_matrix(correlation_matrix)}

                <h2>8. Interpretation notes</h2>
                <p>Feature importance was interpreted using SHAP for predictive models and mutual information for upstream feature ranking. This provides a structured explanation of how much each feature contributes to the final decision process.</p>
                <p>Preprocessing included duplicate removal, missing-value treatment, outlier clipping, and filtering of weak or high-cardinality features to keep the modeling pipeline robust and understandable.</p>
            </div>
        </body>
        </html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f'HTML report saved: {filepath}')
        return filepath

    def generate_pdf_report(self, html_filepath: str = None, filename: str = None) -> str:
        """Generate PDF report from HTML."""
        print_subsection('Generating PDF Report')

        try:
            from weasyprint import HTML
        except ImportError:
            print(' WeasyPrint not installed. Install with: pip install weasyprint')
            return None

        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        filepath = os.path.join(self.reports_dir, filename)

        try:
            if html_filepath:
                HTML(html_filepath).write_pdf(filepath)
            else:
                print(' HTML filepath required for PDF generation')
                return None

            print(f'PDF report saved: {filepath}')
            return filepath
        except Exception as e:
            print(f' Error generating PDF: {str(e)}')
            return None

    def generate_summary(self, results: Dict, best_model: str, best_metrics: Dict) -> str:
        """Generate text summary."""
        print_subsection('Summary')

        summary = f"""
╔═══════════════════════════════════════════════════════════╗
║                   AutoML PROJECT SUMMARY                  ║
╚═══════════════════════════════════════════════════════════╝

BEST MODEL: {best_model}
{'─' * 59}
"""

        for metric, value in best_metrics.items():
            summary += f"  {metric}: {value:.4f}\n"

        summary += f"""
{'─' * 59}
ALL MODELS PERFORMANCE:
"""

        for model_name, metrics in results.items():
            summary += f"\n  {model_name}:\n"
            for metric, value in metrics.items():
                summary += f"    - {metric}: {value:.4f}\n"

        summary += f"""
{'═' * 59}
Reports generated in:
  - Reports: output/reports/
  - Metrics: output/metrics/
"""
        
        return summary
