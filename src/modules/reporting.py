"""Report generation module"""

import pandas as pd
import json
import os
from typing import Dict, List
from datetime import datetime
from .utils import print_subsection


class ReportGenerator:
    """Generate reports in multiple formats"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.reports_dir = os.path.join(output_dir, 'reports')
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def generate_json_report(self, results: Dict, filename: str = None) -> str:
        """Generate JSON report"""
        print_subsection('Generating JSON Report')
        
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {
                metric: float(value) for metric, value in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f' JSON report saved: {filepath}')
        return filepath
    
    def generate_csv_report(self, results: Dict, filename: str = None) -> str:
        """Generate CSV report"""
        print_subsection('Generating CSV Report')
        
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results).T
        df.index.name = 'Model'
        df.to_csv(filepath)
        
        print(f' CSV report saved: {filepath}')
        return filepath
    
    def generate_html_report(self, results: Dict, model_info: Dict = None, filename: str = None) -> str:
        """Generate HTML report"""
        print_subsection('Generating HTML Report')
        
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoML Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #4CAF50;
                    margin-top: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .highlight {{
                    background-color: #fff3cd;
                    font-weight: bold;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                }}
                .info {{
                    background-color: #e7f3fe;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1> AutoML Project Report</h1>
                <div class="info">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
        """
        
        # Add model information
        if model_info:
            html_content += """
                <h2>Model Information</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
            """
            for key, value in model_info.items():
                html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
                """
            html_content += """
                </table>
            """
        
        # Add results table
        html_content += """
            <h2>Model Performance Metrics</h2>
            <table>
                <tr>
                    <th>Model</th>
        """
        
        # Get all unique metrics
        all_metrics = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        
        for metric in sorted(all_metrics):
            html_content += f"<th>{metric}</th>"
        
        html_content += "</tr>"
        
        # Add model rows
        for model_name, metrics in results.items():
            html_content += f"<tr><td><strong>{model_name}</strong></td>"
            for metric in sorted(all_metrics):
                value = metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    value = f"{value:.4f}"
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += """
            </table>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f'HTML report saved: {filepath}')
        return filepath
    
    def generate_pdf_report(self, html_filepath: str = None, filename: str = None) -> str:
        """Generate PDF report from HTML"""
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
        """Generate text summary"""
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
