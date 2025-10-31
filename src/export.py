"""
Export Module for CATIA
Exports analysis results in multiple formats (JSON, CSV, HTML).
"""

import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGGING_CONFIG, OUTPUT_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


class ReportExporter:
    """Export analysis results in multiple formats."""
    
    def __init__(self, analysis_results: Dict[str, Any], output_dir: str = None):
        """
        Initialize report exporter.
        
        Args:
            analysis_results: Dictionary with analysis results
            output_dir: Output directory for exports (uses OUTPUT_CONFIG if None)
        """
        self.results = analysis_results
        self.output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        self.timestamp = datetime.now().isoformat()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ReportExporter initialized. Output dir: {self.output_dir}")
    
    def export_json(self, filename: str = None) -> str:
        """
        Export results to JSON format.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"catia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        export_data = {
            'timestamp': self.timestamp,
            'metrics': self.results,
            'version': '1.0'
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"✓ JSON export: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"✗ JSON export failed: {e}")
            raise
    
    def export_csv(self, filename: str = None) -> str:
        """
        Export results to CSV format.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"catia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Flatten nested dictionaries for CSV
            flat_results = self._flatten_dict(self.results)
            df = pd.DataFrame([flat_results])
            df.to_csv(filepath, index=False)
            logger.info(f"✓ CSV export: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"✗ CSV export failed: {e}")
            raise
    
    def export_html_summary(self, filename: str = None) -> str:
        """
        Export results to HTML summary format.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"catia_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            html = self._generate_html()
            with open(filepath, 'w') as f:
                f.write(html)
            logger.info(f"✓ HTML export: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"✗ HTML export failed: {e}")
            raise
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _generate_html(self) -> str:
        """
        Generate HTML report.
        
        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>CATIA Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                    border-left: 4px solid #007bff;
                    padding-left: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background-color: #007bff;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                tr:hover {{
                    background-color: #f9f9f9;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 15px;
                    background-color: #f0f0f0;
                    border-radius: 4px;
                    border-left: 4px solid #007bff;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 5px;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #999;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CATIA Analysis Report</h1>
                <p><strong>Generated:</strong> {self.timestamp}</p>
                
                <h2>Key Metrics</h2>
                <div>
        """
        
        # Add metrics
        for key, value in self.results.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = f"{value:,}"
                html += f"""
                    <div class="metric">
                        <div class="metric-value">{formatted_value}</div>
                        <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    </div>
                """
        
        html += """
                </div>
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        # Add detailed results
        for key, value in self.results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    html += f"""
                    <tr>
                        <td>{key} - {sub_key}</td>
                        <td>{sub_value}</td>
                    </tr>
                    """
            else:
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """
        
        html += """
                </table>
                
                <div class="footer">
                    <p>CATIA: Catastrophe AI System for Climate Risk Modeling</p>
                    <p>This report was automatically generated by CATIA.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def export_all(self) -> Dict[str, str]:
        """
        Export results in all formats.
        
        Returns:
            Dictionary with paths to all exported files
        """
        logger.info("Exporting results in all formats...")
        
        return {
            'json': self.export_json(),
            'csv': self.export_csv(),
            'html': self.export_html_summary()
        }

