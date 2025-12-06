"""
Risk Alerts Module for CATIA
Provides threshold-based risk monitoring and alerting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from catia.config import LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RiskAlert:
    """Represents a single risk alert."""
    
    def __init__(self, metric: str, current_value: float, threshold: float,
                 severity: AlertSeverity, message: str):
        """
        Initialize a risk alert.
        
        Args:
            metric: Name of the metric that triggered the alert
            current_value: Current value of the metric
            threshold: Threshold that was exceeded
            severity: Alert severity level
            message: Human-readable alert message
        """
        self.metric = metric
        self.current_value = current_value
        self.threshold = threshold
        self.severity = severity
        self.message = message
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"RiskAlert({self.severity.value}: {self.metric})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'metric': self.metric,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }


class RiskAlertSystem:
    """Threshold-based risk monitoring and alerting system."""
    
    def __init__(self, thresholds: Dict[str, float]):
        """
        Initialize the risk alert system.
        
        Args:
            thresholds: Dictionary of metric thresholds
                e.g., {'var_max': 100, 'mean_loss_max': 50, 'tvar_max': 150}
                Values are in millions (will be scaled internally)
        """
        # Scale thresholds to actual values (input is in millions)
        self.thresholds = {k: v * 1e6 for k, v in thresholds.items()}
        self.alerts: List[RiskAlert] = []
        logger.info(f"RiskAlertSystem initialized with {len(thresholds)} thresholds")
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[RiskAlert]:
        """
        Check metrics against thresholds and generate alerts.
        
        Args:
            metrics: Dictionary of current metric values
                e.g., {'var_95': 120000000, 'mean_loss': 45000000, ...}
        
        Returns:
            List of triggered alerts
        """
        self.alerts = []
        
        # Check VaR threshold
        if 'var_95' in metrics and 'var_max' in self.thresholds:
            if metrics['var_95'] > self.thresholds['var_max']:
                self._add_alert(
                    metric='VaR (95%)',
                    current=metrics['var_95'],
                    threshold=self.thresholds['var_max'],
                    severity=AlertSeverity.CRITICAL,
                    msg_template="VaR exceeds maximum threshold"
                )
        
        # Check TVaR threshold
        if 'tvar_95' in metrics and 'tvar_max' in self.thresholds:
            if metrics['tvar_95'] > self.thresholds['tvar_max']:
                self._add_alert(
                    metric='TVaR (95%)',
                    current=metrics['tvar_95'],
                    threshold=self.thresholds['tvar_max'],
                    severity=AlertSeverity.CRITICAL,
                    msg_template="TVaR exceeds maximum threshold"
                )
        
        # Check mean loss threshold
        if 'mean_loss' in metrics and 'mean_loss_max' in self.thresholds:
            if metrics['mean_loss'] > self.thresholds['mean_loss_max']:
                self._add_alert(
                    metric='Mean Loss',
                    current=metrics['mean_loss'],
                    threshold=self.thresholds['mean_loss_max'],
                    severity=AlertSeverity.WARNING,
                    msg_template="Mean loss exceeds expected threshold"
                )
        
        # Check loss ratio threshold
        if 'loss_ratio' in metrics and 'loss_ratio_max' in self.thresholds:
            if metrics['loss_ratio'] > self.thresholds['loss_ratio_max']:
                self._add_alert(
                    metric='Loss Ratio',
                    current=metrics['loss_ratio'],
                    threshold=self.thresholds['loss_ratio_max'],
                    severity=AlertSeverity.WARNING,
                    msg_template="Loss ratio exceeds acceptable range"
                )
        
        logger.info(f"Alert check complete: {len(self.alerts)} alerts triggered")
        return self.alerts
    
    def _add_alert(self, metric: str, current: float, threshold: float,
                   severity: AlertSeverity, msg_template: str):
        """Add an alert to the list."""
        if metric in ['Loss Ratio']:
            message = f"{msg_template}: {current:.2f} > {threshold:.2f}"
        else:
            message = f"{msg_template}: ${current/1e6:.1f}M > ${threshold/1e6:.1f}M"
        
        alert = RiskAlert(
            metric=metric,
            current_value=current,
            threshold=threshold,
            severity=severity,
            message=message
        )
        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert.message}")
    
    def format_alerts(self) -> str:
        """
        Format all alerts as a string for logging/display.
        
        Returns:
            Formatted alert string
        """
        if not self.alerts:
            return "\n  ✓ No risk alerts triggered - all metrics within thresholds"
        
        lines = [f"\n  ⚠ {len(self.alerts)} Risk Alert(s) Triggered:"]
        
        for alert in self.alerts:
            icon = "🔴" if alert.severity == AlertSeverity.CRITICAL else "🟡"
            lines.append(f"    {icon} [{alert.severity.value}] {alert.message}")
        
        return "\n".join(lines)

