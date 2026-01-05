import logging
from datetime import datetime

class RiskAlertSystem:
    def __init__(self):
        self.logger = logging.getLogger("RiskScanner")
        self.logger.setLevel(logging.INFO)
        # Create console handler
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def trigger(self, alert):
        """
        Log and potentially notify user of a risk alert.
        """
        msg = f"[{alert['type']}] {alert['message']} (Value: {alert['value']:.4f}, Severity: {alert['severity']})"
        if alert['severity'] == "HIGH":
            self.logger.error(msg)
        else:
            self.logger.warning(msg)
            
        # Here you could add email/Slack/Telegram notification logic
        return msg

    def log_rebalance(self, trades):
        for trade in trades:
            self.logger.info(f"REBALANCE: {trade['action']} {trade['asset']} amount {trade['amount']:.2f}")
