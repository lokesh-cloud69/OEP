import time
import csv
import os

class AlertLogger:
    def __init__(self, log_file="proctor_log.csv"):
        self.log_file = log_file
        self.anomalies = {}
        self.init_log_file()

    def init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Anomaly Type", "Duration/Details"])

    def log_anomaly(self, anomaly_type, details=""):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, anomaly_type, details])
        print(f"ALERT LOGGED: {anomaly_type} - {details}")

    def track_state(self, anomaly_type, condition, threshold_seconds=2.0):
        """Tracks the state of an anomaly over time to avoid spamming alerts."""
        current_time = time.time()
        if condition:
            if anomaly_type not in self.anomalies:
                self.anomalies[anomaly_type] = current_time
            elif current_time - self.anomalies[anomaly_type] >= threshold_seconds:
                duration = current_time - self.anomalies[anomaly_type]
                self.log_anomaly(anomaly_type, f"Detected for {duration:.1f} seconds")
                self.anomalies[anomaly_type] = current_time
                return True # Anomaly triggered
        else:
            if anomaly_type in self.anomalies:
                del self.anomalies[anomaly_type]
        
        return False
