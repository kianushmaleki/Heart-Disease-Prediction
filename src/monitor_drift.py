import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset



# Load the data
reference = pd.read_csv("data/heart_disease_uci.csv")
new_data = pd.read_csv("bad_data/heart_disease_uci_bad.csv")

report_month1 = Report(metrics=[DataDriftPreset()])
snapshot_month1 = report_month1.run(reference_data=reference, current_data=new_data)
snapshot_month1.save_html("reports/drift_month1.html")
print("Report saved to reports/drift_month1.html")

