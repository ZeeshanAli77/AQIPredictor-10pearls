# delete_bad_rows.py
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project="zeeshanproject"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=7)

# Delete rows after backfill period (after Aug 5, 2026)
# This keeps only the good backfill data
fg.delete("timestamp > '2026-08-05'")
print("✓ Deleted bad rows from feature store")
