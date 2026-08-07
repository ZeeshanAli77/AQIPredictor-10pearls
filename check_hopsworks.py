import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

print("API key loaded:", bool(HOPSWORKS_API_KEY))
print("API key length:", len(HOPSWORKS_API_KEY) if HOPSWORKS_API_KEY else 0)

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project="zeeshanproject"
)

fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=7)

df = fg.read(online=True)

print(f"Total rows: {len(df)}")
print(f"Last 5 rows:\n{df.tail()}")