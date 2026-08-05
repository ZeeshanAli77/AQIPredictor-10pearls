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
fg = fs.get_feature_group("aqi_features", version=1)

# Get the data from online storage (where we're currently writing)
df = fg.read(online=True)
print(f"Total rows: {len(df)}")
print(f"Last 5 rows:\n{df.tail()}")