import pandas as pd
import requests
from io import StringIO
import datetime

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
tables = pd.read_html(StringIO(response.text))

current_df = tables[0]
changes_df = tables[1]

print("Current table shape:", current_df.shape)
print("Changes table shape:", changes_df.shape)
print("Changes columns:", changes_df.columns)
print(changes_df.head())
