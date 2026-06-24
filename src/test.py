import pandas as pd
from config import CSV_PATH, CYCLE_COL, VOLTAGE_COL

df = pd.read_csv(CSV_PATH)
print(df.groupby(CYCLE_COL)[VOLTAGE_COL].agg(['min', 'max', 'count']))