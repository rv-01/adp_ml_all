import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///data/bank_loan.db')
df = pd.read_csv('data/bank_loan.csv')
df.to_sql('bank_loan', engine, if_exists='replace', index=False)
print("Data loaded to SQLite.")
