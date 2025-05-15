import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

engine = create_engine('sqlite:///data/bank_loan.db')
df = pd.read_sql('bank_loan', engine)

num_feats = ['Age','Experience','Income']
cat_feats = ['Family','Education']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(sparse=False), cat_feats),
])

X = preprocessor.fit_transform(df[num_feats + cat_feats])
