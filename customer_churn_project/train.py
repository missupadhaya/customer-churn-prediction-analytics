import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURES = ['tenure_months','monthly_charges','support_tickets','service_outages','contract_type','internet_service','payment_method']
NUM = FEATURES[:4]; CAT = FEATURES[4:]
data = pd.read_csv('data/customer_churn.csv')
X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data['churn'], test_size=0.2, stratify=data['churn'], random_state=42)
preprocessor = ColumnTransformer([('num', Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]), NUM),('cat', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]), CAT)])
model = Pipeline([('preprocessor',preprocessor),('classifier',RandomForestClassifier(n_estimators=350,max_depth=10,min_samples_leaf=3,class_weight='balanced',random_state=42))])
model.fit(X_train,y_train); predicted=model.predict(X_test); probability=model.predict_proba(X_test)[:,1]
metrics={'accuracy':accuracy_score(y_test,predicted),'precision':precision_score(y_test,predicted,zero_division=0),'recall':recall_score(y_test,predicted,zero_division=0),'f1':f1_score(y_test,predicted,zero_division=0),'roc_auc':roc_auc_score(y_test,probability)}
joblib.dump(model,'models/churn_pipeline.pkl')
with open('models/evaluation_metrics.json','w') as f: json.dump({k:float(v) for k,v in metrics.items()},f,indent=2)
print('Saved models/churn_pipeline.pkl'); print(json.dumps(metrics,indent=2))
