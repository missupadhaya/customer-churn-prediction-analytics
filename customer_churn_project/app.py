import streamlit as st
import pandas as pd, joblib, json
from pathlib import Path

st.set_page_config(page_title='Churn Insight | Customer Analytics', page_icon='📊', layout='wide')
ROOT=Path(__file__).parent
model=joblib.load(ROOT/'models/churn_pipeline.pkl')
data=pd.read_csv(ROOT/'data/customer_churn.csv')
metrics=json.load(open(ROOT/'models/evaluation_metrics.json'))
st.title('📊 Customer Churn Prediction & Insight Dashboard')
st.caption('A machine-learning demo that estimates customer churn risk using customer service and subscription attributes.')
col1,col2,col3=st.columns(3)
col1.metric('Customers in sample',f'{len(data):,}')
col2.metric('Churn rate',f"{data.churn.mean():.1%}")
col3.metric('Model ROC-AUC',f"{metrics['roc_auc']:.3f}")
with st.expander('Model evaluation metrics'):
    st.json(metrics)
tab1,tab2=st.tabs(['📈 Dataset overview','🔮 Predict customer risk'])
with tab1:
    left,right=st.columns(2)
    with left:
        st.subheader('Churn distribution')
        st.bar_chart(data['churn'].value_counts().rename(index={0:'Retained',1:'Churned'}))
    with right:
        st.subheader('Average monthly charges by churn status')
        st.bar_chart(data.groupby('churn')['monthly_charges'].mean().rename(index={0:'Retained',1:'Churned'}))
    st.subheader('Sample records')
    st.dataframe(data.head(25),use_container_width=True)
with tab2:
    st.subheader('Enter customer details')
    with st.form('customer_input'):
        c1,c2=st.columns(2)
        tenure=c1.slider('Tenure (months)',0,72,12)
        monthly=c2.slider('Monthly charges ($)',20.0,120.0,65.0,step=1.0)
        tickets=c1.slider('Support tickets',0,12,1)
        outages=c2.slider('Service outages',0,10,0)
        contract=c1.selectbox('Contract type',['Month-to-month','One year','Two year'])
        internet=c2.selectbox('Internet service',['Fiber optic','DSL','No'])
        payment=c1.selectbox('Payment method',['Electronic check','Credit card','Bank transfer','Mailed check'])
        submit=st.form_submit_button('Predict churn risk',type='primary')
    if submit:
        record=pd.DataFrame([{'tenure_months':tenure,'monthly_charges':monthly,'support_tickets':tickets,'service_outages':outages,'contract_type':contract,'internet_service':internet,'payment_method':payment}])
        probability=float(model.predict_proba(record)[0][1])
        st.metric('Estimated churn probability',f'{probability:.1%}')
        if probability>=.65: st.error('High risk — consider proactive retention outreach (demo recommendation).')
        elif probability>=.35: st.warning('Medium risk — monitor satisfaction and usage.')
        else: st.success('Low risk based on the model inputs.')
        st.progress(probability)
st.caption('Demo project only. The dataset is synthetically generated and predictions are not validated for real business use.')
