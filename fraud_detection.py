import streamlit as st
import numpy as np 
import pandas as pd
import joblib 
from xgboost import XGBClassifier
from datetime import datetime

data=pd.read_csv("retail_fraud_detection_100k.csv")
import sys
import __main__

class custom_xgboost(XGBClassifier):
    def __init__(self,threshold,**kwargs):
        self.threshold=kwargs.pop("threshold",threshold)
        super().__init__(**kwargs)
    def get_params (self,deep=True):
        params=super().get_params(deep=deep)
        params['threshold']=self.threshold
        return params
    def predict(self,x):
        probility=self.predict_proba(x)
        prob=np.argmax(probility,axis=1)
        prob[probility[:,2]>=self.threshold] =2
        # prob[probility[:,1]>=0.42]=1
        return prob



__main__.custom_xgboost = custom_xgboost

pipeline=joblib.load("pipeline5.pkl")
model=joblib.load("model5.pkl")
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJYBy5UCwmpREJTWI39oxe82Ueysd7LehhHzSNOovl2f1OiHV5fmlKvoY&s=10",width=400)
st.title("Fraud Detection App ")
paymant_list=data['payment_method'].unique().tolist()
payment_method=st.selectbox("Enter the Payment Method",paymant_list)
transaction_amount=st.number_input("Enter the Payment of Transactions in dollar",value=20)
device_list=data['device_type'].unique().tolist()
device_type=st.selectbox("Select the device",device_list)

location=st.text_input("Enter the location",value="India")
merchant_category_list=data['merchant_category'].unique().tolist()
merchant_category=st.selectbox("Select the merchant category",merchant_category_list)
transaction_frequency_24h=st.number_input("Enter the transaction frequency of 24  hours",value=2)
customer_id=st.text_input("Enter the customer id",value="C21546")
failed_transaction_count_24h=st.number_input("Enter the failed transaction in 24 hours",value=1)
avg_transaction_amount_7d=st.number_input("Enter the average transaction of 7 days",value=5)


Trans_Aday = st.date_input("Select Transaction Date", datetime.now())



is_international_drop =st.selectbox("Is International",['Yes','No'])
is_international=1 if is_international_drop=='Yes' else 0


unusual_location_flag_drop=st.selectbox('unusual_location',['Yes',"No"])
unusual_location_flag=1 if unusual_location_flag_drop=='Yes' else 0

velocity_flag_drop=st.selectbox('Is Transcation high ?',['Yes',"No"])
velocity_flag=1 if velocity_flag_drop=='Yes' else 0

unusual_amount_flag_drop=st.selectbox('Is any unusual amount Transaction ?',['Yes',"No"])
unusual_amount_flag=1 if unusual_amount_flag_drop=='Yes' else 0

if st.button("Click Here"):
    input_dict={
        "payment_method":[payment_method],"transaction_amount":[transaction_amount],"device_type":[device_type],"location":[location],
        "merchant_category":[merchant_category],"transaction_frequency_24h":[transaction_frequency_24h],"customer_id":[customer_id],
        "failed_transaction_count_24h":[failed_transaction_count_24h],"avg_transaction_amount_7d":[avg_transaction_amount_7d] ,
        "unusual_location_flag": [unusual_location_flag],
        "velocity_flag": [velocity_flag],
        "unusual_amount_flag": [unusual_amount_flag],
         "is_international": [is_international],
        "payment_location_combo": [f"{payment_method}_{is_international}"],
        "merchant_payment_combo": [f"{merchant_category}_{payment_method}"],
        "device_location_combo": [f"{device_type}_{location}"],
        "ratio_amount": [0.0],
        "avg_faild_transaction": [0.0],
        "Trans_Aday" : [Trans_Aday] ,
        # "transaction_timesta/mp" : [T] ,
        "amt_deviation_by_payment": [0.0],
        "amt_deviation_by_merchant": [0.0],
        "Per_day_random_transaction": [0.0],
        "Per_day_Avgtransaction": [0.0],
        "high_risk_device_flag": [0],
        
    
    }
    df =pd.DataFrame(input_dict)
    try:
       
        transform_data = pipeline.transform(df)
        prediction =model.predict(transform_data)
       
        # xgb_prob = model['xgb'].predict_proba(transform_data)
        # lgb_prob = model['lgm'].predict_proba(transform_data)
        
        
        # final_pred = ((xgb_prob * 0.4) + (lgb_prob * 0.6))
        # final_predict = np.argmax(final_pred, axis=1)
        
       
        # final_predict[final_pred[:, 2] >= 0.45] = 2
        
        
        # prediction = final_predict[0]
        # st.success(f"The Fraud is catched by the system successfully{prediction[0]:.2f} ")
        
        predict_1=int(prediction[0])
        if predict_1==0:
            st.info("Congrat , There are no Fraud i your Transactions ")
        elif predict_1==1:
            st.warning(" Warning ! , Fraud may be Occured (Medium Fraud Risk)")
        elif predict_1==2:
            st.error(" Alert ! Some Fraud Transaction Occured(High Fraud Risk) ")
        
    except Exception as e :
        st.error(f"There are some error occured{e}")
    try:
        from sklearn.metrics import precision_recall_curve, auc
        import matplotlib.pyplot as plt
        data=pd.read_csv("data_test.csv")
        datatest=data.drop("fraud_risk_labels",axis=1)
        data_test=data["fraud_risk_labels"]
        fraudtest_prepared=pipeline.transform(datatest)
        prob_xgb =model.predict_proba(fraudtest_prepared)[:, 2]
        # prob_lgb = custom_light.predict_proba(fraudtest_prepared)[:, 2]
        
        # final_predict=((prob_xgb*0.4)+(prob_lgb*0.6))
        y_test_binary = (data_test == 2).astype(int)
        
        precision_f, recall_f, _ = precision_recall_curve(y_test_binary, prob_xgb)
        auc_f = auc(recall_f, precision_f)
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(recall_f, precision_f, color='red', label=f'ensemble Learning (PR-AUC = {auc_f:.4f})') 
        ax.set_title('Precision-Recall Curve for High Risk (Class 2)')
        ax.set_xlabel('Recall (capacity of detecte  fraud)')
        ax.set_ylabel('Precision (Perfection of fraud detection )')
        ax.legend(loc='lower left')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"The error is occured :{e}")
