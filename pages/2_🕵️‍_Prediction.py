# imports
import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle as pkl
import shap


# # Load the saved model
# model = pkl.load(open("model_LR.p","rb"))

# setting
st.set_page_config(
    page_title="Customers Churning Prediction App",
    # layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)


######################
#main page layout
######################

st.subheader("To predict the default probability, you need to follow the steps below:")
st.markdown(
    """
    1. Choose the model threhold to determine how conservative the model is;
    2. Choose the parameters that best descibe your customers on the left side bar;
    3. Press the "Predict" button and wait for the result.
    """
    )
st.subheader("Data Dictionary")
st.markdown(
    """
    |  Feature  | Description |
    |  ----  | ----  | 
    | Total_Trans_Ct   | Total Transaction Count (Last 12 months) |
    | Total_Trans_Amt   | Total Transaction Amount (Last 12 months) |
    | Total_Ct_Chng_Q4_Q1   | Change in Transaction Count (Q4 over Q1)|
    | Total_Amt_Chng_Q4_Q1| Change in Transaction Amount (Q4 over Q1)|
    | Total_Revolving_Bal     |Total Revolving Balance on the Credit Card|
    | Total_Relationship_Count|Total no. of products held by the customer|
    | Avg_Utilization_Ratio     |Average Card Utilization Ratio|
    | Months_Inactive_12_mon|No. of months inactive in the last 12 months|
    | Contacts_Count_12_mon    |No. of Contacts in the last 12 months|
    | Credit_Limit    |Credit Limit on the Credit Card|
    """
    )

######################
#sidebar layout
######################


# Load the saved model
model = pkl.load(open("pages/model_RF.p","rb"))


# input threshold
st.sidebar.title("Model threshold")
st.sidebar.write("The smaller the threshold, the more conservative the model")
threshold = st.sidebar.slider("Threshold:",min_value=0.0, max_value=1.0, step=0.01, value = 0.50)


# input user features
st.sidebar.title("Customer Info")
st.sidebar.write("Please choose parameters that descibe the customer")

Total_Trans_Ct = st.sidebar.slider("Total_Trans_Ct:",min_value=0, max_value=150, step=1, value = 30)
Total_Trans_Amt = st.sidebar.slider("Total_Trans_Amt:",min_value=0, max_value=20000, step=100, value = 2000)
Total_Ct_Chng_Q4_Q1 = st.sidebar.slider("Total_Ct_Chng_Q4_Q1:",min_value=0.0, max_value=5.0, step=0.01, value = 0.3)
Total_Amt_Chng_Q4_Q1 = st.sidebar.slider("Total_Amt_Chng_Q4_Q1:",min_value=0.0, max_value=5.0, step=0.01, value = 0.3)
Total_Revolving_Bal = st.sidebar.slider("Total_Revolving_Bal:",min_value=0, max_value=3000, step=100, value = 500)
Total_Relationship_Count = st.sidebar.slider("Total_Relationship_Count:",min_value=1, max_value=6, step=1, value=1)
Avg_Utilization_Ratio = st.sidebar.slider("Avg_Utilization_Ratio:",min_value=0, max_value=5, step=1, value = 1)
Months_Inactive_12_mon = st.sidebar.slider("Months_Inactive_12_mon:",min_value=0, max_value=6, step=1, value = 5)
Contacts_Count_12_mon = st.sidebar.slider("Contacts_Count_12_mon:",min_value=0, max_value=6, step=1, value = 1)
Credit_Limit = st.sidebar.slider("Credit_Limit:",min_value=0, max_value=40000, step=100, value = 2000)
# credit_class = st.sidebar.selectbox("Credit class:",("A", "B", "C", "D", "E", "F", 'G'), index = 1)




######################
#Interpret the result
######################

# preprocess user-input data
def preprocess(Total_Trans_Ct, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Total_Amt_Chng_Q4_Q1, Total_Revolving_Bal, Total_Relationship_Count, Avg_Utilization_Ratio, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit):
    # Pre-processing user input

    user_input_dict = {
        'Total_Trans_Ct':[Total_Trans_Ct],
        'Total_Trans_Amt':[Total_Trans_Amt], 
        'Total_Ct_Chng_Q4_Q1':[Total_Ct_Chng_Q4_Q1],
        'Total_Amt_Chng_Q4_Q1':[Total_Amt_Chng_Q4_Q1],
        'Total_Revolving_Bal':[Total_Revolving_Bal],
        'Total_Relationship_Count':[Total_Relationship_Count],
        'Avg_Utilization_Ratio':[Avg_Utilization_Ratio],
        'Months_Inactive_12_mon':[Months_Inactive_12_mon],
        'Contacts_Count_12_mon':[Contacts_Count_12_mon],
        'Credit_Limit':[Credit_Limit],
        
    }
    user_input = pd.DataFrame(data=user_input_dict)

    # cleaner_type = {
    #     "class": {"A": 1, "B": 2, "C": 3, "D": 4,"E": 5, "F": 6,'G': 7},
    # }

    # user_input = user_input.replace(cleaner_type)

    return user_input



# user_input = preprocessed data
user_input = preprocess(
    Total_Trans_Ct, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Total_Amt_Chng_Q4_Q1, Total_Revolving_Bal, 
    Total_Relationship_Count, Avg_Utilization_Ratio, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit)

cf = st.sidebar.selectbox("Choose a feature for denpendence plot", (user_input.columns),0)

# predict button
btn_predict = st.sidebar.button("Predict")

# load the data for shap
loans = st.cache(pd.read_csv)("pages/BankChurners.csv") # allow_output_mutation=True
# class_dict = {
#     'A': 1,
#     'B': 2,
#     'C': 3,
#     'D': 4,
#     'E': 5,
#     'F': 6,
#     'G': 7,
# }
# loans_data['class'] = loans['class'].map(class_dict)
X = loans[[
        'Total_Trans_Ct',
        'Total_Trans_Amt', 
        'Total_Ct_Chng_Q4_Q1',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Revolving_Bal',
        'Total_Relationship_Count',
        'Avg_Utilization_Ratio',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        ]]
y = loans[['Attrition_Flag']]
y_ravel = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.2, random_state=2023, stratify=y)

if btn_predict:

    st.subheader("Your input")
    st.write(user_input)


    st.subheader("The prediction result: ")
    pred = model.predict_proba(user_input)[:, 1]
    if pred[0] > threshold:
        st.error('The applicant has a high probability to churn!')
        st.write(f'Probability of churn: {round(pred[0],2)}')
    else:
        st.success('The aplicant has a low probability to churn!')
        st.write(f'Probability of churn: {round(pred[0],2)}')


    st.subheader('Model Interpretability - Customer Level')
    from shap.plots import _waterfall
    shap.initjs()
    explainer = shap.TreeExplainer(model,X_test)
    shap_values = explainer(user_input)
    waterfall = _waterfall.waterfall_legacy(explainer.expected_value[1], shap_values.values[0][:,1], X_test.iloc[0])
    st.pyplot(waterfall)

    

    st.subheader('Model Interpretability - Overall Level')
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    # shap绝对值平均值-特征重要性
    fig_importance = shap.summary_plot(shap_values[1], X_test, plot_type = 'bar', plot_size = (10,8))
    st.pyplot(fig_importance)

    # summary_plot
    fig_cellular = shap.summary_plot(shap_values[1], X_test, plot_size = (10,6)) # plot_type = 'violin'
    st.pyplot(fig_cellular)

    # Dependence plot for features
    fig_denpendence = shap.dependence_plot(cf, shap_values[1], X_test, interaction_index=None)
    st.pyplot(fig_denpendence)

   


