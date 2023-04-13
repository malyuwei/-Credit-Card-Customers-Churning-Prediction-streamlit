import streamlit as st

st.set_page_config(
    page_title="Customers Churning Prediction App",
    # page_icon="💳",
)

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Credit Card Customers Churning Prediction App💳")
st.write("### Welcome to our app! 👋")
st.write("#### Are you struggling with credit card customers churning?")
st.write("#### This machine learning application can assist you in making a prediction to find customers who might churn!")
st.image("credit card.jpg")
st.write(
"""
A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.
""")
