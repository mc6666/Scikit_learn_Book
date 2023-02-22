import streamlit as st
import joblib

# 載入模型與標準化轉換模型
clf = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('鳶尾花（Iris）預測')
sepal_length = st.slider('花萼長度:', min_value=3.0, max_value=8.0, value=5.8)
sepal_width = st.slider('花萼寬度:', min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.slider('花瓣長度:', min_value=1.0, max_value=7.0, value=4.4)
petal_width = st.slider('花瓣寬度:', min_value=0.1, max_value=2.5, value=1.3)

labels = ['setosa', 'versicolor', 'virginica']
if st.button('預測'):
    X_new = [[sepal_length,sepal_width,petal_length,petal_width]]
    X_new = scaler.transform(X_new)
    st.write('### 預測品種是：', labels[clf.predict(X_new)[0]])
