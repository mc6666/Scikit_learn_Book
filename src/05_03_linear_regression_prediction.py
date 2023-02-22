import streamlit as st
import joblib

# 載入模型與標準化轉換模型
model = joblib.load('lr_model.joblib')
scaler = joblib.load('lr_scaler.joblib')

list1 = [0 for _ in range(13)]
st.title('Boston 房價預測')
col1, col2 = st.columns(2)
with col1:
    list1[0] = st.slider('犯罪率:', value=1.7, min_value=0.0, max_value=10.0)
    list1[1] = st.slider('大坪數房屋比例:', value=11.0, min_value=0.0, max_value=100.0)
    list1[2] = st.slider('非零售業的營業面積比例:', value=11.0, min_value=0.0, max_value=100.0)
    list1[3] = 0 if st.radio('是否靠近河岸:', options=('否', '是'))=='否' else 1
    list1[4] = st.slider('一氧化氮濃度:', value=0.5, min_value=0.0, max_value=1.0)
    list1[5] = st.slider('平均房間數:', value=6.0, min_value=3.0, max_value=9.0)
    list1[6] = st.slider('屋齡(1940年前建造比例):', value=0.0, min_value=68.0, max_value=100.0)
with col2:    
    list1[7] = st.slider('與商業區距離:', value=3.8, min_value=1.0, max_value=12.5)
    list1[8] = st.slider('與高速公路距離:', value=10.0, min_value=1.0, max_value=25.0)
    list1[9] = st.slider('地價稅:', value=408.0, min_value=180.0, max_value=720.0)
    list1[10] = st.slider('師生比例:', value=18.0, min_value=12.0, max_value=22.0)
    list1[11] = st.slider('黑人比例(Bk — 0.63)²:', value=356.0, min_value=0.0, max_value=400.0)
    list1[12]= st.slider('低下階級的比例:', value=12.0, min_value=0.0, max_value=38.0)

if st.button('預測'):
    X_new = [list1]
    X_new = scaler.transform(X_new)
    st.write(f'### 預測房價：{model.predict(X_new)[0]:.2f}')
