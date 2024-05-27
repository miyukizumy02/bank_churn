import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu và mô hình
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing giống như trước
labelencode = LabelEncoder()
def encode(data):
    for col in data.columns:
        if data[col].dtypes == 'object':
            data[col] = labelencode.fit_transform(data[col])
    return data

train_data = encode(train_data)
test_data = encode(test_data)

# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])
test_data = pd.get_dummies(test_data, columns=['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])

# Chuẩn bị dữ liệu huấn luyện
X = train_data.drop("Exited", axis=1)
y = train_data["Exited"]

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier()
model.fit(X, y)

# Ứng dụng Streamlit
st.title('Bank Churn Prediction App')

st.write("""
### Dự đoán khách hàng có rời bỏ dịch vụ hay không
""")

# Tạo form nhập dữ liệu
Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
Gender = st.selectbox('Gender', ['Female', 'Male'])
CreditScore = st.slider('Credit Score', 350, 850, 650)
Age = st.slider('Age', 18, 92, 35)
Tenure = st.slider('Tenure', 0, 10, 5)
Balance = st.number_input('Balance', 0.00, 250000.00, 10000.00)
NumOfProducts = st.selectbox('Num Of Products', [1, 2, 3, 4])
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
EstimatedSalary = st.number_input('Estimated Salary', 0.00, 200000.00, 50000.00)

# Chuyển đổi các giá trị đầu vào thành DataFrame
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# Áp dụng encoding và one-hot encoding cho dữ liệu đầu vào
input_data = encode(input_data)
input_data = pd.get_dummies(input_data, columns=['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])

# Đảm bảo rằng các cột khớp với mô hình huấn luyện
missing_cols = set(X.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[X.columns]

# Dự đoán
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.write('### Kết quả dự đoán')
st.write(f'Khách hàng {"rời bỏ" if prediction[0] else "không rời bỏ"} dịch vụ.')
st.write(f'Xác suất: {prediction_proba[0][1]:.2f}')

