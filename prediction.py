import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
import warnings
import random
import joblib
import streamlit as st


warnings.filterwarnings('ignore', category=FutureWarning)

random.seed(42)
np.random.seed(42)

# 全局定义特征列表
features = ['disease_count', 'Age', 'ADL_total_y']
feature_name_mapping = {'disease_count': 'Disease Count', 'Age': 'Age', 'ADL_total_y': 'ADL Self-Rating'}
feature_names_new = [feature_name_mapping[feat] for feat in features]

# 主函数
def main():
    # 加载数据
    df = pd.read_excel('1_filtered_data_with_trajectory.xlsx')

    X = df[features].copy()
    y = df['trajectory'].copy()

    # 编码和标准化
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 为有序分类变量进行编码
    ordinal_features = ['ADL_total_y', 'Age']  # 注意特征顺序
    ordinal_encoder = OrdinalEncoder(categories=[[1, 2, 3], [1, 2, 3]])  # 指定有序分类的类别顺序
    X[ordinal_features] = ordinal_encoder.fit_transform(X[ordinal_features])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # 数据拆分
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled_df, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

    # 模型调参
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 500],
        'gamma': [0, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }

    model = XGBClassifier(objective='binary:logistic', random_state=42)
    grid_search = RandomizedSearchCV(
        model, param_grid, n_iter=10, scoring='accuracy', cv=3, random_state=42, verbose=1)
    grid_search.fit(X_train_full, y_train_full)
    best_model = grid_search.best_estimator_

    # 保存最优模型
    joblib.dump(best_model, 'best_xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')

    # 保存训练集和测试集以供后续使用
    X_train_full.to_csv('X_train_full.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    np.savetxt('y_train_full.csv', y_train_full, delimiter=',')
    np.savetxt('y_test.csv', y_test, delimiter=',')

    # 返回模型、标准化器等
    return best_model, scaler, ordinal_encoder, X_train_full, X_test, y_train_full, y_test

# 初始化模型和其他工具
try:
    model = joblib.load('best_xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    ordinal_encoder = joblib.load('ordinal_encoder.pkl')
    X_train_full = pd.read_csv('X_train_full.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train_full = np.loadtxt('y_train_full.csv', delimiter=',')
    y_test = np.loadtxt('y_test.csv', delimiter=',')
except FileNotFoundError:
    model, scaler, ordinal_encoder, X_train_full, X_test, y_train_full, y_test = main()

# Streamlit UI 部分
st.title("Online Nomogram for Multimorbidity Trajectory Prediction")

# Disease count (Continuous)
disease_count = st.slider('Disease Count (Continuous)', min_value=0, max_value=14, step=1, value=7)

# Age (Ordinal)
age = st.selectbox('Age (Ordinal)', options=[1, 2, 3], index=1)

# ADL Self-Rating (Ordinal)
adl_rating = st.selectbox('ADL Self-Rating (Ordinal)', options=[1, 2, 3], index=1)

# 创建一个包含特征名的 DataFrame 并转换为适合模型输入的编码
input_data = pd.DataFrame([[disease_count, age, adl_rating]], columns=['disease_count', 'Age', 'ADL_total_y'])

# 将有序分类变量转换为模型训练时的编码
ordinal_features = ['ADL_total_y', 'Age']  # 确保顺序与训练时一致
input_data[ordinal_features] = ordinal_encoder.transform(input_data[ordinal_features])

# 标准化输入数据
input_data_scaled = scaler.transform(input_data)

# 进行预测
prediction_prob = model.predict_proba(input_data_scaled)[0, 1]
prediction_output = f"Predicted Probability of Deteriorating Trajectory: {prediction_prob:.3f}"

# 显示预测输出
st.write(prediction_output)
