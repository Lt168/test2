import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# 读取数据
insurance_df = pd.read_csv('insurance-chinese.csv', encoding='gbk')
features = insurance_df[['年龄', '性别', 'BMI', '子女数量', '是否吸烟', '区域']]
output = insurance_df['医疗费用']

# 独热编码，并保存特征列（名称+顺序）
features = pd.get_dummies(features)
# 保存特征列到文件
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(features.columns.tolist(), f)  # 保存特征列列表

# 训练模型（略）
x_train, x_test, y_train, y_test = train_test_split(features, output, train_size=0.8, random_state=42)
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)

# 保存模型
with open('rfr_model.pkl', 'wb') as f:
    pickle.dump(rfr, f)
