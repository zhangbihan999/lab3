import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from SVM import SVM

# 读取数据集
df = pd.read_csv('../dataset/heart_failure_clinical_records_dataset.csv')
#print(df.head())
#print(f'The dataset have included {df.shape[0]} columns and {df.shape[1]} rows.')

# 切分数据集
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10086)

# 归一化 X_train 和 X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print(X_test_scaled[:5])

# 将 y 转换成 -1 和 +1
y_train = y_train.apply(lambda x: 1 if x == 1 else -1)  # apply 方法会将函数执行后的结果存储在一个新的列表中
y_test = y_test.apply(lambda x: 1 if x == 1 else -1)

# 重置 y_train 和 y_test 的索引
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 将 y_train 和 y_test 转化为 numpy 数组
y_train = y_train.values
y_test = y_test.values

# 实例化 SVM 对象
svm = SVM()

# 使用训练集进行优化
svm.optimize(X_train_scaled, y_train)

print("Optimal weights:", svm.w_optimal)
print("Optimal bias:", svm.b_optimal)

# 预测
y_pred = svm.predict(X_test_scaled)

# 输出前10个预测结果
#print("Predicted labels (first 10):", y_pred[:10])

# 计算预测的准确率
accuracy = np.mean(y_pred == y_test)
print("\nAccuracy on test set:", accuracy)