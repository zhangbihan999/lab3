import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

class SVM:
    def __init__(self):
        """ 初始化 """
        self.w_optimal = None
        self.b_optimal = None
    
    def objective(self, w, X, y):
        """ 目标函数 """
        return 0.5 * np.dot(w, w)
    
    def constraint(self, w, X, y):
        """ 约束条件 """
        return np.array([y[i] * (np.dot(X[i], w[:-1]) + w[-1]) - 1 for i in range(len(X))])

    def optimize(self, X, y):
        """ 优化过程 """
        # 初始化 w 和 b
        w0 = np.zeros(X.shape[1] + 1)
        # 目标函数
        obj = lambda w: self.objective(w, X, y)
        # 约束条件
        cons = [{'type': 'ineq', 'fun': lambda w: self.constraint(w, X, y)}]
        # 调用最优化函数进行优化
        result = minimize(obj, w0, constraints=cons)
        # 提取最优的 w 和 b
        self.w_optimal = result.x[:-1]
        self.b_optimal = result.x[-1]
    
    def predict(self, X):
        # 计算决策值
        decision_values = np.dot(X, self.w_optimal) + self.b_optimal
        # 获取预测标签
        predictions = np.sign(decision_values)
        # 转换为 0 和 1
        return np.where(predictions == -1, 0, 1)