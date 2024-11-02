import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 生成一些示例数据，使用多个特征
X = np.random.rand(100, 3)  # 100个样本，3个特征
y = 3 * X[:, 0] + 5 * X[:, 1] + 2 * X[:, 2] + np.random.randn(100) * 0.5  # 目标变量

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 可视化结果（由于是多变量，无法直接可视化所有特征，以下仅为示例）
plt.scatter(y_test, y_pred, color='blue')  # 原始数据点与预测结果
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # 理想预测线
plt.title('Multiple Variable Linear Regression')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.show()
