import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 生成一些示例数据
X = np.array([[1], [2], [3], [4], [5]])  # 特征
y = np.array([1, 2, 3, 4, 5])            # 目标变量

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 可视化结果
plt.scatter(X, y, color='blue')  # 原始数据点
plt.plot(X_test, y_pred, color='red', linewidth=2)  # 预测结果
plt.title('Single Variable Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
