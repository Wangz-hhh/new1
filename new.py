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

# 训练模型并记录损失
losses = []
for _ in range(100):  # 进行多次迭代以模拟训练过程
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    loss = np.mean((y_train - y_pred_train) ** 2)  # 均方误差
    losses.append(loss)

# 进行预测
y_pred = model.predict(X_test)

# 可视化结果（由于是多变量，无法直接可视化所有特征，以下仅为示例）
plt.scatter(y_test, y_pred, color='blue')  # 原始数据点与预测结果
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # 理想预测线
plt.title('Multiple Variable Linear Regression')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.show()

# 可视化损失
plt.plot(losses, color='green')
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.show()

# 可视化权重和偏置
weights = model.coef_
bias = model.intercept_

plt.bar(range(len(weights)), weights, color='orange', label='Weights')
plt.axhline(y=bias, color='red', linestyle='--', label='Bias')
plt.title('Weights and Bias')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.show()
