import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 生成一些示例数据，使用多个特征
X = np.random.rand(100, 3)  # 100个样本，3个特征
y = 3 * X[:, 0] + 5 * X[:, 1] + 2 * X[:, 2] + np.random.randn(100) * 0.5  # 目标变量

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 解析解
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # 添加偏置项
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # 计算解析解

# 使用梯度下降
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置项
    theta = np.random.randn(X_b.shape[1])  # 随机初始化权重
    losses = []

    for iteration in range(n_iterations):
        y_pred = X_b.dot(theta)
        loss = np.mean((y - y_pred) ** 2)  # 均方误差
        losses.append(loss)
        gradients = 2/m * X_b.T.dot(y_pred - y)  # 计算梯度
        theta -= learning_rate * gradients  # 更新权重

    return theta, losses

# 训练模型并记录损失
theta_gd, losses = gradient_descent(X_train, y_train)

# 进行预测
y_pred = X_test.dot(theta_gd[1:]) + theta_gd[0]  # 使用梯度下降的权重进行预测

# 可视化结果（由于是多变量，无法直接可视化所有特征，以下仅为示例）
plt.scatter(y_test, y_pred, color='blue')  # 原始数据点与预测结果
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # 理想预测线
plt.title('Multiple Variable Linear Regression')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.show()

# 可视化损失
plt.plot(losses, color='green')
plt.title('Loss over Iterations (Gradient Descent)')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.show()

# 可视化权重和偏置
weights = theta_gd[1:]
bias = theta_gd[0]

plt.bar(range(len(weights)), weights, color='orange', label='Weights')
plt.axhline(y=bias, color='red', linestyle='--', label='Bias')
plt.title('Weights and Bias (Gradient Descent)')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# 输出解析解
print("解析解的权重和偏置:", theta_best)
