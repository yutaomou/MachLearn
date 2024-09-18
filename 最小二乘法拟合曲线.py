import numpy as np
import scipy as sp
from jedi.api.refactoring import inline
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from IPython import get_ipython
plt.rcParams['figure.figsize'] = [10, 5]  # 可选，设置默认图表大小

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)#表示2π * x 的正弦函数

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# 十个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]#通过向输入数据添加噪声，模型被迫学习对输入中的微小变化具有鲁棒性的特征，这可以帮助它在新的、看不见的数据上表现更好。

def main(M=8):
    """
    M为多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)#生成一个长度为M + 1的随机数组p_init，其中的每个元素都遵循均匀分布，范围是从0到1之间
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('拟合得到的参数值:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    return p_lsq

if __name__ == "__main__":
    main()
    plt.show()


