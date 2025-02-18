import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from bezier_renderer import diffvgProcess
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.linalg import sqrtm

# 自定义组合数计算函数
def comb_custom(n, k):
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    num = 1
    denom = 1
    for i in range(k):
        num *= (n - i)
        denom *= (i + 1)
    return num // denom
#bezier曲线参数化
def bezier_curve(t, P):
    '''
    p:list:num_points,2
    '''
    n = len(P) - 1  # 贝塞尔曲线的阶数
    result = np.zeros_like(P[0])  # 假设每个P[i]都是(x, y)点

    for i in range(n + 1):
        # 计算Bernstein基函数 B_{i,n}(t)
        bernstein = comb(n, i) * (1 - t) ** (n - i) * t ** i
        result += bernstein * P[i]

    return result


# 贝塞尔曲线的参数化 (不使用 numpy)
def bezier_curve(t, P):
    '''
    P: list of control points, each control point is a tuple (x, y, ...)
    t: parameter from 0 to 1
    '''
    n = len(P) - 1  # 贝塞尔曲线的阶数
    result = [0] * len(P[0])  # 假设每个P[i]是一个n维的点，用列表初始化

    for i in range(n + 1):
        # 计算 Bernstein 基函数 B_{i,n}(t)
        bernstein = comb_custom(n, i) * (1 - t) ** (n - i) * t ** i

        # 对每个控制点进行加权
        for j in range(len(P[0])):  # 假设每个控制点是二维（x, y），此处可以适应更高维度
            result[j] += bernstein * P[i][j]

    return result
# 贝塞尔曲线的一阶和二阶导数
def bezier_derivative(P, t):
    n = len(P) - 1
    first_derivative = torch.zeros_like(torch.tensor(P[0]))  # 初始化一阶导数
    second_derivative = torch.zeros_like(torch.tensor(P[0]))  # 初始化二阶导数

    # 计算一阶导数
    for i in range(n):
        first_derivative += comb_custom(n, i) * (n - i) * (1 - t) ** (n - i - 1) * t ** i * (torch.tensor(P[i + 1]) - torch.tensor(P[i]))

    # 计算二阶导数
    for i in range(n - 1):
        second_derivative += comb_custom(n, i) * (n - i) * (n - i - 1) * (1 - t) ** (n - i - 2) * t ** i * (torch.tensor(P[i + 2]) - 2 * torch.tensor(P[i + 1]) + torch.tensor(P[i]))

    return first_derivative, second_derivative

# 计算曲率
def calculate_curvature(P, t):
    first_derivative, second_derivative = bezier_derivative(P, t)

    # 一阶导数的分量
    x_prime, y_prime = first_derivative[0], first_derivative[1]

    # 二阶导数的分量
    x_double_prime, y_double_prime = second_derivative[0], second_derivative[1]

    # 使用曲率公式
    numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
    denominator = (x_prime ** 2 + y_prime ** 2) ** (3 / 2)

    curvature = numerator / denominator
    return curvature

#计算FID
def calculate_fid(mtx1, mtx2):
    # 计算均值和协方差矩阵
    mean1 = np.mean(mtx1, axis=1)
    mean2 = np.mean(mtx2, axis=1)
    print(mtx1.shape, mean1.shape)
    cov1 = np.cov(mtx1, rowvar=False)
    cov2 = np.cov(mtx2, rowvar=False)
    print(cov1.shape, mtx2.shape)
    # 计算FID的第一部分：均值差的平方
    mean_diff = np.sum((mean1 - mean2) ** 2)

    # 计算FID的第二部分：协方差的迹部分
    cov_diff = cov1 + cov2 - 2 * sqrtm(np.dot(np.dot(sqrtm(cov1), cov2), sqrtm(cov1)))

    # FID的最终值
    fid = mean_diff + np.trace(cov_diff)

    return fid


if __name__ == '__main__':
    # 加载控制点参数
    control_points = torch.load(f"/home/dhu/zyl/CLIPasso/CLIPasso-main/control_points_epoch.pt")
    control_points=(torch.stack(control_points,dim=0))
    control_points_tensor=control_points.clone().cpu().detach().numpy()
    #计算曲率
    test=control_points_tensor[0]
    bezier_curve=bezier_curve(0.5,test)
    print("Curvature at t=0.5:", bezier_curve)
    curvature = calculate_curvature(test, 0.5)
    print("Curvature at t=0.5:", curvature)
    test2 = control_points_tensor[1]
    test3=control_points_tensor[2]

    #计算FID
    feature = torch.load(f"/home/dhu/zyl/CLIPasso/CLIPasso-main/feature.pt")
    x_i=feature[0].clone().cpu().detach().numpy()
    x_j=feature[1].clone().cpu().detach().numpy()

    # 将 x_i 和 x_j 转换为 2D 数组，每行一个特征点
    x_i = x_i.reshape(-1, 1)
    x_j = x_j.reshape(-1, 1)
    print(x_i.shape, x_j.shape)
    # 执行 Procrustes 分析
    mtx1 , mtx2,_ = procrustes(x_i, x_j)
    mtx1 = mtx1.T  # 转置，使其形状变为 (1, 8)
    mtx2 = mtx2.T  # 转置，使其形状变为 (1, 8)

    # 计算FID
    fid = calculate_fid(mtx1, mtx2)
    print("FID:", fid)


    # img = diffvgProcess(control_points,224,224)
    # img2 = img.detach().cpu().numpy()
    # # 使用 matplotlib 绘制图像
    # plt.imshow(img2,cmap='gray')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    # plt.savefig("img2")
    # # 打印控制点参数
    # print(img.shape)
