import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_supernova_data(file_path):
    """
    从文件中加载超新星数据
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: 包含以下元素的元组
            - z (numpy.ndarray): 红移数据
            - mu (numpy.ndarray): 距离模数数据
            - mu_err (numpy.ndarray): 距离模数误差
    """
    data = np.loadtxt(file_path)
    z = data[:, 0]
    mu = data[:, 1]
    mu_err = data[:, 2]
    return z, mu, mu_err


def hubble_model(z, H0):
    """
    哈勃模型：距离模数与红移的关系
    
    参数:
        z (float or numpy.ndarray): 红移
        H0 (float): 哈勃常数 (km/s/Mpc)
        
    返回:
        float or numpy.ndarray: 距离模数
    """
    # 假设 c 为光速，单位为 km/s
    c = 299792.458
    # 距离 d_L 单位为 Mpc
    d_L = (c / H0) * z
    # 转换为距离模数
    mu = 5 * np.log10(d_L) + 25
    return mu


def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的哈勃模型
    
    参数:
        z (float or numpy.ndarray): 红移
        H0 (float): 哈勃常数 (km/s/Mpc)
        a1 (float): 拟合参数，对应于减速参数q0
        
    返回:
        float or numpy.ndarray: 距离模数
    """
    c = 299792.458
    # 包含一阶减速参数的距离公式
    d_L = (c / H0) * z * (1 + z * (1 - a1) / 2)
    mu = 5 * np.log10(d_L) + 25
    return mu


def hubble_fit(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        
    返回:
        tuple: 包含以下元素的元组
            - H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
            - H0_err (float): 哈勃常数的误差
    """
    # 初始猜测值
    p0 = [70]  # 哈勃常数初始猜测值
    popt, pcov = curve_fit(hubble_model, z, mu, p0=p0, sigma=mu_err, absolute_sigma=True)
    H0 = popt[0]
    H0_err = np.sqrt(pcov[0, 0])
    return H0, H0_err


def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数和减速参数
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        
    返回:
        tuple: 包含以下元素的元组
            - H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
            - H0_err (float): 哈勃常数的误差
            - a1 (float): 拟合得到的a1参数
            - a1_err (float): a1参数的误差
    """
    # 初始猜测值
    p0 = [70, 0.5]  # 哈勃常数和a1参数的初始猜测值
    popt, pcov = curve_fit(hubble_model_with_deceleration, z, mu, p0=p0, sigma=mu_err, absolute_sigma=True)
    H0, a1 = popt
    H0_err = np.sqrt(pcov[0, 0])
    a1_err = np.sqrt(pcov[1, 1])
    return H0, H0_err, a1, a1_err


def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制哈勃图（距离模数vs红移）
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    plt.figure(figsize=(10, 6))
    # 绘制数据点
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', markersize=5, capsize=3, label='观测数据')
    
    # 绘制拟合曲线
    z_range = np.linspace(min(z), max(z), 100)
    plt.plot(z_range, hubble_model(z_range, H0), 'r-', 
             label=f'哈勃模型拟合 (H0 = {H0:.2f} km/s/Mpc)')
    
    plt.xlabel('红移 z')
    plt.ylabel('距离模数 μ')
    plt.title('哈勃图 - 超新星距离模数 vs 红移')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
        a1 (float): 拟合得到的a1参数
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    plt.figure(figsize=(10, 6))
    # 绘制数据点
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', markersize=5, capsize=3, label='观测数据')
    
    # 绘制拟合曲线
    z_range = np.linspace(min(z), max(z), 100)
    plt.plot(z_range, hubble_model_with_deceleration(z_range, H0, a1), 'g-', 
             label=f'含减速参数模型 (H0 = {H0:.2f}, a1 = {a1:.2f})')
    
    plt.xlabel('红移 z')
    plt.ylabel('距离模数 μ')
    plt.title('哈勃图 - 包含减速参数的模型')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


if __name__ == "__main__":
    # 数据文件路径
    data_file = "data/supernova_data.txt"
    
    # 加载数据
    z, mu, mu_err = load_supernova_data(data_file)
    
    # 拟合哈勃常数
    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    
    # 绘制哈勃图
    fig = plot_hubble_diagram(z, mu, mu_err, H0)
    plt.show()
    
    # 可选：拟合包含减速参数的模型
    H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"拟合得到的a1参数: a1 = {a1:.2f} ± {a1_err:.2f}")
    
    # 绘制包含减速参数的哈勃图
    fig = plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1)
    plt.show()
