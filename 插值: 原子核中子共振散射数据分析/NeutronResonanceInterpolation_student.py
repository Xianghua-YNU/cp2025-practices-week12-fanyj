import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 实验数据
energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # mb
error = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # mb

def lagrange_interpolation(x, x_data, y_data):
    """
    实现拉格朗日多项式插值
    
    参数:
        x: 插值点或数组
        x_data: 已知数据点的x坐标
        y_data: 已知数据点的y坐标
        
    返回:
        插值结果
    """
    x = np.asarray(x)
    result = np.zeros_like(x)
    
    for i in range(len(x_data)):
        # 计算拉格朗日基函数 Li(x)
        Li = np.ones_like(x, dtype=float)  # 明确指定数据类型为float
        for j in range(len(x_data)):
            if i != j:
                Li *= (x - x_data[j]) / (x_data[i] - x_data[j])
        # 累加基函数乘以对应y值
        result += y_data[i] * Li
    
    return result

def cubic_spline_interpolation(x, x_data, y_data):
    """
    实现三次样条插值
    
    参数:
        x: 插值点或数组
        x_data: 已知数据点的x坐标
        y_data: 已知数据点的y坐标
        
    返回:
        插值结果
    """
    # 使用scipy的interp1d函数实现三次样条插值
    f = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return f(x)

def find_peak(x, y):
    """
    寻找峰值位置和半高全宽(FWHM)
    
    参数:
        x: x坐标数组
        y: y坐标数组
        
    返回:
        tuple: (峰值位置, FWHM)
    """
    # 找到峰值位置
    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    
    # 计算半高
    half_max = peak_y / 2
    
    # 寻找半高对应的左右位置
    # 左边：找到最后一个小于等于half_max的点
    left_idx = np.where(y[:peak_idx] <= half_max)[0]
    left_idx = left_idx[-1] if len(left_idx) > 0 else 0
    
    # 右边：找到第一个小于等于half_max的点
    right_idx = np.where(y[peak_idx:] <= half_max)[0]
    right_idx = right_idx[0] + peak_idx if len(right_idx) > 0 else len(y) - 1
    
    # 线性插值获取更准确的半高位置
    if left_idx < peak_idx and y[left_idx] < half_max < y[left_idx + 1]:
        t = (half_max - y[left_idx]) / (y[left_idx + 1] - y[left_idx])
        left_x = x[left_idx] + t * (x[left_idx + 1] - x[left_idx])
    else:
        left_x = x[left_idx]
        
    if right_idx > peak_idx and y[right_idx - 1] > half_max > y[right_idx]:
        t = (half_max - y[right_idx]) / (y[right_idx - 1] - y[right_idx])
        right_x = x[right_idx] + t * (x[right_idx - 1] - x[right_idx])
    else:
        right_x = x[right_idx]
    
    # 计算FWHM
    fwhm = right_x - left_x
    
    return peak_x, fwhm

def plot_results():
    """
    绘制插值结果和原始数据对比图
    
    提示:
        1. 生成密集的插值点
        2. 调用前面实现的插值函数
        3. 绘制原始数据点和插值曲线
    """
    # 生成密集的插值点
    x_interp = np.linspace(0, 200, 500)
    
    # 计算两种插值结果
    lagrange_result = lagrange_interpolation(x_interp, energy, cross_section)
    spline_result = cubic_spline_interpolation(x_interp, energy, cross_section)
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    
    # 原始数据点
    plt.errorbar(energy, cross_section, yerr=error, fmt='o', color='black', 
                label='Original Data', capsize=5)
    
    # 插值曲线
    plt.plot(x_interp, lagrange_result, '-', label='Lagrange Interpolation')
    plt.plot(x_interp, spline_result, '--', label='Cubic Spline Interpolation')
    
    # 标记峰值
    lagrange_peak, lagrange_fwhm = find_peak(x_interp, lagrange_result)
    spline_peak, spline_fwhm = find_peak(x_interp, spline_result)
    
    plt.axvline(lagrange_peak, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(spline_peak, color='orange', linestyle=':', alpha=0.5)
    
    # 在峰值线上标注峰值能量和FWHM
    plt.text(lagrange_peak, max(lagrange_result) * 0.9, 
             f'Lagrange: E={lagrange_peak:.1f} MeV\nFWHM={lagrange_fwhm:.1f} MeV', 
             ha='center', va='top', rotation=90, backgroundcolor='w', alpha=0.7)
    
    plt.text(spline_peak, max(spline_result) * 0.9, 
             f'Spline: E={spline_peak:.1f} MeV\nFWHM={spline_fwhm:.1f} MeV', 
             ha='center', va='top', rotation=90, backgroundcolor='w', alpha=0.7)
    
    # 图表装饰
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Cross Section (mb)')
    plt.title('Neutron Resonance Scattering Cross Section Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    plot_results()
