import math
import numpy as np

"""
1. 雷达点按照弧长投影理论点，得到理论点的平面，该平面与XY平面平行。
2. 雷达点在上述平面上的投影点 (x, y, z+h)。需要球坐标转直角坐标，然后再直角坐标转球坐标。
3. 该点在理论点云中最近邻居，理论点云的最小间距。 (x1, y1, z1)，直角和球坐标都需要
4. 得到雷达点与理论点的映射集合。统计雷达点云分组中映射计数。
5. 从映射计数最多的分组开始，重构雷达点云。（这部分是细节可以以后再做）
6. 理论点云标签，顶板A，背板A，顶板B，背板B，，，，，，，
7. 根据理论点云查找顶板的最低位置。
"""
def intersect_points(model_points, radar_points):
    """
    对比顶板理论点云与雷达点云，选择水平俯仰角度一样的点。

    参数：
        model_points: 理论点云 (r, theta, phi)
        radar_points: 雷达点云 (r, theta, phi)

    返回：
        水平俯仰角度集合 (theta, phi)
    """
    A = model_points[:, [1,2]]
    B = radar_points[:, [1,2]]
    return np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])

def diff_height(model_points, radar_points, height_bound):
    """
    根据理论点云与雷达点云的高度差重新选择映射的点
    """
    model_new_points = []
    map_points = intersect_points(model_points, radar_points)
    for idx in map_points:
        elev_sin = math.sin(math.radians(idx[1]))
        if idx[0] == radar_points[1] and idx[1] == radar_points[2]:
            radar_height = radar_points[0] * elev_sin
            radar_r = radar_points[0]
        if idx[0] == model_points[1] and idx[1] == model_points[2]:
            model_height = model_points[0] * elev_sin
        diff_h = abs(radar_height-model_height)
        if diff_h > height_bound:
            continue
        new_r = calculate_third_side(radar_r, diff_h, 90+idx[1])
        new_phi = math.asin(model_height/new_r)
        model_new_points.append([new_r, idx[0], new_phi])
    return model_new_points

def calculate_third_side(a, b, C_degrees):
    """
    使用余弦定理计算三角形的第三边长。
    
    参数:
        a (float): 已知边长 a
        b (float): 已知边长 b
        C_degrees (float): 两边之间的夹角 C（以度为单位）
    
    返回:
        float: 第三边长 c
    """
    # 将角度从度转换为弧度
    C_radians = math.radians(C_degrees)
    
    # 使用余弦定理公式计算 c^2
    c_squared = a**2 + b**2 - 2 * a * b * math.cos(C_radians)
    
    # 计算 c 的值（开平方）
    c = math.sqrt(c_squared)
    
    return c

# 示例使用
a = float(input("请输入已知边长 a: "))
b = float(input("请输入已知边长 b: "))
C_degrees = float(input("请输入两边之间的夹角 C（以度为单位）: "))

# 计算第三边长
c = calculate_third_side(a, b, C_degrees)

print(f"第三边长 c 的长度为: {c:.4f}")