import open3d as o3d
import numpy as np
import sys
import time
from scipy.spatial import ConvexHull

support = o3d.io.read_triangle_mesh(sys.argv[1])
support_vertices = np.asarray(support.vertices)
suppport_triangle_index = np.asarray(support.triangles)


# 判断点是否在凸包内
def is_point_inside_convex_hull(point, hull):
    # hull.equations 形式为 [A, b]，表示 Ax + b <= 0
    return np.all(np.dot(hull.equations[:, :3], point) + hull.equations[:, 3] <= 0)

def moller_trumbore(ray_origin, ray_dir, v0, v1, v2, epsilon=1e-8):
    """
    使用Möller-Trumbore算法计算直线与三角形的交点
    :param ray_origin: 直线的起点，numpy数组
    :param ray_dir: 直线的方向向量，numpy数组
    :param v0: 三角形的第一个顶点，numpy数组
    :param v1: 三角形的第二个顶点，numpy数组
    :param v2: 三角形的第三个顶点，numpy数组
    :param epsilon: 用于判断平行的阈值，浮点数
    :return: 交点坐标，numpy数组；如果没有交点或交点不在三角形内，返回None
    """
    E1 = v1 - v0
    E2 = v2 - v0
    P = np.cross(ray_dir, E2)
    det = np.dot(E1, P)
    
    if abs(det) < epsilon:
        return None  # 光线与三角形平行

    inv_det = 1.0 / det
    T = ray_origin - v0
    u = np.dot(T, P) * inv_det
    
    if u < 0 or u > 1:
        return None  # 交点不在三角形内

    Q = np.cross(T, E1)
    v = np.dot(ray_dir, Q) * inv_det
    
    if v < 0 or (u + v) > 1:
        return None  # 交点不在三角形内

    t = np.dot(E2, Q) * inv_det
    
    if t < 0:
        return None  # 交点在光线起点的反方向

    intersection_point = ray_origin + t * ray_dir
    return intersection_point

def temp_method():
    for i, tindex in enumerate(suppport_triangle_index):
        if np.all(support_vertices[tindex][:,2] < 0):
            continue
        # hull = ConvexHull(np.concatenate(([[0,0,0]], support_vertices[tindex])))
        # if is_point_inside_convex_hull(test_point, hull):
        if moller_trumbore(np.array([0, 0, 0]), np.array([0, 0, 1]), support_vertices[tindex][0], support_vertices[tindex][1], support_vertices[tindex][2]) is not None:
            print("点在凸包内")
        if i % 10000 == 0 and i != 0:
            print(f"已处理 {i} 个三角形")
            break

def cartesian_to_spherical(points, decimal_places=2):
    """
    将直角坐标系点云转换为球坐标系，保证角度精度
    
    参数:
        points: 点云数组，形状为 (n, 3)，每行为 [x, y, z]
        decimal_places: 角度保留的小数位数
        
    返回:
        spherical_points: 球坐标系点云，形状为 (n, 3)，每行为 [azimuth, elevation, distance]
    """
    x, y, z = points.T
    distance = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.degrees(np.arctan2(y, x)) % 360
    elevation = np.degrees(np.arcsin(np.clip(z / np.where(distance > 1e-10, distance, 1), -1, 1)))
    if decimal_places is not None:
        azimuth = np.round(azimuth, decimal_places)
        elevation = np.round(elevation, decimal_places)
    return np.column_stack((azimuth, elevation, distance))
def cartesian_to_spherical2(points, decimal_places=2):
    """
    将直角坐标系点云转换为球坐标系，保证角度精度
    
    参数:
        points: 点云数组，形状为 (n, 3)，每行为 [x, y, z]
        decimal_places: 角度保留的小数位数
        
    返回:
        spherical_points: 球坐标系点云，形状为 (n, 3)，每行为 [azimuth, elevation, distance]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    np.set_printoptions(threshold=np.inf)
    
    # 计算距离
    distance = np.sqrt(x**2 + y**2 + z**2)
    
    # 计算方位角（水平角）
    azimuth = np.arctan2(y, x)
    # 转换为度数并确保在 [0, 360) 范围内
    azimuth = np.degrees(azimuth)
    azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)
    
    # 计算俯仰角（垂直角）
    # 注意：避免除以零
    elevation = np.zeros_like(distance)
    valid_indices = distance > 1e-10
    # 使用clip避免浮点误差导致的域错误
    normalized_z = np.zeros_like(distance)
    normalized_z[valid_indices] = np.clip(z[valid_indices] / distance[valid_indices], -1.0, 1.0)
    elevation[valid_indices] = np.arcsin(normalized_z[valid_indices])
    elevation = np.degrees(elevation)
    
    # 将角度四舍五入到指定小数位
    if decimal_places is not None:
        azimuth = np.round(azimuth, decimal_places)
        elevation = np.round(elevation, decimal_places)
    # 组合成球坐标系点云
    spherical_points = np.column_stack((azimuth, 90-elevation, distance))
    
    return spherical_points


# 测试点
test_point = np.array([0, 0, 1])  # 替换为你的测试点
# 计算凸包
print(np.all(support_vertices[suppport_triangle_index[0]][:,2]<3))
hull = ConvexHull(np.concatenate(([[0,0,0]], support_vertices[suppport_triangle_index[0]])))
start_time = time.time()
norm_vectices = np.linalg.norm(support_vertices, axis=1)
cos_dis = np.dot(support_vertices, np.array([0,0,1]))
cos_sim = cos_dis / norm_vectices
ray_dir = np.array([0.01,0.01,1])
ray_dir = ray_dir / np.linalg.norm(ray_dir)
ray_dis = np.dot(ray_dir, np.array([0,0,1]))
print(ray_dis)
for i, tindex in enumerate(suppport_triangle_index):
    if np.any(cos_dis[tindex] < 0):
        continue
    spherical_vertices = cartesian_to_spherical(support_vertices[tindex]/np.linalg.norm(support_vertices[tindex], axis=1))
    #print(np.linalg.norm(support_vertices[tindex], axis=1))
    #print(support_vertices[tindex]/np.linalg.norm(support_vertices[tindex], axis=1))
    #print(support_vertices[tindex])
    #print(spherical_vertices)
    elevate_ranges = np.arange(np.floor(np.min(spherical_vertices[:, 1])), np.ceil(np.max(spherical_vertices[:, 1])), 0.1)
    azimuth_ranges = np.arange(np.floor(np.min(spherical_vertices[:, 0])), np.ceil(np.max(spherical_vertices[:, 0])), 0.25)
    #print(azimuth_ranges, elevate_ranges)
    phis = np.radians(elevate_ranges)
    thetas = np.radians(azimuth_ranges)
    ray_dirs = []
    for phi in phis:
        for theta in thetas:
            x = np.cos(phi) * np.cos(theta)
            y = np.cos(phi) * np.sin(theta)
            z = np.sin(phi)
            ray_dirs.append(np.array([x, y, z]))
    #print(len(ray_dirs))
    #print(ray_dirs[0])
    #print(cartesian_to_spherical(np.asarray(ray_dirs))[0])
    ray_indices = []
    for j, ray_dir in enumerate(ray_dirs):
        if moller_trumbore(np.array([0, 0, 0]), ray_dir, support_vertices[tindex][0], support_vertices[tindex][1], support_vertices[tindex][2]) is not None:
            ray_indices.append(j)
    print('ray count', len(ray_indices))
    if i % 10000 == 0 and i != 0:
        print(f"已处理 {i} 个三角形")
        break
# 处理所有三角面大概30秒，
# 每个三角面包含多少条射线？计算三角面每个点的球面坐标，取角度范围，再次计算交点并保存
total_time = time.time() - start_time
print(f"总耗时: {total_time:.2f}秒")