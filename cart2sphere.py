import sys
import numpy as np
from pathlib import Path
import open3d as o3d

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

def interpolate_spherical_points(spherical_points):
    """
    插值球坐标系点云

    参数:
        spherical_points: 原始球坐标系点云，形状为 (n, 3)，每行为 [azimuth, elevation, distance]
    """
    points_dict = {}
    for point in spherical_points:
        key = (int(point[0]), int(point[1]))
        if key not in points_dict:
            points_dict[key] = []
        points_dict[key].append(point[2])
    append_arr = []
    for key, values in points_dict.items():
        points_dict[key] = np.mean(values)
        append_arr.append([key[0], key[1], np.mean(values)])
    return append_arr

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <arg1> <arg2> ...")
        return

    print("Script name:", sys.argv[0])
    print("Arguments:")
    for i, arg in enumerate(sys.argv[1:], start=1):
        print(f"Argument {i}: {arg}")

    model_file_path = sys.argv[1]
    # model_file_path = "5_bai\\5_bai\\Thu_Apr__3_10_40_37_2025_raw_data\\simulated_model\\left_model"
    simulated_model_files = list(Path(model_file_path).glob("*.pcd"))
    for file in simulated_model_files:
        transformed_points_cloud = o3d.io.read_point_cloud(str(file))
        transformed_points = np.asarray(transformed_points_cloud.points)
        spherical_transformed_points = cartesian_to_spherical2(transformed_points)
        intered = interpolate_spherical_points(spherical_transformed_points)
        print(len(intered), len(spherical_transformed_points))
        spherical_path = str(file.parent) + "/" + str(file.stem) + "_spherical.pcd"
        pcd_spherical = o3d.geometry.PointCloud()
        pcd_spherical.points = o3d.utility.Vector3dVector(np.append(spherical_transformed_points, intered, axis=0))
        o3d.io.write_point_cloud(spherical_path, pcd_spherical)

if __name__ == "__main__":
    main()
