import sys
import numpy as np
import open3d as o3d
from pathlib import Path
import shutil

def transformed_supports_points_comparing(model_file_path, compared_points):
    """
    理论点云的顶板，与雷达点云比较，选择一个距离最小的理论点云
    """
    #simulated_model_files = list(Path(model_file_path).glob("*.pcd"))
    # 创建一个字典，键为(azimuth, elevation)，值为对应的距离值
    radar_points_dict = {}
    for point in compared_points:
        key = (round(point[0], 2), round(point[1], 2)) # 使用舍入值作为键以处理浮点误差
        if key not in radar_points_dict:
            radar_points_dict[key] = []
        radar_points_dict[key].append(point[2])
    
    best_file = ""
    #transformed_file_folder = file.parent / file.stem
    transformed_files = list(Path(model_file_path).glob('*_spherical.pcd'))
    distance_metric = np.inf
    for file in transformed_files:
        print("=", end="")
        transformed_points_cloud = o3d.io.read_point_cloud(str(file))
        spherical_transformed_points = np.asarray(transformed_points_cloud.points)
        # spherical_transformed_points = cartesian_to_spherical(transformed_points)

        this_loop_distance_metric = distance_roof_points(spherical_transformed_points, radar_points_dict)
        """
        for point in spherical_transformed_points:
            mask = (compared_points[:, 0] == point[0]) & (compared_points[:, 1] == point[1])
            if compared_points[mask, 2].size > 0:
                this_loop_distance_metric += np.abs(compared_points[mask, 2][0] - point[2])
        """    
            # print(f'mean distance is {np.mean(this_loop_distance_metric)}')
        if len(this_loop_distance_metric) == 0:
            print(f'no matched points in')
            continue
        mean_distance = np.mean(this_loop_distance_metric)
        if mean_distance < distance_metric:
            distance_metric = mean_distance
            best_file = file
            print(distance_metric)
    print(f' best file {best_file}')
    print(distance_metric)
    print('---------------------------------')
    return str(best_file)
"""
    tp = o3d.io.read_point_cloud(str(best_file))
    file_name = str(best_file.parent.parent) + f"/{str(file.parent.stem)}_transformed.pcd"
    print(file_name)
    o3d.io.write_point_cloud(file_name, tp)
    best_files.append(best_file)
    
    # 存储对应的文件名
    print('write best files to irmatrix.txt')
    matrix_log_file = model_file_path + '/irmatrix.txt'
    with open(matrix_log_file, 'w+') as f:
        data = str(Path(best_file).stem).split('_')
        f.write(f'{data[0]}, {data[3]}, {data[4]}\n')
"""
def distance_roof_points(model_points, radar_points_dict):
    """
    Calculate the cumulative absolute difference in the r-coordinate between corresponding points
    in two sets of 3D points. Corresponding points are determined by matching theta and phi coordinates.

    Args:
        model_points (numpy.ndarray): [theta, phi, r]. 理论点云
        radar_points_dict : {(theta,phi):r} 雷达点云的字典

    Returns:
        float: The cumulative absolute difference in the r-coordinate between corresponding points.
    """
    # Initialize the cumulative distance to 0
    distance = []
    # Iterate through each point in the model_points array
    for point in model_points:
        key = (round(point[0], 2), round(point[1], 2))
        # print(key)
        if key in radar_points_dict:
            # 如果有多个对应点，取最近的一个
            radar_distances = radar_points_dict[key]
            # 计算与所有匹配点的距离差，取最小值
            min_distance_diff = min(abs(point[2] - d) for d in radar_distances)
            distance.append(min_distance_diff)
    return distance


if len(sys.argv) < 2:
    print("Usage: python script.py <arg1> <arg2> ...")

print("Script name:", sys.argv[0])
print("Arguments:")
for i, arg in enumerate(sys.argv[1:], start=1):
    print(f"Argument {i}: {arg}")

model_file_path = sys.argv[1]
compared_points_cloud_spherical = o3d.io.read_point_cloud(sys.argv[2])
compared_points = np.asarray(compared_points_cloud_spherical.points)
best_file = transformed_supports_points_comparing(sys.argv[1], compared_points)
# best_file = 'right_model_transformed_0_0_spherical.pcd'
splitted = str(Path(best_file).stem).split('_')
model_file = model_file_path + f'/{splitted[0]}_{splitted[1]}_{splitted[2]}_{splitted[3]}_{splitted[4]}.pcd'
print(model_file)
shutil.copy(model_file, str(Path(model_file_path).parent)+'/'+str(Path(model_file_path).stem)+'_transformed.pcd')
print(f'{splitted[0]}, {splitted[3]}, {splitted[4]}\n')
# print(str(Path(model_file_path).parent)+str(Path(model_file_path).stem)+'_transformed.pcd')
