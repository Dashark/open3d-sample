import sys
import open3d as o3d
import numpy as np
from pathlib import Path

def rotate_matrix(axis, theta):
    """
    绕任意轴旋转矩阵
    :param axis: 旋转轴向量，如 [u_x, u_y, u_z]
    :param theta: 旋转角度，单位为弧度
    :return: 旋转矩阵
    """
    axis = np.array(axis) / np.linalg.norm(axis)  # 归一化旋转轴
    ux, uy, uz = axis
    I = np.eye(3)
    u_outer = np.outer(axis, axis)
    u_cross = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    return np.cos(theta) * I + (1 - np.cos(theta)) * u_outer + np.sin(theta) * u_cross
def rotate_point_around_vector(point, vector, angle):
    """
    此函数用于计算三维空间中一个点围绕一个向量旋转一定角度后的结果
    :param point: 待旋转的点，三维向量，如 [x, y, z]
    :param vector: 旋转轴向量，三维向量，如 [u_x, u_y, u_z]
    :param angle: 旋转角度，单位为弧度
    :return: 旋转后的点，三维向量，如 [x', y', z']
    """
    # 将输入转换为 numpy 数组
    point = np.array(point)
    vector = np.array(vector)
    # 对旋转轴向量进行归一化处理
    vector = vector / np.linalg.norm(vector)

    # 计算旋转矩阵的各项
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_product = np.cross(vector, point)
    dot_product = np.dot(vector, point)

    # 应用罗德里格斯旋转公式
    rotated_point = (
        point * cos_theta
        + cross_product * sin_theta
        + vector * dot_product * (1 - cos_theta)
    )

    return rotated_point

def azimuth_rotate_matrix(angles):
    # 根据para中的设定的旋转角度范围对指定的文件夹中的存有支架模型顶板pcd文件进行旋转
    # 注意：右手坐标系
    vector_x = [1, 0, 0]
    vector_z = [0, 0, 1]

    first_irmatrix = []
    # 估计理论坐标系到物料坐标系需要旋转的角度，在配置中写好
    for angle in angles:
        # 围绕Z轴计算旋转矩阵
        rmatrix = rotate_matrix(vector_z, np.pi * angle / 180)
        irmatrix = np.linalg.inv(rmatrix)  # 逆矩阵
        first_irmatrix.append(irmatrix)
        # print("同一坐标旋转后的点:", np.dot(irmatrix, vector_x))
    return first_irmatrix
    # 使用 irmatrix 把理论点云的所有点旋转到物料机的坐标系

def elevation_rotate_matrix(azimuthes, elevations):
    # 由于物料机安装不标准需要估计一定的倾斜
    # 在物料机的坐标系下，估计倾斜的水平角和俯仰角
    vector_x = [1, 0, 0]
    vector_z = [0, 0, 1]
    second_irmatrix = []
    for azimuth in azimuthes:
        # X轴绕Z轴旋转得到向量
        azimuth = (azimuth + 90) % 360  # 在XY平面的法向量的角度
        normal_tilt = rotate_point_around_vector(
            vector_x, vector_z, np.pi * azimuth / 180
        )
        # normal_vec = np.cross(normal_tilt, vector_z)  # 法向量
        for elev in elevations:
            rmatrix = rotate_matrix(normal_tilt, np.pi * elev / 180)  # 俯仰角
            irmatrix = np.linalg.inv(rmatrix)
            second_irmatrix.append(irmatrix)
    return second_irmatrix
    # 使用 irmatrix 把理论点云的所有点在变换
def rotate_model(model_points):
    # 给定保存了对应理论支架三个顶板的文件夹
    # 将三个理论支架顶板进行旋转并存储到对应的文件夹中
    first_irmatrix = azimuth_rotate_matrix(np.range(145, 157))
    second_irmatrix = elevation_rotate_matrix(np.range(345, 355), np.range(2, 8))
    for i, first_m in enumerate(first_irmatrix):
        transformed_points = np.dot(model_points, first_m.T)
        for j, second_m in enumerate(second_irmatrix):
            transformed_points = np.dot(transformed_points, second_m.T)
            return transformed_points

if len(sys.argv) < 2:
    print("Usage: python script.py <arg1> <arg2> ...")

print("Script name:", sys.argv[0])
print("Arguments:")
for i, arg in enumerate(sys.argv[1:], start=1):
    print(f"Argument {i}: {arg}")

model_file = sys.argv[1]
pcd = o3d.io.read_point_cloud(model_file)
model_points = np.asarray(pcd.points)
first_irmatrix = azimuth_rotate_matrix(np.arange(140, 150))
second_irmatrix = elevation_rotate_matrix(np.arange(345, 355), np.arange(2, 8))
np.savez_compressed(sys.argv[2] + "/irmatrix.npz", first_irmatrix=first_irmatrix, second_irmatrix=second_irmatrix)
for i, first_m in enumerate(first_irmatrix):
    transformed_points = np.dot(model_points, first_m.T)
    for j, second_m in enumerate(second_irmatrix):
        tilt_points = np.dot(transformed_points, second_m.T)
        if np.any(transformed_points[:, 2] < 0):
            print(f"negative z value in {i} & {j}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tilt_points)
        o3d.io.write_point_cloud(sys.argv[2] + f"/{str(Path(model_file).stem)}_transformed_{i}_{j}.pcd", pcd)
