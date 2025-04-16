import sys
import numpy as np
import open3d as o3d


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
    rotated_point = point * cos_theta + cross_product * sin_theta + vector * dot_product * (1 - cos_theta)

    return rotated_point


def rotate_around_arbitrary_axis(vector, axis, theta):
    """
    绕任意轴旋转向量
    :param vector: 三维向量，如 [x, y, z]
    :param axis: 旋转轴向量，如 [u_x, u_y, u_z]
    :param theta: 旋转角度，单位为弧度
    :return: 旋转后的向量
    """
    axis = np.array(axis) / np.linalg.norm(axis)  # 归一化旋转轴
    ux, uy, uz = axis
    I = np.eye(3)
    u_outer = np.outer(axis, axis)
    u_cross = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    rotation_matrix = np.cos(theta) * I + (1 - np.cos(theta)) * u_outer + np.sin(theta) * u_cross
    return np.dot(rotation_matrix, vector)

def rotate_matrix(axis, theta):
    axis = np.array(axis) / np.linalg.norm(axis)  # 归一化旋转轴
    ux, uy, uz = axis
    I = np.eye(3)
    u_outer = np.outer(axis, axis)
    u_cross = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    return np.cos(theta) * I + (1 - np.cos(theta)) * u_outer + np.sin(theta) * u_cross

# 示例使用
vector_x = [1, 0, 0]
vector_z = [0, 0, 1]

angles = np.arange(145, 157)  # 估计理论坐标系到物料坐标系需要旋转的角度，在配置中写好
direct = []
for angle in angles:
  # 围绕Z轴计算旋转矩阵
  rmatrix = rotate_matrix(vector_z, np.pi * angle / 180)
  irmatrix = np.linalg.inv(rmatrix)  # 逆矩阵
  direct.append(np.dot(vector_x, irmatrix.T))
# 使用 irmatrix 把理论点云的所有点旋转到物料机的坐标系

# 由于物料机安装不标准需要估计一定的倾斜
# 在物料机的坐标系下，估计倾斜的水平角和俯仰角
azimuthes = np.arange(345, 355)   # 估计倾斜的水平角，在配置中写好
elevations = np.arange(2,8)    # 估计倾斜的俯仰角，在配置中写好
for azimuth in azimuthes:
    # X轴绕Z轴旋转得到向量
    azimuth = (azimuth + 90) % 360     # 在XY平面的法向量的角度
    normal_tilt = rotate_point_around_vector(vector_x, vector_z, np.pi * azimuth / 180)
    normal_vec = np.cross(normal_tilt, vector_z)  # 法向量
    for elev in elevations:
        rmatrix = rotate_matrix(normal_tilt, np.pi * elev / 180) # 俯仰角
        irmatrix = np.linalg.inv(rmatrix)
        for dir in direct:
            print("同一坐标旋转前的点:", np.dot(dir, irmatrix.T))
# 使用 irmatrix 把理论点云的所有点在变换

irmatrix = np.load(sys.argv[3])
first_irmatrix = irmatrix['first_irmatrix']
second_irmatrix = irmatrix['second_irmatrix']
vector_x = [0, 1, 0]
rotate_x = np.dot(vector_x, first_irmatrix[8].T)
direction = np.dot(rotate_x, second_irmatrix[55].T)
normalized_direction = direction / np.linalg.norm(direction)
move_support = normalized_direction * 1
pcd = o3d.io.read_point_cloud(sys.argv[1])
pcd.translate(move_support)
o3d.io.write_point_cloud(sys.argv[2], pcd)