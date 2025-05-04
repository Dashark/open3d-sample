import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.cluster import DBSCAN
import json
from typing import Counter


def angle2p(N1, N2):
    # Input two normals, return the angle
    dt = N1[0] * N2[0] + N1[1] * N2[1] + N1[2] * N2[2]
    dt = np.arccos(np.clip(dt, -1, 1))
    r_Angle = np.degrees(dt)
    return r_Angle


class RegionGrowing:
    def __init__(self):
        """
        Init parameters
        """
        self.pcd = None  # input point clouds
        self.NPt = 0  # input point clouds
        self.nKnn = 20  # normal estimation using k-neighbour
        self.nRnn = 0.1  # normal estimation using r-neighbour
        self.rKnn = 5  # region growing using k-neighbour
        self.rRnn = 0.1  # region growing using r-neighbour
        self.pcd_tree = None  # build kdtree
        self.TAngle = 5.0
        self.Clusters = []
        self.minCluster = 5  # minimal cluster size

    def SetDataThresholds(self, pc, t_a=10.0):
        self.pcd = pc
        self.TAngle = t_a
        self.NPt = len(self.pcd.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    def RGKnn(self):
        """
        Region growing with KNN-neighbourhood while searching
        return: a list of clusters after region growing
        """
        # Are the normals should be re-estimated?
        if len(self.pcd.normals) < self.NPt:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.nRnn, max_nn=self.nKnn
                )
            )
        processed = np.full(self.NPt, False)
        for i in range(self.NPt):
            if processed[i]:
                continue
            seed_queue = []
            sq_idx = 0
            seed_queue.append(i)
            processed[i] = True
            while sq_idx < len(seed_queue):
                queryPt = self.pcd.points[seed_queue[sq_idx]]
                thisNormal = self.pcd.normals[seed_queue[sq_idx]]
                [k, idx, _] = self.pcd_tree.search_radius_vector_3d(queryPt, self.rRnn)
                idx = idx[1:k]  # indexed point itself
                theseNormals = np.asarray(self.pcd.normals)[idx, :]
                for j in range(len(theseNormals)):
                    if processed[idx[j]]:  # Has this point been processed before ?
                        continue
                    thisA = angle2p(thisNormal, theseNormals[j])
                    if thisA < self.TAngle:
                        seed_queue.append(idx[j])
                        processed[idx[j]] = True
                sq_idx = sq_idx + 1
            if len(seed_queue) > self.minCluster:
                self.Clusters.append(seed_queue)

    def ReLabeles(self):
        # Based on the generated clusters, assign labels to all points
        labels = np.zeros(self.NPt)  # zero = other-clusters
        for i in range(len(self.Clusters)):
            for j in range(len(self.Clusters[i])):
                labels[self.Clusters[i][j]] = i + 1
        return labels


class MatrixSaver:
    """
    A class to save a matrix to a file with a description.

    Attributes:
        matrix (np.array): The matrix to be saved.
        description (str): A description of the matrix. 'vector_innder_product_and_labels' or 'arc_length_and_lables'
        file_name (str): The name of the file to save the matrix to.

    Methods:
        save(): Saves the matrix to a file with the given description.
        set_metadata(): Sets the meatadata to explian the matrix
    """

    def __init__(self, matrix: np.array, description: str, file_name: str):
        self.matrix = matrix
        self.description = description
        self.file_name = file_name
        self.metadata = None

    def save(self):

        self.set_metadata()

        np.savez_compressed(self.file_name, array=self.matrix, metadata=self.metadata)

    def set_metadata(self):
        if self.description == "vector_innder_product_and_labels":
            self.metadata = """
            This file contains a 2D array representing:

            Dimension 1: Points in spherical coordinates and its categorical labels
            Dimension 2: Azimuth Angle | Elevation Angle | Distance | Vector Inner Product Categories Tags

            Units:
            - Points: arbitrary
            - Channels: - Azimuth Angle: Angles
                        - Elevation Angle: Angles
                        - Distance: Meters
                        - Categories: unitless
            """
        if self.description == "arc_length_and_lables":
            self.metadata = """
            This file contains a 2D array representing:

            Dimension 1: Points in spherical coordinates and its categorical labels
            Dimension 2: Azimuth Angle | Elevation Angle | Distance | Arc Length Categories Tags

            Units:
            - Points: arbitrary
            - Channels: - Azimuth Angle: Angles
                        - Elevation Angle: Angles
                        - Distance: Meters
                        - Categories: unitless
            """


class MatrixLoader:
    """
    A class to load a matrix from a file.

    Attributes:
        file_name (str): The name of the file to load the matrix from.

    Methods:
        load_matrix(): Gets the matrix from a file.
        load_metadata(): Gets the information of the matrix from a file.
    """

    def __init__(self, file_name: str):
        self.file_name = file_name

    def get_matrix(self):
        loaded_data = np.load(self.file_name)
        return loaded_data["array"]

    def get_info(self):
        loaded_data = np.load(self.file_name)
        return loaded_data["metadata"]


def find_files_by_extension(folder_path, extension):
    """
    在指定文件夹中查找特定扩展名的文件
    :param folder_path: 文件夹路径
    :param extension: 文件扩展名（如 '.csv', '.pcd'）
    :return: 文件路径列表
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"文件夹 {folder_path} 不存在")
        return []

    if not extension.startswith("."):
        extension = f".{extension}"

    files = list(folder_path.glob(f"*{extension}"))
    files.sort()

    if not files:
        print(f"在 {folder_path} 中没有找到{extension}文件")

    return files


def read_csv_file(file_path):
    """
    读取CSV文件并返回点云数据
    :param file_path: CSV文件路径
    :return: 点云数据列表（跳过标题行）
    """
    with open(file_path, "r") as csvfile:
        lines = csvfile.readlines()
    return [line.strip() for line in lines if line.strip()]


def spherical_to_cartesian(azimuth, elevation, distance):
    """
    将球坐标系转换为直角坐标系
    :param azimuth: 方位角（度）
    :param elevation: 俯仰角（度）
    :param distance: 距离
    :return: (x, y, z) 直角坐标
    """
    azimuth_rad = np.radians(float(azimuth))
    elevation_rad = np.radians(float(elevation))
    distance = float(distance)

    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)

    return x, y, z


def process_points(points_data, use_spherical=False):
    """
    处理CSV点云数据，可选择保持球坐标或转换为直角坐标
    :param points_data: 点数据列表
    :param use_spherical: 是否使用球坐标
    :return: 处理后的点云数组
    """
    processed_points = []
    for point in points_data[1:]:  # 跳过标题行
        values = point.split(",")
        if len(values) >= 3:
            if use_spherical:
                # 直接使用球坐标 [azimuth, elevation, distance]
                processed_points.append(
                    [float(values[0]), float(values[1]), float(values[2])]
                )
            else:
                # 转换为直角坐标 [x, y, z]
                x, y, z = spherical_to_cartesian(values[0], values[1], values[2])
                processed_points.append([x, y, z])

    return np.array(processed_points)


def save_point_cloud(points, output_path):
    """
    保存点云数据为PCD文件
    :param points: 点云数组
    :param output_path: 输出路径
    """
    if len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"保存点云文件: {output_path}")


def segment_by_elevation(points_data, elevation_tolerance=2):
    """
    按俯仰角对点进行分组
    :param points_data: 点云数据数组
    :param elevation_tolerance: 俯仰角容差（度）
    :return: 字典，键为俯仰角，值为对应的点数组
    """
    elevation_groups = {}

    for point in points_data:
        elevation = point[1]  # 对于球坐标，第二个值是俯仰角
        elevation_key = round(elevation / elevation_tolerance) * elevation_tolerance

        if elevation_key not in elevation_groups:
            elevation_groups[elevation_key] = []
        elevation_groups[elevation_key].append(point)

    return {k: np.array(v) for k, v in elevation_groups.items()}


def extract_points_in_spherical_range(points, range_limits):
    """
    在球坐标系中提取指定范围内的点
    :param points: 点云数组（球坐标格式）
    :param range_limits: 范围限制，格式为 {
        'azimuth': [min_azimuth, max_azimuth],
        'elevation': [min_elevation, max_elevation],
        'distance': [min_distance, max_distance]
    }
    :return: 提取的点云数组
    """
    if len(points) == 0:
        return np.array([])

    # 创建每个维度的掩码
    azimuth_mask = (points[:, 0] >= range_limits["azimuth"][0]) & (
        points[:, 0] <= range_limits["azimuth"][1]
    )
    elevation_mask = (points[:, 1] >= range_limits["elevation"][0]) & (
        points[:, 1] <= range_limits["elevation"][1]
    )
    distance_mask = (points[:, 2] >= range_limits["distance"][0]) & (
        points[:, 2] <= range_limits["distance"][1]
    )

    # 组合所有维度的掩码
    combined_mask = azimuth_mask & elevation_mask & distance_mask

    return points[combined_mask]


def save_point_clouds_both_coords(points_spherical, output_path_base):
    """
    同时保存球坐标系和直角坐标系的点云文件
    :param points_spherical: 球坐标系点云数组
    :param output_path_base: 基础输出路径（不包含扩展名）
    """
    if len(points_spherical) > 0:
        # 保存球坐标系点云
        pcd_spherical = o3d.geometry.PointCloud()
        pcd_spherical.points = o3d.utility.Vector3dVector(points_spherical)
        spherical_path = str(output_path_base) + "_spherical.pcd"
        o3d.io.write_point_cloud(spherical_path, pcd_spherical)

        # 转换为直角坐标系并保存
        points_cartesian = np.array(
            [spherical_to_cartesian(p[0], p[1], p[2]) for p in points_spherical]
        )
        pcd_cartesian = o3d.geometry.PointCloud()
        pcd_cartesian.points = o3d.utility.Vector3dVector(points_cartesian)
        cartesian_path = str(output_path_base) + "_cartesian.pcd"
        o3d.io.write_point_cloud(cartesian_path, pcd_cartesian)

        print(f"保存点云文件: {spherical_path}")
        print(f"保存点云文件: {cartesian_path}")


def segment_by_azimuth(points_data, azimuth_tolerance=1.0):
    """
    按方位角对点进行分组
    :param points_data: 点云数据数组
    :param azimuth_tolerance: 方位角容差（度）
    :return: 字典，键为方位角，值为对应的点数组
    """
    azimuth_groups = {}

    for point in points_data:
        azimuth = point[0]  # 对于球坐标，第一个值是方位角
        azimuth_key = round(azimuth / azimuth_tolerance) * azimuth_tolerance

        if azimuth_key not in azimuth_groups:
            azimuth_groups[azimuth_key] = []
        azimuth_groups[azimuth_key].append(point)

    return {k: np.array(v) for k, v in azimuth_groups.items()}


def segment_by_arc_length(
    points_data, method: str, dbscan_eps=0.03, dbscan_min_samples=70
):
    """
    对物料机的点云按照其长度进行初步聚类
    :param points_data: 球坐标的点云
    :param method: "dbscan" | "common_length"
    :dbscan_eps: dbscan的eps
    :dbscan_min_samples: dbscan的最少聚类数量
    :return 分类标签以及对应的点云、 分类的标签对应的mask
    """
    if method == "dbscan":
        arc_length = points_data[:, [2]]
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster = dbscan.fit(arc_length)
        total = {}
        labels = np.unique(cluster.labels_)
        for label in labels:
            if label == -1:
                continue
            mask = cluster.labels_ == label
            total[arc_length[mask].mean()] = points_data[mask]

        return total, cluster.labels_

    if method == "common_length":

        arc_length = np.round(points_data[:, 2], 3)
        cnt = {}
        for i, distance in enumerate(arc_length):
            if distance not in cnt:
                cnt[distance] = [i]
            else:
                cnt[distance].append(i)

        all_mask = np.zeros(len(arc_length))
        for i, (k, v) in enumerate(cnt.items()):
            all_mask[v] = i

        total = {}
        labels = np.unique(all_mask)
        for label in labels:
            if label == -1:
                continue
            mask = all_mask == label
            total[arc_length[mask].mean()] = points_data[mask]

        return total, all_mask


def find_min_interval(data_labels):
    """
    找到一个最小长度的角度区间，使其包含圆环上所有有数据的块

    参数:
        data_flags: 长度为360的布尔列表，表示每个角度是否有数据

    返回:
        区间表示: 如果跨越0度，返回[(start, 359), (0, end)]；否则返回(start, end)
    """
    # 收集所有有数据的角度索引
    data_indices = np.unique(data_labels)

    # if not data_indices:
    #     return None  # 如果没有数据，返回None

    if len(data_indices) == 360:
        return [0, 359]  # 如果所有角度都有数据，返回完整圆环

    # 使用更高效的算法：计算相邻数据点之间的最大间隔
    n = len(data_indices)
    max_gap = 0
    gap_start = 0

    # 检查所有相邻点之间的间隔
    for i in range(n):
        next_i = (i + 1) % n
        current_gap = (data_indices[next_i] - data_indices[i]) % 360
        if current_gap > max_gap:
            max_gap = current_gap
            gap_start = i

    # 最小区间的起点是最大间隔之后的点
    best_start = data_indices[(gap_start + 1) % n]
    # 最小区间的终点是最大间隔之前的点
    best_end = data_indices[gap_start]

    # 处理特殊情况：如果区间跨越0度
    # if best_end < best_start:
    #     return [[best_start, 359], [0, best_end]]
    return [best_start, best_end]


def process_csv_files(folder_path, angel_tolerance, dbscan_param):
    """
    处理文件夹中的所有CSV文件
    :param folder_path: 文件夹路径
    :param angel_tolerance : (elevation_tolerance: 俯仰角容差（度）, azimuth_tolerance: 方位角容差（度）)
    :param dbscan_param: 如果使用dbscan方式计算半径聚类,对应(distance_dbscan_eps, distance_dbscan_min_samples)
    """
    folder_path = Path(folder_path)
    csv_files = find_files_by_extension(folder_path, ".csv")

    if not csv_files:
        return

    # 用于存储所有点云数据的列表
    all_points = []

    # 处理每个CSV文件
    for csv_file in csv_files:
        # print(f"\n处理文件: {csv_file.name}")

        # 读取CSV文件
        points_data = read_csv_file(csv_file)

        # 处理点云数据（保持球坐标）
        processed_points = process_points(points_data, use_spherical=True)

        # 添加到总体点云
        all_points.extend(processed_points)

        # 创建对应的文件夹
        output_folder = folder_path / csv_file.stem
        output_folder.mkdir(exist_ok=True)

        # 保存原始点云
        pcd_output_file = str(output_folder)
        save_point_clouds_both_coords(processed_points, pcd_output_file)

        # 创建俯仰角分组文件夹
        elevation_folder = output_folder / "elevation_groups"
        elevation_folder.mkdir(exist_ok=True)

        # 按俯仰角分割并保存
        elevation_groups = segment_by_elevation(processed_points, angel_tolerance[0])
        for elevation, points in elevation_groups.items():
            output_path_base = elevation_folder / f"elevation_{elevation:.1f}"
            save_point_clouds_both_coords(points, output_path_base)

        # 创建方位角分组文件夹
        azimuth_folder = output_folder / "azimuth_groups"
        azimuth_folder.mkdir(exist_ok=True)

        # 按方位角分割并保存
        azimuth_groups = segment_by_azimuth(processed_points, angel_tolerance[1])
        for azimuth, points in azimuth_groups.items():
            output_path_base = azimuth_folder / f"azimuth_{azimuth:.1f}"
            save_point_clouds_both_coords(points, output_path_base)

        # 创建弧长分组文件夹
        arc_length_folder = output_folder / "arc_length_groups"
        arc_length_folder.mkdir(exist_ok=True)

        # 按弧长分割并保存
        arc_groups, arc_length_labels = segment_by_arc_length(
            processed_points, "common_length", dbscan_param[0], dbscan_param[1]
        )
        for arc, points in arc_groups.items():
            # 将没有标签的点存为点云
            output_path_base = arc_length_folder / f"arc_{arc:.3f}"
            save_point_clouds_both_coords(points, output_path_base)

        # 将带有标签的点存下来
        arc_groups_with_lables_path = (
            str((arc_length_folder / "arc")) + "_spherical.npz"
        )
        arc_group_points_with_lables = np.concatenate(
            [processed_points, arc_length_labels.reshape((-1, 1))], axis=1
        )
        MS = MatrixSaver(
            arc_group_points_with_lables,
            "arc_length_and_lables",
            arc_groups_with_lables_path,
        )
        MS.save()

    # 处理合并后的点云
    if all_points:
        all_points = np.array(all_points)

        # 创建并保存总体点云文件
        combined_base = folder_path / "combined_cloud"
        save_point_clouds_both_coords(all_points, combined_base)

        # 创建总体点云的分组文件夹
        combined_folder = folder_path / "combined_cloud"
        combined_folder.mkdir(exist_ok=True)

        # 创建总体点云的俯仰角分组文件夹
        combined_elevation_folder = combined_folder / "elevation_groups"
        combined_elevation_folder.mkdir(exist_ok=True)

        # 对总体点云进行俯仰角分割
        elevation_groups = segment_by_elevation(all_points, angel_tolerance[0])
        for elevation, points in elevation_groups.items():
            output_path_base = combined_elevation_folder / f"elevation_{elevation:.1f}"
            save_point_clouds_both_coords(points, output_path_base)

        # 创建总体点云的方位角分组文件夹
        combined_azimuth_folder = combined_folder / "azimuth_groups"
        combined_azimuth_folder.mkdir(exist_ok=True)

        # 对总体点云进行方位角分割
        azimuth_groups = segment_by_azimuth(all_points, angel_tolerance[1])
        for azimuth, points in azimuth_groups.items():
            output_path_base = combined_azimuth_folder / f"azimuth_{azimuth:.1f}"
            save_point_clouds_both_coords(points, output_path_base)

        # 创建总体点云的弧长分组文件夹
        combined_arc_length_folder = combined_folder / "arc_length_groups"
        combined_arc_length_folder.mkdir(exist_ok=True)

        # 对总体点云进行弧长分割
        arc_groups, arc_length_labels = segment_by_arc_length(
            all_points, "dbscan", dbscan_param[0], dbscan_param[1]
        )
        for arc, points in arc_groups.items():
            output_path_base = combined_arc_length_folder / f"arc_{arc:.3f}"
            save_point_clouds_both_coords(points, output_path_base)


def get_the_file(folder_path, pattern="arc_*_spherical.pcd"):
    """
    获取指定文件夹下的所有相关文件。

    参数:
    folder_path (str): 文件夹路径

    pattern (str): 查询文件格式

    返回:
    pcd_files (list): 包含所有相关文件的列表
    """

    folder_path = Path(folder_path)

    # 获取所有相关文件
    pcd_files = list(folder_path.glob(pattern))

    return pcd_files


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


def cartesian_to_spherical(points, decimal_places=2):
    """
    将直角坐标系点云转换为球坐标系，保证角度精度
    
    参数:
        points: 点云数组，形状为 (n, 3)，每行为 [x, y, z]
        decimal_places: 角度保留的小数位数
        
    返回:
        spherical_points: 球坐标系点云，形状为 (n, 3)，每行为 [azimuth, elevation, distance]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
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
    spherical_points = np.column_stack((azimuth, elevation, distance))
    
    return spherical_points


def transformed_supports_points_comparing(model_file_path, radar_file_path):
    """
    理论点云的顶板，与雷达点云比较，选择一个距离最小的理论点云
    """
    simulated_model_files = list(Path(model_file_path).glob("*.pcd"))
    compared_points_cloud_spherical = o3d.io.read_point_cloud(radar_file_path)
    compared_points = np.asarray(compared_points_cloud_spherical.points)
    
    best_files = []
    for file in simulated_model_files:
        transformed_file_folder = file.parent / file.stem
        transformed_files = list(transformed_file_folder.glob('*_spherical.pcd'))
        distance_metric = np.inf
        for file in transformed_files:
            transformed_points_cloud = o3d.io.read_point_cloud(str(file))
            transformed_points = np.asarray(transformed_points_cloud.points)
            spherical_transformed_points = cartesian_to_spherical(transformed_points)
            
            this_loop_distance_metric = 0
            for point in spherical_transformed_points:
                mask = (compared_points[:, 0] == point[0]) & (compared_points[:, 1] == point[1])
                if compared_points[mask, 2].size > 0:
                    this_loop_distance_metric += np.abs(compared_points[mask, 2][0] - point[2])
            
            # print(f'mean distance is {np.mean(this_loop_distance_metric)}')
            if np.mean(this_loop_distance_metric) < distance_metric:
                distance_metric = np.mean(this_loop_distance_metric)
                best_file = file
        print(f'{str(transformed_file_folder)}\'s best file {best_file}')
        print(distance_metric)
        print('---------------------------------')
        tp = o3d.io.read_point_cloud(str(best_file))
        file_name = str(best_file.parent.parent) + f"/{str(file.parent.stem)}_transformed.pcd"
        print(file_name)
        o3d.io.write_point_cloud(file_name, tp)
        best_files.append(best_file)
    
    # 存储对应的文件名
    print('write best files to irmatrix.txt')
    matrix_log_file = model_file_path + '/irmatrix.txt'
    with open(matrix_log_file, 'w') as f:
        for best_file in best_files:
            data = str(Path(best_file).stem).split('_')
            f.write(f'{data[0]}, {data[3]}, {data[4]}\n')
    
        
def preprocess_points(file_folder: str):
    with open("para.json", "r") as json_file:
        config = json.load(json_file)

    # ************************************************************
    # 按照俯仰角分，计算其法向量并保存于俯仰角直角坐标系文件中
    elevation_file_folder = config.get("elevation_groups")
    elevation_cartesian_pcd_files = get_the_file(
        elevation_file_folder, "elevation_*_cartesian.pcd"
    )
    for file in elevation_cartesian_pcd_files:

        arc_length = float(file.stem.split("_")[1])
        # 读取直角坐标点云,计算其点云并保存
        pcd = o3d.io.read_point_cloud(str(file))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        o3d.io.write_point_cloud(str(file), pcd)
    # ************************************************************

    # ************************************************************
    # 按照水平角分类，计算其法向量并保存于水平角直角坐标系文件中
    azimuth_file_folder = config.get("azimuth_groups")
    azimuth_cartesian_pcd_files = get_the_file(
        azimuth_file_folder, "azimuth_*_cartesian.pcd"
    )
    # 根据水平角度找到半环和其文件的索引
    azimuth_index = {}
    for i, file in enumerate(azimuth_cartesian_pcd_files):
        arc_length = int(float(file.stem.split("_")[1])) % 180
        if azimuth_index.get(arc_length) is None:
            azimuth_index[arc_length] = [i]
        else:
            azimuth_index[arc_length].append(i)
    azimuth_index = sorted(azimuth_index.items(), key=lambda x: x[0], reverse=False)

    # 将20度作为一个小组，对其中的点云进行合并后计算其中的法向量
    interval = 20
    for i in range(0, 180, interval):
        this_interval_points = []

        for j in range(interval):
            for k in azimuth_index[i + j][1]:
                pcd = o3d.io.read_point_cloud(str(azimuth_cartesian_pcd_files[k]))
                p = np.asarray(pcd.points)
                this_interval_points.append(p)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.concatenate(this_interval_points, axis=0)
        )
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        azimuth_file_path = (
            azimuth_file_folder + f"/azimuth_angle{i}to{i+19}_with_normals.pcd"
        )
        o3d.io.write_point_cloud(azimuth_file_path, pcd)
    # ************************************************************

    # ************************************************************
    # 读取按弧长分组的点云文件夹路径
    arc_length_file_folder = config.get("arc_length_groups")
    spherical_pcd_files = get_the_file(arc_length_file_folder)

    all_spherical_points = []
    all_normals = []
    for file in spherical_pcd_files:
        # 跳过距离为0的文件
        arc_length = float(file.stem.split("_")[1])
        if arc_length == 0:
            continue

        # 读取球坐标点云,并添加到all_points中
        spherical_pcd = o3d.io.read_point_cloud(str(file))
        spherical_points = np.asarray(spherical_pcd.points)
        all_spherical_points.append(spherical_points)

        # 将球坐标转到直角坐标，并计算法向量、添加到all_normals中
        points = np.array(
            [spherical_to_cartesian(*point) for point in spherical_points]
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        normals = np.asarray(pcd.normals)
        all_normals.append(normals)
    # ************************************************************
    normals = np.concatenate(all_normals, axis=0)
    spherical_points = np.concatenate(all_spherical_points, axis=0)

    # 按照物料机的参数设定的矩阵大小，其中忽略了俯仰角为90的部分
    spherical_shape = 3
    normal_shape = 3
    col = spherical_shape + normal_shape
    azimuth_num = 361
    elevation_num = 45
    row = azimuth_num * elevation_num

    spherical_table = np.zeros((row, col))
    for i in range(0, elevation_num * 2, 2):
        for j in range(azimuth_num):
            index = int(i * 361 / 2) + j
            spherical_table[index][1] = int(i)
            spherical_table[index][0] = int(j)

    for i, (azimuth, elevation, distance) in enumerate(spherical_points):
        if elevation == 90:
            continue
        spherical_table[int(361 * elevation / 2 + azimuth)][2] = distance
        spherical_table[int(361 * elevation / 2 + azimuth)][3:6] = normals[i]

    labels_matrix = np.dot(spherical_table[:, 3:6], spherical_table[:, 3:6].T)

    labels = np.zeros(labels_matrix.shape[0]) - 1
    lable = 0
    for i in range(labels_matrix.shape[0]):
        if labels_matrix[0 + i][-1 + i] > 0.999:
            labels[i] = lable
            labels[i - 1] = lable
        elif labels_matrix[0 + i][-1 + i] == 0:
            continue
        else:
            lable += 1
    # print(Counter(labels))
    # unique_labels = np.unique(labels)
    # n_labels = len(unique_labels)
    spherical_table = np.concatenate([spherical_table, labels.reshape((-1, 1))], axis=1)
    p = Path(file_folder)

    MS = MatrixSaver(
        spherical_table, "spherical_table", str(p.parent) + "/spherical_table.npz"
    )
    MS.save()

    # ************************************************************
    valid_value_mask = labels > -1
    valid_cartesian_points = np.array(
        [spherical_to_cartesian(*point) for point in spherical_table[:, :3]]
    )[valid_value_mask]

    # 整体均值和标准差
    m = np.mean(valid_cartesian_points, axis=0)
    v = np.std(valid_cartesian_points, axis=0)
    # print(f'整体均值为{m}, 整体方差为{v}')
    x_threshold = config.get("first_std_x_threshold") * v[0]
    y_threshold = config.get("first_std_y_threshold") * v[1]
    z_threshold = config.get("first_std_z_threshold") * v[2]

    first_mask = (
        (valid_cartesian_points[:, 0] >= (m[0] - x_threshold))
        & (valid_cartesian_points[:, 0] <= (m[0] + x_threshold))
        & (valid_cartesian_points[:, 1] >= (m[1] - y_threshold))
        & (valid_cartesian_points[:, 1] <= (m[1] + y_threshold))
        & (valid_cartesian_points[:, 2] >= (m[2] - z_threshold))
        & (valid_cartesian_points[:, 2] <= (m[2] + z_threshold))
    )
    # ------------------------------------------------------重复部分
    valid_cartesian_points = valid_cartesian_points[first_mask]
    m = np.mean(valid_cartesian_points, axis=0)
    v = np.std(valid_cartesian_points, axis=0)
    # print(f'整体均值为{m}, 整体方差为{v}')
    x_threshold = config.get("seconed_std_x_threshold") * v[0]
    y_threshold = config.get("seconed_std_y_threshold") * v[1]
    z_threshold = config.get("seconed_std_z_threshold") * v[2]

    second_mask = (
        (valid_cartesian_points[:, 0] >= (m[0] - x_threshold))
        & (valid_cartesian_points[:, 0] <= (m[0] + x_threshold))
        &
        # (valid_cartesian_points[:, 1] >= (m[1] - y_threshold)) & (valid_cartesian_points[:, 1] <= (m[1] + y_threshold)) &
        (valid_cartesian_points[:, 2] >= (m[2] - z_threshold))
        & (valid_cartesian_points[:, 2] <= (m[2] + z_threshold))
    )

    valid_cartesian_points = valid_cartesian_points[second_mask]
    m = np.mean(valid_cartesian_points, axis=0)
    v = np.std(valid_cartesian_points, axis=0)
    print(f"整体均值为{m}, 整体方差为{v}")

    index_in_second_mask = np.where(second_mask)[0]
    index_in_first_mask = np.where(first_mask)[0][index_in_second_mask]
    index_in_valid_value_mask = np.where(valid_value_mask)[0][index_in_first_mask]

    labels = np.zeros(labels_matrix.shape[0])
    for i in index_in_valid_value_mask:
        labels[i] = 1
    spherical_table = np.concatenate([spherical_table, labels.reshape((-1, 1))], axis=1)
    print(spherical_table.shape)
    MS = MatrixSaver(
        spherical_table, "spherical_table", str(p.parent) + "/spherical_table.npz"
    )
    MS.save()
    dingban_pcd = o3d.geometry.PointCloud()
    dingban_pcd.points = o3d.utility.Vector3dVector(valid_cartesian_points)
    o3d.io.write_point_cloud(str(p.parent) + "/dingban.pcd", dingban_pcd)

    # ************************************************************

    cartesian_pcd_files = get_the_file(arc_length_file_folder, "arc_*_spherical.pcd")

    all_spherical_points = []
    areas = []

    for file in cartesian_pcd_files:
        # print(file)
        # 跳过距离为0的文件
        arc_length = float(file.stem.split("_")[1])
        if arc_length == 0:
            continue

        spherical_pcd = o3d.io.read_point_cloud(str(file))
        spherical_points = np.asarray(spherical_pcd.points)
        all_spherical_points.append(spherical_points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            [
                spherical_to_cartesian(*point)
                for point in np.asarray(spherical_pcd.points)
            ]
        )
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )

        RGKNN = RegionGrowing()
        RGKNN.SetDataThresholds(
            pcd, 2.0
        )  # the growing angle threshold is set to 10.0 degree
        RGKNN.RGKnn()  # Run region growing
        labels = RGKNN.ReLabeles()

        # spherical_points = np.concatenate([spherical_points, labels.reshape(-1,1)], axis=1)
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == 0:
                continue
            label_mask = labels == label
            label_spherical_points = spherical_points[label_mask]
            elevation_range = [
                np.min(label_spherical_points[:, 1]),
                np.max(label_spherical_points[:, 1]),
            ]
            azimuth_range = find_min_interval(label_spherical_points[:, 0])
            areas.append([azimuth_range, elevation_range])

    all_spherical_points = np.concatenate(all_spherical_points, axis=0)
    all_label_masks = []
    for this_area in areas:

        azimuth_range = this_area[0]
        # 处理跨越了0度情况
        if azimuth_range[0] > azimuth_range[1]:
            this_azimuth_mask = (
                (all_spherical_points[:, 0] >= azimuth_range[0])
                & (all_spherical_points[:, 0] <= 360)
            ) | (
                (all_spherical_points[:, 0] >= 0)
                & (all_spherical_points[:, 0] <= azimuth_range[1])
            )
        else:
            this_azimuth_mask = (all_spherical_points[:, 0] >= azimuth_range[0]) & (
                all_spherical_points[:, 0] <= azimuth_range[1]
            )

        elevation_range = this_area[1]
        this_elevation_mask = (all_spherical_points[:, 1] >= elevation_range[0]) & (
            all_spherical_points[:, 1] <= elevation_range[1]
        )
        this_area_mask = this_azimuth_mask & this_elevation_mask
        all_label_masks.append(this_area_mask)

    # 创建一个保存分类后的点的文件夹
    arc_length_file_folder = Path(arc_length_file_folder)
    arc_classify_folder = arc_length_file_folder / "arc_classify"
    arc_classify_folder.mkdir(exist_ok=True)

    # 将areas保存下来
    areas = np.array(areas)
    np.savez(str(arc_classify_folder) + "/areas", areas)

    for i in range(len(all_label_masks)):
        points = all_spherical_points[all_label_masks[i]]
        pcd = o3d.geometry.PointCloud()
        # 保存直角坐标
        pcd.points = o3d.utility.Vector3dVector(
            [spherical_to_cartesian(*point) for point in points]
        )
        o3d.io.write_point_cloud(
            str(arc_classify_folder)
            + "/azimuth{a}to{b}_elevation{c}to{d}_cartesian.pcd".format(
                a=areas[i][0][0], b=areas[i][0][1], c=areas[i][1][0], d=areas[i][1][1]
            ),
            pcd,
        )
        # 保存球坐标
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(
            str(arc_classify_folder)
            + "/azimuth{a}to{b}_elevation{c}to{d}_spherical.pcd".format(
                a=areas[i][0][0], b=areas[i][0][1], c=areas[i][1][0], d=areas[i][1][1]
            ),
            pcd,
        )

    fulled_label = np.zeros(all_spherical_points.shape[0])

    for i in range(len(all_label_masks)):
        for j in range(len(all_label_masks[i])):
            if all_label_masks[i][j]:
                fulled_label[j] = i + 1
    # print(Counter(fulled_label))
    cartesian_points = [
        spherical_to_cartesian(*point) for point in all_spherical_points
    ]
    cartesian_points = np.concatenate(
        [np.array(cartesian_points), fulled_label.reshape(-1, 1)], axis=1
    )

    num_points = cartesian_points.shape[0]
    header = f"""# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z label
    SIZE 4 4 4 4
    TYPE F F F F
    COUNT 1 1 1 1
    WIDTH {num_points}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {num_points}
    DATA ascii
    """
    pcd_data = "\n".join(
        ["{:.6f} {:.6f} {:.6f} {:.0f}".format(*p) for p in cartesian_points]
    )
    # 保存 PCD 文件
    with open(str(arc_length_file_folder / "all_points_with_arc_lables.pcd"), "w") as f:
        f.write(header + pcd_data)

    # 将arc_claddify中的文件根据每个俯仰角的统计生成新的文件存于new_arc_classify文件夹下
    path_folder = config.get("arc_length_groups") + "/arc_classify"
    path_folder = Path(path_folder)

    new_path_folder = path_folder.parent / "new_arc_classify"
    new_path_folder.mkdir(exist_ok=True)

    pcd_files = path_folder.glob("azimuth*_spherical.pcd")
    for file in pcd_files:
        # print(file)
        pcd = o3d.io.read_point_cloud(str(file))
        points = np.asarray(pcd.points)
        # 按照俯仰角分理处
        unique_elevation_labels = np.unique(points[:, 1])
        for label in unique_elevation_labels:
            indices = np.where(points[:, 1] == label)[0]
            arc_cnt = Counter(np.round(points[indices, 2], 3))
            arc_length = arc_cnt.most_common()[0][0]
            points[indices, 2] = arc_length

        pcd.points = o3d.utility.Vector3dVector(points)
        file_name = new_path_folder / file.stem

        o3d.io.write_point_cloud(str(file_name) + ".pcd", pcd)

        cartesian_points = np.array(
            [spherical_to_cartesian(*point) for point in points]
        )
        pcd.points = o3d.utility.Vector3dVector(cartesian_points)
        cartesian_file_name = (
            str(file.stem).split("_")[0]
            + "_"
            + str(file.stem).split("_")[1]
            + "_cartesian"
        )
        file_name = str(new_path_folder) + "/" + cartesian_file_name
        # print(file_name)
        o3d.io.write_point_cloud(file_name + ".pcd", pcd)

    new_arc_classify_files = new_path_folder.glob("azimuth*_cartesian.pcd")
    all_points = []
    for file in new_arc_classify_files:
        pcd = o3d.io.read_point_cloud(str(file))
        points = np.asarray(pcd.points)
        all_points.append(points)

    # pcd_points = np.array(all_points)
    # print(len(all_points))
    all_points = np.concatenate(all_points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    o3d.io.write_point_cloud(str(new_path_folder / "all_points.pcd"), pcd)


    # 根据para中的设定的旋转角度范围对指定的文件夹中的存有支架模型顶板pcd文件进行旋转
    # 注意：右手坐标系
    vector_x = [1, 0, 0]
    vector_z = [0, 0, 1]

    first_irmatrix = []
    angles_range = config.get("angles_range")
    angles = np.arange(
        *angles_range
    )  # 估计理论坐标系到物料坐标系需要旋转的角度，在配置中写好
    for angle in angles:
        # 围绕Z轴计算旋转矩阵
        rmatrix = rotate_matrix(vector_z, np.pi * angle / 180)
        irmatrix = np.linalg.inv(rmatrix)  # 逆矩阵
        first_irmatrix.append(irmatrix)
        print("同一坐标旋转后的点:", np.dot(irmatrix, vector_x))

    # 使用 irmatrix 把理论点云的所有点旋转到物料机的坐标系

    # 由于物料机安装不标准需要估计一定的倾斜
    # 在物料机的坐标系下，估计倾斜的水平角和俯仰角
    azimuthes_range = config.get("azimuthes_range")
    azimuthes = np.arange(*azimuthes_range)  # 估计倾斜的水平角，在配置中写好
    elevations_range = config.get("elevations_range")
    elevations = np.arange(*elevations_range)  # 估计倾斜的俯仰角，在配置中写好
    second_irmatrix = []
    for azimuth in azimuthes:
        # X轴绕Z轴旋转得到向量
        azimuth = (azimuth + 90) % 360  # 在XY平面的法向量的角度
        normal_tilt = rotate_point_around_vector(
            vector_x, vector_z, np.pi * azimuth / 180
        )
        normal_vec = np.cross(normal_tilt, vector_z)  # 法向量
        for elev in elevations:
            rmatrix = rotate_matrix(normal_tilt, np.pi * elev / 180)  # 俯仰角
            irmatrix = np.linalg.inv(rmatrix)
            second_irmatrix.append(irmatrix)
    # 使用 irmatrix 把理论点云的所有点在变换

    # 给定保存了对应理论支架三个顶板的文件夹
    simulated_model_folder = config.get("simulated_model_folder")
    simulated_model_files = list(Path(simulated_model_folder).glob("*.pcd"))
    for file in simulated_model_files:
        folder = file.parent / file.stem
        folder.mkdir(exist_ok=True)
        cartesian_simulated_semicircles = o3d.io.read_point_cloud(str(file))
        cartesian_simulated_semicircles_points = np.asarray(
            cartesian_simulated_semicircles.points
        )
        # 将三个理论支架顶板进行旋转并存储到对应的文件夹中
        for i, first_m in enumerate(first_irmatrix):
            for j, second_m in enumerate(second_irmatrix):
                transformed_points = np.dot(
                    cartesian_simulated_semicircles_points, second_m.T
                )
                transformed_points = np.dot(transformed_points, first_m.T)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(transformed_points)
                o3d.io.write_point_cloud(
                    str(folder) + f"/{str(file.stem)}_transformed_{i}_{j}.pcd", pcd
                )


def main():
    """
    主函数
    """

    """
    根据json文件param_file设定初步得到顶板目标需要的参数
    :param param_file:
                        "process_csv_files": 0, 是否将csv文件转为pcd
                        "csv2pcd_file_folder_path": "folder_path", 如果不与该文件在同一级或者下级路径中需要给出绝对路径
                        "azimuth_tolerance": 1, 水平角的步进角度
                        "elevation_tolerance": 2, 俯仰角的步进角度
                        "distance_dbscan_eps": 0.03, 初次按照点云中距离进行dbscan分类的参数
                        "distance_dbscan_min_samples": 70, 初次按照点云中距离进行dbscan分类的参数
                        "first_std_x_threshold": 3, 第一次计算均值滤波时x轴方向的阈值
                        "first_std_y_threshold": 1.5, --y轴方向--
                        "first_std_z_threshold": 1, --z轴方向--
                        "seconed_std_x_threshold": 3, 第二次--x轴方向--
                        "seconed_std_y_threshold": 1.75, --y轴方向--
                        "seconed_std_z_threshold": 1 --z轴方向--
                        "preprocess_points" : 0, 
                        "arc_length_groups": "zhijiadibu/Mon_Feb_10_14_38_40_2025_raw_data/arc_length_groups"
                        "elevation_groups": "zhijiadibu/Mon_Feb_10_14_38_40_2025_raw_data/elevation_groups",
                        "azimuth_groups" : "zhijiadibu/Mon_Feb_10_14_38_40_2025_raw_data/azimuth_groups",
                        "angles_range" : [235, 247],
                        "azimuthes_range" : [345, 355],
                        "elevations_range" : [2, 8],
                        "simulated_model_folder" : "H:/simulated_model"
    """
    # 从配置文件中读取相关参数
    with open("para.json", "r") as json_file:
        config = json.load(json_file)

    # 设置参数
    folder_path = config.get("csv2pcd_file_folder_path", "")

    elevation_tolerance = config.get("elevation_tolerance", 2)
    azimuth_tolerance = config.get("azimuth_tolerance", 1.0)
    angel_tolerance = (elevation_tolerance, azimuth_tolerance)

    distance_dbscan_eps = config.get("distance_dbscan_eps", 0.03)
    distance_dbscan_min_samples = config.get("distance_dbscan_min_samples", 70)
    dbscan_param = (distance_dbscan_eps, distance_dbscan_min_samples)
    # 处理文件夹
    if config.get("process_csv_files"):
        process_csv_files(folder_path, angel_tolerance, dbscan_param)

    if config.get("preprocess_points"):
        preprocess_points(config.get("arc_length_groups", ""))
        
    if config.get("transformed_supports_points_comparing"):
        model_file_path = config.get("simulated_model_folder", "")
        radar_file_name = str(Path(config.get("arc_length_groups")).parent.stem) + '_spherical.pcd'
        radar_file_name = Path(config.get("arc_length_groups")).parent.parent / radar_file_name
        transformed_supports_points_comparing(model_file_path, str(radar_file_name))
    

    simulated_model_folder = config.get("simulated_model_folder")
    simulated_semicircles_folder = Path(simulated_model_folder)
    files = list(simulated_semicircles_folder.glob('*transformed.pcd'))

    areas_file_path = Path(config.get('arc_length_groups')) / 'arc_classify/areas.npz'
    areas = np.load(str(areas_file_path))['arr_0']

    # areas_file_folder = Path(areas_file_path).parent.parent / 'simulated_semicircles'
    # areas_file_folder.mkdir(exist_ok=True)

    for file in files:
        # 读取点云文件
        pcd = o3d.io.read_point_cloud(str(file))
        all_spherical_points = cartesian_to_spherical(np.asarray(pcd.points))
        # 创建对应的变换后的文件夹
        folder = file.parent / file.stem
        folder.mkdir(exist_ok=True)

        for this_area in areas:
            azimuth_range = this_area[0]
            # 处理跨越了0度情况
            if azimuth_range[0] > azimuth_range[1]:
                this_azimuth_mask = (
                    (all_spherical_points[:, 0] >= azimuth_range[0]) & (all_spherical_points[:, 0] <= 360) 
                    ) | (
                    (all_spherical_points[:, 0] >= 0) & (all_spherical_points[:, 0] <= azimuth_range[1])
                    )
            else:
                this_azimuth_mask = (
                    (all_spherical_points[:, 0] >= azimuth_range[0])
                    & (all_spherical_points[:, 0] <= azimuth_range[1])
                )
            
            elevation_range = this_area[1]
            this_elevation_mask = (
                (all_spherical_points[:, 1] >= elevation_range[0])
                & (all_spherical_points[:, 1] <= elevation_range[1])
            )
            this_area_mask = (
                this_azimuth_mask & this_elevation_mask
            )
            this_area_points = all_spherical_points[this_area_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector([spherical_to_cartesian(*this_area_point) for this_area_point in this_area_points])
            o3d.io.write_point_cloud(
                str(folder) + '/azimuth{a}to{b}_elevation{c}to{d}_cartesian.pcd'.format(
                    a=azimuth_range[0], b=azimuth_range[1], c=elevation_range[0], d=elevation_range[1]
                ),
                pcd,
            )
            pcd.points = o3d.utility.Vector3dVector(this_area_points)
            o3d.io.write_point_cloud(
                str(folder) + '/azimuth{a}to{b}_elevation{c}to{d}_spherical.pcd'.format(
                    a=azimuth_range[0], b=azimuth_range[1], c=elevation_range[0], d=elevation_range[1]
                ),
                pcd,
            )
    # # 示例：在球坐标系中提取指定范围的点
    # range_limits = {
    #     "azimuth": [-0.092 - 0.1256, -0.092 + 0.1256],
    #     "elevation": [0.078 - 0.3, 0.078 + 0.3],
    #     "distance": [1.56 - 0.0435, 1.56 + 0.0435],
    # }

    # # 读取合并后的点云文件进行范围提取
    # combined_cloud_path = Path(folder_path) / "combined_cloud_spherical.pcd"
    # if combined_cloud_path.exists():
    #     pcd = o3d.io.read_point_cloud(str(combined_cloud_path))
    #     points = np.asarray(pcd.points)
    #     filtered_points = extract_points_in_spherical_range(points, range_limits)

    #     if len(filtered_points) > 0:
    #         filtered_base = Path(folder_path) / "filtered_cloud"
    #         save_point_clouds_both_coords(filtered_points, filtered_base)


if __name__ == "__main__":
    main()
