import open3d as o3d
import numpy as np
import sys

# 加载三角面模型
mesh = o3d.io.read_triangle_mesh(sys.argv[1])  # 替换为你的模型文件路径

# 提取顶点和三角形索引
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

# 顶点实例归一化
norm_vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
print(norm_vertices[0], np.linalg.norm(vertices[0]))
print(vertices[0])

# 计算雷达扫描线的点云
elevation_angles = np.linspace(0, np.pi / 2, 900)  # 0.1 度
azimuth_angles = np.linspace(0, 2 * np.pi, 1440)  # 0.25 度
ray_directions = []
for elevation in elevation_angles:
    for azimuth in azimuth_angles:
        x = np.sin(elevation) * np.cos(azimuth)
        y = np.sin(elevation) * np.sin(azimuth)
        z = np.cos(elevation)
        ray_directions.append([x, y, z])
# 创建 Octree
max_depth = 6  # 设置最大深度
ray_octree = o3d.geometry.Octree(max_depth)

# 插入顶点并记录索引
"""
vertex_indices = []
for i, vertex in enumerate(vertices):
    octree.insert_point(vertex, None, None)
    vertex_indices.append(i)
print(vertices[0])
print(triangles[0])
"""
# 关联顶点和三角面索引
vertex_triangle_map = {}
for triangle_index, triangle in enumerate(triangles):
    for vertex_index in triangle:
        if vertex_index not in vertex_triangle_map:
            vertex_triangle_map[vertex_index] = []
        vertex_triangle_map[vertex_index].append(triangle_index)
# 将归一化的点构建 Octree
points = o3d.geometry.PointCloud()
points.points = o3d.utility.Vector3dVector(ray_directions)
ray_octree.convert_from_point_cloud(points, 0.01)

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

# 遍历 Octree 节点并打印顶点索引和关联的三角面索引
def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop

def plane_equation(point1, point2):
    """
    计算两个点与原点构成的平面方程
    :param point1: 第一个点的坐标，格式为 (x1, y1, z1)
    :param point2: 第二个点的坐标，格式为 (x2, y2, z2)
    :return: 平面方程的系数 A, B, C
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    A = y1 * z2 - y2 * z1
    B = -(x1 * z2 - x2 * z1)
    C = x1 * y2 - x2 * y1
    return A, B, C

# 遍历三角面划分点云
for triangle in triangles:
    v0, v1, v2 = vertices[triangle]
    A, B, C = plane_equation(v0, v1)
    break
node, nodeinfo = ray_octree.locate_leaf_node(ray_directions[0])
ray_octree.traverse(traverse_octree)

if node:
    print(node.indices)
    print(nodeinfo)
    tri_indices = np.where(np.any(triangles == node.indices[0], axis=1))
    # print(vertices[triangles[tri_indices]])
    unique_vertices = vertices[np.unique(triangles[tri_indices].flatten())]
    # print(unique_vertices)
    #mesh.triangles = o3d.utility.Vector3iVector(triangles[tri_indices])
# 可视化 Octree
# o3d.visualization.draw_geometries([mesh])