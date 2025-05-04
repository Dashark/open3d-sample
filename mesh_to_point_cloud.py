import open3d as o3d
import numpy as np

# 手动创建一个简单的三角面网格
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
])
triangles = np.array([
    [0, 1, 2]
])

# 创建三角面网格对象
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)

# 均匀采样
num_points_uniform = 100
pcd_uniform = mesh.sample_points_uniformly(number_of_points=num_points_uniform)

# 泊松盘采样
num_points_poisson = 100
pcd_poisson = mesh.sample_points_poisson_disk(number_of_points=num_points_poisson)

# 可视化结果
# o3d.visualization.draw_geometries([pcd_uniform])
# o3d.visualization.draw_geometries([pcd_poisson])
print(vertices[:,2])