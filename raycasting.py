import open3d as o3d
import numpy as np
import sys

print(o3d.core.cuda.is_available())
# Load mesh and convert to open3d.t.geometry.TriangleMesh
# 加载三角面模型
mesh = o3d.t.io.read_triangle_mesh(sys.argv[1])  # 替换为你的模型文件路径
# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
cube_id = scene.add_triangles(mesh)
print(cube_id)
# We create two rays:
# The first ray starts at (0.5,0.5,10) and has direction (0,0,-1).
# The second ray start at (-1,-1,-1) and has direction (0,0,-1).
# 计算雷达扫描线的点云
"""
elevation_angles = np.linspace(0, np.pi / 2, 900)  # 0.1 度
azimuth_angles = np.linspace(0, 2 * np.pi, 1440)  # 0.25 度
ray_directions = []
for elevation in elevation_angles:
    for azimuth in azimuth_angles:
        x = np.sin(elevation) * np.cos(azimuth)
        y = np.sin(elevation) * np.sin(azimuth)
        z = np.cos(elevation)
        ray_directions.append([0, 0, 0, x, y, z])
np.savez_compressed('ray_directions.npz', ray_directions=ray_directions)
"""
ray_directions = np.load('ray_directions.npz')['ray_directions']
rays = o3d.core.Tensor(ray_directions,
                       dtype=o3d.core.Dtype.Float32)

ans = scene.cast_rays(rays)
print(len(ans['t_hit'].numpy()))
tri_indices = ans['primitive_ids'].numpy()
tri_indices = tri_indices[tri_indices != o3d.t.geometry.RaycastingScene.INVALID_ID]
print(tri_indices)
ans_hits = ans['t_hit'].numpy()
ans_hits[ans_hits == np.inf] = 0
print(len(ans_hits), ray_directions[:,[3,4,5]].shape)
points = ray_directions[:,[3,4,5]] * ans_hits.reshape(-1, 1)
roof_points = points[points[:,2]>2]
left_roof_points = roof_points[roof_points[:,0]<-0.9]
left_roof_points = left_roof_points[left_roof_points[:,1]<2.9]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(left_roof_points)
o3d.io.write_point_cloud('left_model.pcd', pcd)
right_roof_points = roof_points[roof_points[:,0]>0.9]
right_roof_points = right_roof_points[right_roof_points[:,1]<2.9]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(right_roof_points)
o3d.io.write_point_cloud('right_model.pcd', pcd)
middle_roof_points = roof_points[np.abs(roof_points[:,0])<0.9]
middle_roof_points = middle_roof_points[middle_roof_points[:,1]<2.9]
middle_roof_points[:,2] = middle_roof_points[:,2] - 1.5
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(middle_roof_points)
o3d.io.write_point_cloud('middle_model.pcd', pcd)
# o3d.visualization.draw_geometries([pcd, mesh.to_legacy()])
"""
下降顶板高度，与雷达点云对比。直接做Octree对比
legacy_mesh = mesh.to_legacy()
triangles = np.asarray(legacy_mesh.triangles)
whole_mesh = o3d.geometry.TriangleMesh()
whole_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(legacy_mesh.vertices))
whole_mesh.triangles = o3d.utility.Vector3iVector(triangles[tri_indices])
o3d.visualization.draw_geometries([whole_mesh])
"""