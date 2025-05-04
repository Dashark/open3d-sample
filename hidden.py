import open3d as o3d
import numpy as np
import sys

print(f"Convert mesh to a point cloud and estimate dimensions {sys.argv[1]}")
mesh = o3d.io.read_triangle_mesh(sys.argv[1])
mesh.compute_vertex_normals()

# pcd = mesh.sample_points_poisson_disk(50000)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
# o3d.visualization.draw_geometries([pcd])
print(diameter)
print(len(mesh.vertices))
triangle_index = np.asarray(mesh.triangles)
print(triangle_index.shape)

print("Define parameters used for hidden_point_removal")
camera = [0, 0, 0]
radius = diameter * 10

o3d.visualization.draw_geometries([pcd, mesh])
print("Get all points that are visible from given view point")
shmesh, pt_map = pcd.hidden_point_removal(camera, radius)
print(len(pt_map))
print(len(shmesh.vertices))
results = np.array([], dtype=np.int64)
for pt in pt_map:
    results = np.append(results, np.where(np.any(triangle_index == pt, axis=1)))
# 把原始Mesh的顶点和三角形索引更新为只包含可见点的Mesh
mesh.triangles = o3d.utility.Vector3iVector(triangle_index[results])
# print(results)
print("Visualize result")
pcd = pcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries([shmesh])
o3d.io.write_triangle_mesh(sys.argv[2], mesh)