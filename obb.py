import open3d as o3d
import numpy as np

# 读取点云数据
pcd = o3d.io.read_point_cloud("true_support.ply")

# 计算定向边界框
obb = pcd.get_minimal_oriented_bounding_box()
obb.color = (0, 1, 0) # 设置边界框颜色为绿色
# obb.translate([1, 1, 1], False)
print(np.asarray(obb.get_box_points()))
obb.scale(1.5, obb.get_box_points()[4])
# extent = np.copy(obb.extent)
# extent[0] *= 2
# scaled_aabb = o3d.geometry.AxisAlignedBoundingBox(obb.center - extent / 2, obb.center + extent / 2)
# scaled_aabb.color = (0, 1, 0)

# 可视化点云和定向边界框
# o3d.visualization.draw_geometries([pcd, obb], zoom=0.7, front=[0.5439, -0.2333, -0.8060], lookat=[2.4615, 2.1331, 1.338], up=[-0.1781, -0.9708, 0.1608])
o3d.visualization.draw_geometries([pcd, obb])