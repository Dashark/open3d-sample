import open3d as o3d
import numpy as np
import sys
# 读取点云数据
source = o3d.io.read_point_cloud(sys.argv[1])
target = o3d.io.read_point_cloud(sys.argv[2])

# 初始化变换矩阵为单位矩阵
trans_init = np.identity(4)

# 设置 ICP 参数
threshold = 0.035  # 最大对应点距离
max_iteration = 30  # 最大迭代次数

# 运行 ICP 算法
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
)

# 输出配准结果
print("配准结果：")
print(reg_p2p)
indices = np.asarray(reg_p2p.correspondence_set)
print(indices.shape)
source_points = np.asarray(source.points)
selected_source_points = source_points[indices[:,0]]
target_points = np.asarray(target.points)
selected_target_points = target_points[indices[:,1]]
selected_pcd = o3d.geometry.PointCloud()
selected_pcd.points = o3d.utility.Vector3dVector(selected_source_points)
# selected_pcd.colors = o3d.utility.Vector3dVector(selected_target_points)
# 应用变换矩阵到源点云
print("变换矩阵：")
print(reg_p2p.transformation)
source_trans = source.transform(reg_p2p.transformation)

# 可视化配准前后的点云
source.paint_uniform_color([1, 0, 0])  # 源点云设为红色
target.paint_uniform_color([0, 1, 0])  # 目标点云设为绿色

# 配准前的可视化
o3d.visualization.draw_geometries([source, selected_pcd, target], zoom=0.015,
                                  window_name="配准前",
                                  width=800,
                                  height=600)

# 配准后的可视化
o3d.visualization.draw_geometries([source_trans, selected_pcd, target], zoom=0.015,
                                  window_name="配准后",
                                  width=800,
                                  height=600)
# 它们没有明显的对应特征的。按照扫描线的定义，相同角度扫描得到相同的点。