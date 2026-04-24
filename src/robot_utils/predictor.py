import numpy as np 
from  wrapper_3d.pointnet_pointnet2 import pointnet2_wrapper_connect_bfs
from generate_datasets_utils.point_cloud_mask_utils import get_point_cloud_mask_around_points
from pybullet.env.path_tools import in_bound
from pybullet.path_tools import  generate_pos 


wrapper = pointnet2_wrapper_connect_bfs(root_dir='results/model_training/pointnet2_3d/checkpoints/best_pointnet2_3d', device='cuda') 


#pc = env.get_point_cloud() 

# 2.2 定义起点和终点
x_start , x_goal = generate_pos(max_radius=20.0)
# 2.3 定义环境字典 (主要用于可视化，如果不开启可视化，障碍物信息可以为空)


# neighbor_radius: 判断两个点是否连通的阈值 (根据你的地图比例调整，例如 0.5米)
# max_trial_attempts: 如果一次预测没连上，最多尝试几次“搭桥”修复
success, num_runs, path_pred_mask = wrapper.generate_connected_path_points(
    pc=pc,
    x_start=x_start,
    x_goal=x_goal,
    env_dict=env_dict,
    neighbor_radius=0.5, 
    max_trial_attempts=5,
    visualize=False # 设为 True 可以看到它如何一步步“搭桥”
)

# --- 4. 使用结果 ---
if success:
    print(f"路径生成成功！尝试了 {num_runs} 次。")
    heuristic_points = pc[path_pred_mask == 1] 
else:
    print("路径生成失败，无法连通起点和终点。")
   