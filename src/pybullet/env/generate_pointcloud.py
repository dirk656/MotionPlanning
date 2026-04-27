import argparse
import os
import time

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import yaml

from pointcloud_tools import load_workspace_bounds ,depth_buffer_to_points_world, filter_points_by_bounds
from save_tools import save_pointcloud_image, export_frames_to_mp4, save_points_to_ply






def main():
    #argparse
    parser = argparse.ArgumentParser(description="Capture dynamic point cloud images in Franka PyBullet scene.")
    parser.add_argument("--gui", action="store_true", help="Run with GUI. Default is DIRECT for stable capture.")
    parser.add_argument("--frames", type=int, default=180, help="Number of frames to capture.")
    parser.add_argument("--outdir", type=str, default="results/dynamic_pointcloud", help="Output directory.")
    parser.add_argument("--width", type=int, default=320, help="Camera image width.")
    parser.add_argument("--height", type=int, default=240, help="Camera image height.")
    parser.add_argument("--save_npz", action="store_true", help="Save raw xyz point cloud per frame as npz.")
    parser.add_argument("--save_ply", action="store_true", help="Save true 3D point cloud per frame as PLY.")
    parser.add_argument("--export_mp4", action="store_true", help="Export captured PNG frames as an mp4 video.")
    parser.add_argument("--fps", type=int, default=30, help="FPS for exported mp4.")
    parser.add_argument("--mp4_name", type=str, default="dynamic_pointcloud.mp4", help="Output mp4 filename.")
    parser.add_argument("--env_config",type=str,default="src/config/env.yaml",help="Path to env yaml config, used for workspace_bounds.",)
    args = parser.parse_args()
    #env config
    bounds_min, bounds_max = load_workspace_bounds(args.env_config)

    matplotlib.use("Agg")
    os.makedirs(args.outdir, exist_ok=True)

    #pybullet init 
    connection_mode = p.GUI if args.gui else p.DIRECT
    client = p.connect(connection_mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    bounds_min, bounds_max = load_workspace_bounds(args.env_config)
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    robot_base_center = np.array([0.0, 0.0, 0.0])
    sphere_radius = 0.5
    box_size = 0.2
    visual_shape_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[box_size / 2] * 3,
        rgbaColor=[1, 0, 0, 1],
    )
    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_size / 2] * 3)
    obstacle_id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision_shape_id,baseVisualShapeIndex=visual_shape_id,basePosition=[0.3, 0, 0.3],)
    joint_indices = list(range(7))
    target_positions = [0, -0.5, 0, -1.0, 0, 1.5, 0]
    #camera params
    fov = 60.0
    aspect = args.width / args.height
    near = 0.05
    far = 3.0
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)

    try:
        t = 0.0
        generated_frame_paths = []
        for frame_idx in range(args.frames):
            t += 0.02

            new_x = 0.6 + 0.3 * np.sin(t)
            candidate_pos = np.array([new_x, 0.0, 0.5])
            offset = candidate_pos - robot_base_center
            distance = np.linalg.norm(offset)
            if distance > sphere_radius:
                candidate_pos = robot_base_center + (offset / distance) * sphere_radius

            p.resetBasePositionAndOrientation(obstacle_id, candidate_pos.tolist(), [0, 0, 0, 1])

            p.setJointMotorControlArray(
                robot_id,
                joint_indices,
                p.POSITION_CONTROL,
                targetPositions=target_positions,
                positionGains=[0.1] * 7,
                velocityGains=[1.0] * 7,
            )

            p.stepSimulation()

            cam_target = [0.25, 0.0, 0.25]
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_target,
                distance=1.3,
                yaw=45,
                pitch=-35,
                roll=0,
                upAxisIndex=2,
            )

            _, _, _, depth_buffer, _ = p.getCameraImage(
                width=args.width,
                height=args.height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
            )
            depth_buffer = np.asarray(depth_buffer, dtype=np.float64)

            points = depth_buffer_to_points_world(depth_buffer, view_matrix, proj_matrix)
            points = filter_points_by_bounds(points, bounds_min, bounds_max)
            if points.shape[0] == 0:
                continue

            image_path = os.path.join(args.outdir, f"pc_frame_{frame_idx:04d}.png")
            save_pointcloud_image(points, image_path, bounds_min, bounds_max)
            generated_frame_paths.append(image_path)

            if args.save_npz:
                npz_path = os.path.join(args.outdir, f"pc_frame_{frame_idx:04d}.npz")
                np.savez_compressed(npz_path, points=points)

            if args.save_ply:
                ply_path = os.path.join(args.outdir, f"pc_frame_{frame_idx:04d}.ply")
                save_points_to_ply(points, ply_path)

            if args.gui:
                time.sleep(1.0 / 240.0)

        if args.export_mp4:
            mp4_path = os.path.join(args.outdir, args.mp4_name)
            export_frames_to_mp4(generated_frame_paths, mp4_path, args.fps)
            print(f"MP4 exported: {mp4_path}")

    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
