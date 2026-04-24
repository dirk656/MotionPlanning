import numpy as np
import yaml
import matplotlib.pyplot as plt
import imageio.v2 as imageio






def save_pointcloud_image(points, output_path, bounds_min, bounds_max):
    fig = plt.figure(figsize=(6, 6), dpi=128)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.8, c=points[:, 2], cmap="viridis")
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Dynamic Point Cloud")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def export_frames_to_mp4(frame_paths, output_mp4_path, fps):
    if not frame_paths:
        raise RuntimeError("No point cloud frames were generated, cannot export mp4")

    with imageio.get_writer(output_mp4_path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def save_points_to_ply(points, output_path):
    """Save xyz points to an ASCII PLY file (true 3D point cloud)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")
