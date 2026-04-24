import numpy as np


def points_in_AABB_3d(points, aabb, clearance=0):
    """
    check whether 3D points are in 3D axis-aligned bounding boxes (box obstacles)
    - inputs
        - points: np (n, 3) or tuple (3,)
        - aabb: np (m, 6), (x, y, z, w, h, d), or None, which means no obstacles
        - clearance: scalar
    - outputs:
        - in_aabb: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp,zp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if aabb is None:
        if points_type==tuple:
            in_aabb = False
        else:
            in_aabb = np.zeros(points.shape[0]).astype(bool)
        return in_aabb
    n, m = points.shape[0], aabb.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)
    xmin, ymin, zmin, w, h, d = aabb[:,0], aabb[:,1], aabb[:,2], aabb[:,3], aabb[:,4], aabb[:,5] # (m,)
    xmax, ymax, zmax = xmin+w+clearance, ymin+h+clearance, zmin+d+clearance
    xmin, ymin, zmin = xmin-clearance, ymin-clearance, zmin-clearance
    xmin, ymin, zmin, xmax, ymax, zmax = \
        xmin*np.ones((n,1)), ymin*np.ones((n,1)), zmin*np.ones((n,1)), xmax*np.ones((n,1)), ymax*np.ones((n,1)), zmax*np.ones((n,1)) # (n,m)
    in_aabb = (xmin<=xp)*(xp<=xmax)*(ymin<=yp)*(yp<=ymax)*(zmin<=zp)*(zp<=zmax) # (n,m)
    in_aabb = np.sum(in_aabb, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_aabb[0]
    return in_aabb


def points_in_ball_3d(points, ball, clearance=0):
    """
    check whether 3D points are in 3D balls (ball obstacles)
    - inputs
        - points: np (n, 3) or tuple (3,)
        - ball: np (m, 4), (x, y, z, r), or None, which means no obstacles
        - clearance: scalar
    - outputs:
        - in_ball: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp,zp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if ball is None:
        if points_type==tuple:
            in_ball = False
        else:
            in_ball = np.zeros(points.shape[0]).astype(bool)
        return in_ball
    n, m = points.shape[0], ball.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)

    xb, yb, zb, r =  ball[:,0], ball[:,1], ball[:,2], ball[:,3] # (m,)
    r = r+clearance

    xb, yb, zb, r = \
        xb*np.ones((n,1)), yb*np.ones((n,1)), zb*np.ones((n,1)), r*np.ones((n,1)) # (n,m)
    in_ball = ((xp-xb)**2+(yp-yb)**2+(zp-zb)**2)<=r**2 # (n,m)
    in_ball = np.sum(in_ball, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_ball[0]
    return in_ball


def points_in_cylinder_3d(points, cylinder, clearance=0):
    """
    check whether 3D points are in 3D cylinders (cylinder obstacles)
    - inputs
        - points: np (n, 3) or tuple (3,)
        - cylinder: np (m, 5), (x, y, z, r, h), center-based, or None
        - clearance: scalar
    - outputs:
        - in_cylinder: np bool (n,) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if cylinder is None:
        if points_type==tuple:
            return False
        return np.zeros(points.shape[0]).astype(bool)
    n, m = points.shape[0], cylinder.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)

    xc, yc, zc, r, h = cylinder[:,0], cylinder[:,1], cylinder[:,2], cylinder[:,3], cylinder[:,4] # (m,)
    r = r+clearance
    half_h = h/2+clearance

    xc, yc, zc, r, half_h = \
        xc*np.ones((n,1)), yc*np.ones((n,1)), zc*np.ones((n,1)), r*np.ones((n,1)), half_h*np.ones((n,1)) # (n,m)
    in_cylinder = (((xp-xc)**2+(yp-yc)**2)<=r**2)*(np.abs(zp-zc)<=half_h) # (n,m)
    in_cylinder = np.sum(in_cylinder, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_cylinder[0]
    return in_cylinder


def points_in_robot_arm_3d(points, robot_collision_bodies, clearance=0):
    """
    check whether 3D points are inside robot arm collision bodies (oriented spheres/cylinders)
    - inputs
        - points: np (n, 3) or tuple (3,)
        - robot_collision_bodies: list of dicts, each with:
            - 'type': 'sphere' or 'cylinder'
            - 'transform': np (4, 4) world transform
            - 'radius': float
            - 'length': float (cylinder only)
        - clearance: scalar
    - outputs:
        - in_robot: np bool (n,) or bool value
    """
    points_type = type(points)
    if points_type == tuple:
        points = np.array(points)[np.newaxis, :]
        assert points.shape == (1, 3)
    if not robot_collision_bodies:
        if points_type == tuple:
            return False
        return np.zeros(points.shape[0]).astype(bool)

    n = points.shape[0]
    in_robot = np.zeros(n, dtype=bool)

    for body in robot_collision_bodies:
        T = body['transform']
        R = T[:3, :3]
        pos = T[:3, 3]

        if body['type'] == 'sphere':
            r = body['radius'] + clearance
            dist_sq = np.sum((points - pos) ** 2, axis=1)
            in_robot |= dist_sq <= r ** 2

        elif body['type'] == 'cylinder':
            r = body['radius'] + clearance
            half_l = body['length'] / 2 + clearance
            # 将点转换到圆柱体局部坐标系（z 轴为圆柱轴向）
            local_pts = (R.T @ (points - pos).T).T  # (n, 3)
            xy_dist_sq = local_pts[:, 0] ** 2 + local_pts[:, 1] ** 2
            z_abs = np.abs(local_pts[:, 2])
            in_robot |= (xy_dist_sq <= r ** 2) & (z_abs <= half_l)

    if points_type == tuple:
        return in_robot[0]
    return in_robot


def points_in_range(points, x_range, y_range, clearance=0):
    """
    check whether 2D points are within workspace range
    - inputs
        - points: np (n, 2) or tuple (2,)
        - x_range: tuple/list (xmin, xmax)
        - y_range: tuple/list (ymin, ymax)
        - clearance: scalar
    - outputs
        - in_range: np bool (n,) or bool value
    """
    points_type = type(points)
    if points_type == tuple:
        points = np.array(points)[np.newaxis, :]
        assert points.shape == (1, 2)

    x, y = points[:, 0], points[:, 1]
    xmin, xmax = x_range
    ymin, ymax = y_range
    in_range = (
        (x >= xmin + clearance)
        * (x <= xmax - clearance)
        * (y >= ymin + clearance)
        * (y <= ymax - clearance)
    ).astype(bool)

    if points_type == tuple:
        return in_range[0]
    return in_range


def points_in_range_3d(points, x_range, y_range, z_range, clearance=0):
    """
    check whether 3D points are within workspace range
    - inputs
        - points: np (n, 3) or tuple (3,)
        - x_range: tuple/list (xmin, xmax)
        - y_range: tuple/list (ymin, ymax)
        - z_range: tuple/list (zmin, zmax)
        - clearance: scalar
    - outputs
        - in_range: np bool (n,) or bool value
    """
    points_type = type(points)
    if points_type == tuple:
        points = np.array(points)[np.newaxis, :]
        assert points.shape == (1, 3)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    in_range = (
        (x >= xmin + clearance)
        * (x <= xmax - clearance)
        * (y >= ymin + clearance)
        * (y <= ymax - clearance)
        * (z >= zmin + clearance)
        * (z <= zmax - clearance)
    ).astype(bool)

    if points_type == tuple:
        return in_range[0]
    return in_range


def points_in_balls_boxes(points, obs_ball, obs_box, clearance=0):
    """
    check whether points are inside any ball or axis-aligned box obstacle
    - inputs
        - points: np (n, 3) or tuple (3,)
        - obs_ball: np (m, 4), (x, y, z, r), or None
        - obs_box: np (k, 6), (x, y, z, w, h, d), or None
        - clearance: scalar
    - outputs
        - in_obs: np bool (n,) or bool value
    """
    in_ball = points_in_ball_3d(points, obs_ball, clearance=clearance)
    in_box = points_in_AABB_3d(points, obs_box, clearance=clearance)
    return in_ball | in_box


def points_validity_3d(
    points,
    obs_ball,
    obs_box,
    x_range,
    y_range,
    z_range,
    obstacle_clearance=0,
    range_clearance=0,
):
    """
    check whether 3D points are valid (in range and not in obstacle)
    """
    in_obs = points_in_balls_boxes(
        points,
        obs_ball,
        obs_box,
        clearance=obstacle_clearance,
    )
    in_range = points_in_range_3d(
        points,
        x_range,
        y_range,
        z_range,
        clearance=range_clearance,
    )
    return (~in_obs) & in_range


def check_collision_line_balls_boxes(line, obs_ball, obs_box, clearance=0, resolution=0.02):
    """
    check whether a 3D line segment collides with any ball or axis-aligned box
    - inputs
        - line: np (2, 3), [start, end]
        - obs_ball: np (m, 4), or None
        - obs_box: np (k, 6), or None
        - clearance: scalar
        - resolution: line sampling resolution
    - outputs
        - collision: bool
    """
    line = np.asarray(line, dtype=float)
    if line.shape != (2, 3):
        raise ValueError(f"line shape must be (2,3), got {line.shape}")

    start = line[0]
    end = line[1]
    dist = np.linalg.norm(end - start)

    if dist == 0:
        return bool(points_in_balls_boxes(tuple(start.tolist()), obs_ball, obs_box, clearance=clearance))

    n_steps = max(2, int(np.ceil(dist / max(resolution, 1e-9))) + 1)
    t = np.linspace(0.0, 1.0, n_steps)[:, np.newaxis]
    samples = start[np.newaxis, :] + t * (end - start)[np.newaxis, :]
    in_obs = points_in_balls_boxes(samples, obs_ball, obs_box, clearance=clearance)
    return bool(np.any(in_obs))










