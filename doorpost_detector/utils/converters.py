import open3d as o3d


def npy2pcd(points: list) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()  # instantiate point cloud
    pc.points = o3d.utility.Vector3dVector(points)  # fill pointcloud with numpy points
    return pc
