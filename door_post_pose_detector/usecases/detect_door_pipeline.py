import numpy as np
from numpy.core.numeric import Inf
import open3d as o3d
import copy

from door_post_pose_detector.entry_points.visualize_pipeline import *
from door_post_pose_detector.entities.pointcloud_processor import PointcloudProcessor
from door_post_pose_detector.utils.utils import npy2pcd
from door_post_pose_detector.utils.o3d_arrow import *

def cropped_pointcloud_to_door_post_poses_pipeline(points:list, vis=0):    

    processor = PointcloudProcessor()

    success = False
    poses = []
    pointcloud = npy2pcd(points)
    pointcloud_orig = copy.deepcopy(pointcloud)
    if vis >= 2: o3d.visualization.draw_geometries([pointcloud])

    '''remove statistical outliers'''
    points, pointcloud, index = processor.remove_outliers_around_door_first_pass(pointcloud)
    if vis >= 2: display_inlier_outlier(pointcloud, index)

    '''try to fit a plane to the pointcloud, corresponding to the U shaped door post plane'''
    best_inliers, outliers = processor.fit_plane_to_U_shaped_door_frame(points)
    if vis >= 2: plot_points(points, best_inliers, outliers)

    '''remove line corresponding to ground in the U shaped door frame'''
    pointcloud = processor.remove_ground_plane_line(points, best_inliers)
    if vis >= 2: o3d.visualization.draw_geometries([pointcloud])

    '''subsample points to make clustering tractable'''
    pointcloud_small = pointcloud.voxel_down_sample(voxel_size=0.05) # apparently this is to help clustering metho
    if vis >= 2: o3d.visualization.draw_geometries([pointcloud_small])

    '''obtain the doorpost locations using clustering and indexing by color'''
    possible_posts, clustered_pointcloud, post_vectors = processor.obtain_door_post_poses_using_clustering(pointcloud)
    if vis >= 2: o3d.visualization.draw_geometries([clustered_pointcloud])

    best_fit_door_post_a, best_fit_door_post_b = None, None
    best_fit_door_width_error = float(Inf)
    for posta in possible_posts:
        for postb in possible_posts:

                door_width = np.linalg.norm(np.array(posta) - np.array(postb))
                print(f"for post {posta} and {postb} the door width is: {door_width}")

                # HACK: this is not a good way to get this width
                # get the doorposts for which the door width is as close to the standard size of a door (0.8) as possible
                door_width_error = np.abs(door_width - 0.8)
                if door_width_error < best_fit_door_width_error:
                    best_fit_door_width_error = door_width_error 
                    best_fit_door_post_a = posta
                    if postb != posta:
                        best_fit_door_post_b = postb 

    print(f"lowest error compared to std doorwidth of 0.8meter: {best_fit_door_width_error}, with posts {best_fit_door_post_a} and {best_fit_door_post_b}")

    # check whether we dont have duplicate posts
    if best_fit_door_post_a and best_fit_door_post_b and best_fit_door_post_a is not best_fit_door_post_b:
        success = True
        poses = [
            best_fit_door_post_a[0],
            best_fit_door_post_a[1],
            best_fit_door_post_b[0],
            best_fit_door_post_b[1] 
        ]

    # print(f">>> success of pipeline: {success}, poses: {poses}")

    # prevent nonetype from fucking up
    FOR = get_o3d_FOR()
    if best_fit_door_post_a:
        xa, ya = best_fit_door_post_a
        arrow_a = get_arrow([xa, ya, 0.01], vec=post_vectors[0])
    
        if best_fit_door_post_b:
            xb, yb = best_fit_door_post_b
            arrow_b = get_arrow([xb, yb, 0.01], vec=post_vectors[1])
            if vis >= 1: draw_geometries([FOR, pointcloud_orig, arrow_a, arrow_b])   
        else:
            if vis >= 1: draw_geometries([FOR, pointcloud_orig, arrow_a])

    return  {
        'poses': poses,
        'success': success,
    }
