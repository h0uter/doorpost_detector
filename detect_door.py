import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.numeric import Inf
from skimage.measure import LineModelND, ransac
import open3d as o3d
import pyransac3d as pyrsc
import copy

try:
    from uncertainty_pose_estimator.utils.o3d_arrow import *
except:
    pass

try:
    from utils.o3d_arrow import *
except:
    pass

### VISUALIZATION ###
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """

    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def plot_points(points, best_inliers, outliers):
    # TODO: update to not take the points argument
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[best_inliers][:, 0], points[best_inliers][:, 1],
               points[best_inliers][:, 2], c='b', marker='o', label='Inlier data')
    ax.scatter(points[outliers][:, 0], points[outliers][:, 1],
               points[outliers][:, 2], c='r', marker='o', label='Outlier data')
    ax.legend(loc='lower left')
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()

def display_inlier_outlier(cloud, ind):
    '''displays the result of the o3d outlier removal'''
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # TODO: make this respect the vis setting
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def npy2pcd(points):
    point_cloud = o3d.geometry.PointCloud()  # instantiate point cloud
    point_cloud.points = o3d.utility.Vector3dVector(
        points)  # fill pointcloud with numpy points
    # TODO: make this respect the vis setting
    return point_cloud

### PROCESSING ###

def remove_outliers_around_door_first_pass(pointcloud, vis=False):
    cl, index = pointcloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    # cl, index = pointcloud.remove_radius_outlier(nb_points=40, radius = 0.075)
    # cl, index = pointcloud.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.01)
    # TODO: make this respect the vis setting
    if vis is True: display_inlier_outlier(pointcloud, index)
    pointcloud_inliers = pointcloud.select_by_index(index)
    inlier_points = np.asarray(pointcloud_inliers.points)
    return inlier_points, pointcloud

def fit_plane_to_U_shaped_door_frame(points):
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(points, 0.1, maxIteration=1000)
    set_difference = set(list(range(len(points)))) - set(best_inliers)
    outliers = list(set_difference)
    return best_inliers, outliers

def remove_ground_plane_line(points, best_inliers):
    ''' removes the ground plane using spots height '''
    dpoints = points[best_inliers]

    # change to expected hight of spot
    not_plane = dpoints[:, 2] > np.min(dpoints[:, 2]) + 0.1
    best_points = dpoints[not_plane]
    pointcloud = npy2pcd(best_points)
    return pointcloud

def obtain_door_post_poses_using_clustering(pcd_small):
    labels = np.array(pcd_small.cluster_dbscan(
        eps=0.2, min_points=10, print_progress=True))

    max_label = labels.max()
    print("pointcloud has %d clusters" % (max_label + 1))
    
    possible_posts = []
    post_vectors = []

    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_small.colors = o3d.utility.Vector3dVector(colors[:, :3])
    color_array = np.asarray(pcd_small.colors)
    color_list = color_array[:, 0] + 10*color_array[:, 1] + 100*color_array[:, 1]
    colors = np.unique(color_list)
    for color in colors:
        temp = color_list == color
        ind = [i for i, x in enumerate(temp) if x]
        pcd_inlier = pcd_small.select_by_index(ind)
        points = np.asarray(pcd_inlier.points)
        line = pyrsc.Line()
        # TODO: use A vector to vizualize the fit
        A, B, best_inliers = line.fit(
            points, 0.01, maxIteration=1000)

        if np.abs(A[2]) > 0.9:
            possible_posts.append([B[0], B[1]])
            # print(f"possible_posts: {possible_posts[-1]}")
            post_vectors.append(A)
    
    print(f"possible_posts: {possible_posts}")
    clustered_pointcloud = pcd_small
    return possible_posts, clustered_pointcloud, post_vectors

def process_pointcloud_to_door_post_poses(points, vis=False):    
    success = False
    poses = []
    pointcloud = npy2pcd(points)
    pointcloud_orig = copy.deepcopy(pointcloud)
    if vis is True: o3d.visualization.draw_geometries([pointcloud])

    '''remove statistical outliers'''
    points, pointcloud = remove_outliers_around_door_first_pass(pointcloud, vis)

    '''try to fit a plane to the pointcloud, corresponding to the U shaped door post plane'''
    best_inliers, outliers = fit_plane_to_U_shaped_door_frame(points)
    if vis is True: plot_points(points, best_inliers, outliers)

    '''remove line corresponding to ground in the U shaped door frame'''
    pointcloud = remove_ground_plane_line(points, best_inliers)
    if vis is True: o3d.visualization.draw_geometries([pointcloud])

    '''subsample points to make clustering tractable'''
    pointcloud_small = pointcloud.voxel_down_sample(voxel_size=0.05) # apparently this is to help clustering metho
    if vis is True: o3d.visualization.draw_geometries([pointcloud_small])

    '''obtain the doorpost locations using clustering and indexing by color'''
    possible_posts, clustered_pointcloud, post_vectors = obtain_door_post_poses_using_clustering(pointcloud)
    if vis is True: o3d.visualization.draw_geometries([clustered_pointcloud])

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

    print(f">>> success of pipeline: {success}, poses: {poses}")

    # prevent nonetype from fucking up
    FOR = get_o3d_FOR()
    if best_fit_door_post_a:
        xa, ya = best_fit_door_post_a
        arrow_a = get_arrow([xa, ya, 0.01], vec=post_vectors[0])
    
        if best_fit_door_post_b:
            xb, yb = best_fit_door_post_b
            arrow_b = get_arrow([xb, yb, 0.01], vec=post_vectors[1])
            draw_geometries([FOR, pointcloud_orig, arrow_a, arrow_b])   
        else:
            draw_geometries([FOR, pointcloud_orig, arrow_a])

    return  {
        'poses': poses,
        'success': success,
    }


if __name__ == "__main__":
    # points = np.load('data/robot_cropped_pointcloud.npy')
    path = os.path.abspath(".")

    for i in range(0, 7):
        points = None
        if i == 0:
            points = np.load(f'{path}/data/door0_cropped.npy')
        else:
            # crop margin 0.8m works decent
            points = np.load(f'{path}/data/door{i}_cropped_m0_8.npy')
            # crop margin 1.5m has lots of problems
            # points = np.load(f'data/door{i}_cropped_m1_5.npy')
        
        process_pointcloud_to_door_post_poses(points, vis=False)
