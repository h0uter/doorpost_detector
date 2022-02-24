import numpy as np

import open3d as o3d
import copy

from doorpost_detector import PointcloudProcessor
from doorpost_detector import vizualisation
from doorpost_detector.utils.converters import npy2pcd
from doorpost_detector.utils.viz_lvl import VizLVL
from doorpost_detector.utils.o3d_arrow import (
    draw_geometries,
    get_o3d_FOR,
    get_arrow,
)

# TODO: make this a dataclass
class Response:
    def __init__(self, success, poses, certainty):
        self.success = success
        self.poses = poses
        self.certainty = certainty


# cropped_pointcloud_to_door_post_poses_usecase
# def detect_doorposts_usecase(points: list, vis=0) -> Response:

# TODO: split into multiple functions
# TODO: create tidy response class
def doorpost_pose_from_cropped_pointcloud_usecase(
    points: list, vis: VizLVL = VizLVL.NONE
) -> Response:
    debug_statements = False
    success = False
    certainty = 0.0
    N = 0
    max_attempts = 50
    processor = PointcloudProcessor()

    # FIXME this condition is not triggered sometimes
    while not success and N < max_attempts:

        points_copy = copy.deepcopy(points)
        poses = []
        # points = copy.deepcopy(points_copy)
        pointcloud_yolo = npy2pcd(points_copy)
        pointcloud_orig = copy.deepcopy(pointcloud_yolo)
        pointcloud = copy.deepcopy(pointcloud_yolo)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pointcloud])

        """remove statistical outliers"""
        (
            points_copy,
            pointcloud,
            index,
        ) = processor.remove_outliers_around_door_first_pass(pointcloud)
        if vis >= VizLVL.EVERY_STEP:
            vizualisation.display_inlier_outlier(pointcloud, index)

        """try to fit a plane to the pointcloud, corresponding to the U shaped door post plane"""
        best_inliers, outliers = processor.fit_plane_to_U_shaped_door_frame(points_copy)
        if vis >= VizLVL.EVERY_STEP:
            vizualisation.plot_points(points_copy, best_inliers, outliers)

        """remove line corresponding to ground in the U shaped door frame"""
        pointcloud = processor.remove_ground_plane_line(points_copy, best_inliers)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pointcloud])

        """subsample points to make clustering tractable"""
        pointcloud_small = pointcloud.voxel_down_sample(
            voxel_size=0.05
        )  # apparently this is to help clustering metho
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pointcloud_small])

        """obtain the doorpost locations using clustering and indexing by color"""
        (
            possible_posts,
            clustered_pointcloud,
            post_vectors,
        ) = processor.obtain_door_post_poses_using_clustering(pointcloud)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([clustered_pointcloud])

        if possible_posts == False:
            N += 1
            success = False
            continue

        # print(f"postvectors {post_vectors}")

        best_fit_door_post_a, best_fit_door_post_b = None, None
        best_fit_door_width_error = float("Inf")
        for posta in possible_posts:
            for postb in possible_posts:

                door_width = np.linalg.norm(np.array(posta) - np.array(postb))
                if debug_statements:
                    print(
                        f"for post {posta} and {postb} the door width is: {door_width}"
                    )

                # HACK: this is not a good way to get this width
                # get the doorposts for which the door width is as close to the standard size of a door (0.8) as possible
                door_width_error = np.abs(door_width - 0.8)
                if door_width_error < best_fit_door_width_error:
                    best_fit_door_width_error = door_width_error
                    best_fit_door_post_a = posta
                    if postb != posta:
                        best_fit_door_post_b = postb

        if debug_statements:
            print(
                f"lowest error compared to std doorwidth of 0.8meter: {best_fit_door_width_error}, with posts {best_fit_door_post_a} and {best_fit_door_post_b}"
            )

        # check whether we dont have duplicate posts
        if (
            best_fit_door_post_a
            and best_fit_door_post_b
            and best_fit_door_post_a is not best_fit_door_post_b
        ):
            success = True
            poses = [
                best_fit_door_post_a[0],
                best_fit_door_post_a[1],
                best_fit_door_post_b[0],
                best_fit_door_post_b[1],
            ]

            """Order door posts so the left one (lowest x coord) always comes first."""
            if poses[1] > poses[3]:
                poses = [poses[2], poses[3], poses[0], poses[1]]

        else:
            N += 1
            success = False
            print(f"Could not find door posts, trying again (attempt {N})")
            continue

        if debug_statements:
            print(f">>> success of pipeline: {success}, poses: {poses}")

        # prevent nonetype from fucking up

        # cleanup this plotting mess
        FOR = get_o3d_FOR()
        if best_fit_door_post_a:
            xa, ya = best_fit_door_post_a
            arrow_a = get_arrow([xa, ya, 0.01], vec=post_vectors[0])

            if best_fit_door_post_b:
                xb, yb = best_fit_door_post_b
                arrow_b = get_arrow([xb, yb, 0.01], vec=post_vectors[1])
                if vis >= VizLVL.RESULT_ONLY:
                    draw_geometries([FOR, pointcloud_orig, arrow_a, arrow_b])
            else:
                if vis >= VizLVL.RESULT_ONLY:
                    draw_geometries([FOR, pointcloud_orig, arrow_a])

        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        # certainty = np.pi - angle_between([1,0,0], post_vectors[0])/ (np.pi), np.pi - angle_between([1,0,0], post_vectors[1])/(np.pi)
        angle_1 = angle_between([0, 0, -1], post_vectors[0])
        angle_2 = angle_between([0, 0, -1], post_vectors[1])
        print(f"angle 1: {angle_1}, angle 2: {angle_2}")
        if 0.5 * np.pi < angle_1 < np.pi:
            angle_1 = abs(angle_1 - np.pi)
        if 0.5 * np.pi < angle_2 < np.pi:
            angle_1 = abs(angle_1 - np.pi)

        certainty = angle_1 / (np.pi), angle_2 / (np.pi)

    response = Response(success, poses, certainty)

    # return {"poses": poses, "success": success, "certainty": certainty}
    return response


def doorpost_pose_from_pointcloud_and_door_location_estimate(np_points, door_location):
    """
    Given a pointcloud and a door location, find the doorpost pose.
    """
    # crop pointcloud
    # run above pipeline
    pass

