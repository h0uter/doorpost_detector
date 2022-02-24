import numpy as np

import open3d as o3d
import copy

from doorpost_detector import PointcloudProcessor
from doorpost_detector import vizualisation
from doorpost_detector.utils.converters import npy2pcd
from doorpost_detector.utils.viz_lvl import VizLVL


# TODO: make this a dataclass
class Response:
    def __init__(
        self, success: bool, poses: list[float], certainty: tuple[float, float]
    ):
        self.success = success
        self.poses = poses
        self.certainty = certainty

# TODO: split into multiple functions
def doorpost_pose_from_cropped_pointcloud_usecase(
    points: list, vis: VizLVL = VizLVL.NONE
) -> Response:
    debug_statements = False
    success = False
    certainty = 0.0
    N = 0
    max_attempts = 50
    processor = PointcloudProcessor()

    while not success and N < max_attempts:
        points_copy = copy.deepcopy(points)
        poses = []
        # FIXME: pointcloud coppying mess
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
            clustered_pc,
            post_vectors,
        ) = processor.obtain_door_post_poses_using_clustering(pointcloud)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([clustered_pc])

        # if we didnt find any post retry
        if possible_posts == False:
            N += 1
            success = False
            print(f">>> no possible doorposts found, retrying {N}")
            continue

        best_fit_door_post_a, best_fit_door_post_b = processor.find_best_fit_doorposts(possible_posts)

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

        if vis >= VizLVL.RESULT_ONLY:
            vizualisation.display_end_result(best_fit_door_post_a, best_fit_door_post_b, post_vectors, pointcloud_orig)

    certainty = processor.determine_certainty_from_angle(post_vectors)
    return Response(success, poses, certainty)


def doorpost_pose_from_pointcloud_and_door_location_estimate_usecase(
    np_points, door_location
) -> Response:
    """
    Given a pointcloud and a door location, find the doorpost pose.
    """
    processor = PointcloudProcessor()
    cropped_points = processor.crop_pointcloud(np_points, door_location)

    response = doorpost_pose_from_cropped_pointcloud_usecase(cropped_points)
    return response
