import copy
from dataclasses import dataclass

# import numpy as np
import open3d as o3d

from doorpost_detector import PointcloudProcessor
from doorpost_detector import vizualisation
from doorpost_detector.utils.converters import npy2pcd
from doorpost_detector.utils.viz_lvl import VizLVL


@dataclass
class Response:
    success: bool
    poses: list[float]
    certainty: tuple[float, float]


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
        poses: list[float] = []
        # FIXME: pointcloud coppying mess
        pcd_yolo = npy2pcd(points_copy)
        pcd_orig = copy.deepcopy(pcd_yolo)
        pcd = copy.deepcopy(pcd_yolo)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pcd])

        """remove statistical outliers"""
        (points_copy, pcd, index,) = processor.remove_outliers_around_door_first_pass(
            pcd
        )
        if vis >= VizLVL.EVERY_STEP:
            vizualisation.display_inlier_outlier(pcd, index)

        """try to fit a plane to the pointcloud, corresponding to the U shaped door post plane"""
        best_inliers, outliers = processor.fit_plane_to_U_shaped_door_frame(points_copy)
        if vis >= VizLVL.EVERY_STEP:
            vizualisation.plot_points(points_copy, best_inliers, outliers)

        """remove line corresponding to ground in the U shaped door frame"""
        pcd = processor.remove_ground_plane_line(points_copy, best_inliers)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pcd])

        """subsample points to make clustering tractable"""
        pcd_small = pcd.voxel_down_sample(
            voxel_size=0.05
        )  # apparently this is to help clustering metho
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pcd_small])

        """obtain the doorpost locations using clustering and indexing by color"""
        (
            possible_posts,
            clustered_pc,
            post_vectors,
        ) = processor.obtain_door_post_poses_using_clustering(pcd)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([clustered_pc])

        # if we didnt find any post retry
        if possible_posts == False:
            N += 1
            success = False
            print(f">>> no possible doorposts found, retrying {N}")
            continue

        best_fit_door_post_a, best_fit_door_post_b = processor.find_best_fit_doorposts(
            possible_posts
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

        if vis >= VizLVL.RESULT_ONLY:
            vizualisation.display_end_result(
                best_fit_door_post_a, best_fit_door_post_b, post_vectors, pcd_orig
            )

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
