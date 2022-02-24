import copy
from dataclasses import dataclass

# import numpy as np
import numpy.typing as npt
import open3d as o3d

from doorpost_detector import PointcloudProcessor, vizualisation
from doorpost_detector.utils.converters import npy2pcd
from doorpost_detector.utils.viz_lvl import VizLVL


@dataclass
class Response:
    success: bool
    # TODO: make this a tuple of two tuples, or named tuples
    poses: tuple[float, float, float, float]
    certainty: tuple[float, float]


# TODO: split into multiple functions
# TODO: remove vizualation
def doorpost_pose_from_cropped_pointcloud_usecase(
    points: list[tuple[float, float, float]], vis: VizLVL = VizLVL.NONE
) -> Response:
    success = False
    certainty = (0.0, 0.0)
    N = 0
    max_attempts = 5
    poses = tuple()
    processor = PointcloudProcessor()
    # post_vectors = None
    # Vector = tuple[float, float, float]
    post_vectors = list()

    while not success and N < max_attempts:
        # FIXME: pointcloud coppying mess
        points_copy = copy.deepcopy(points)
        pcd_yolo = npy2pcd(points_copy)
        pcd_orig = copy.deepcopy(pcd_yolo)
        pcd = copy.deepcopy(pcd_yolo)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pcd], window_name="initial cropped pc")  # type: ignore

        """Temove statistical outliers."""
        (points_copy, pcd, index,) = processor.remove_outliers_around_door_first_pass(
            pcd
        )
        if vis >= VizLVL.EVERY_STEP:
            vizualisation.display_inlier_outlier(pcd, index)

        """Try to fit a plane to the pointcloud, corresponding to the U shaped door post plane."""
        best_inliers, outliers = processor.fit_plane_to_U_shaped_door_frame(points_copy)
        if vis >= VizLVL.EVERY_STEP:
            vizualisation.plot_points(points_copy, best_inliers, outliers)

        """Remove line corresponding to ground in the U shaped door frame."""
        pcd = processor.remove_ground_plane_line(points_copy, best_inliers)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pcd], window_name="Removed line corresponding to ground in the U shaped door frame.")  # type: ignore

        """Subsample points to make clustering tractable."""
        pcd_small = pcd.voxel_down_sample(
            voxel_size=0.05
        )  # apparently this is to help clustering metho
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([pcd_small], window_name="Subsampled points to make clustering tractable.")  # type: ignore

        """Obtain the doorpost locations using clustering and indexing by color."""
        (
            possible_posts,
            clustered_pc,
            post_vectors,
        ) = processor.obtain_door_post_poses_using_clustering(pcd)
        if vis >= VizLVL.EVERY_STEP:
            o3d.visualization.draw_geometries([clustered_pc], window_name="Obtain the doorpost locations using clustering and indexing by color.")  # type: ignore

        # if we didnt find any post retry
        if possible_posts == False:
            N += 1
            success = False
            print(f">>> no possible doorposts found, retrying {N}")
            continue

        best_fit_doorpost_a, best_fit_doorpost_b = processor.find_best_fit_doorposts(
            possible_posts
        )

        # check whether we dont have duplicate posts
        if (
            best_fit_doorpost_a
            and best_fit_doorpost_b
            and best_fit_doorpost_a is not best_fit_doorpost_b
        ):
            success = True
            poses = (
                best_fit_doorpost_a[0],
                best_fit_doorpost_a[1],
                best_fit_doorpost_b[0],
                best_fit_doorpost_b[1],
            )

            """Order door posts so the left one (lowest x coord) always comes first."""
            if poses[1] > poses[3]:
                poses = (poses[2], poses[3], poses[0], poses[1])

        else:
            N += 1
            success = False
            print(f"Could not find door posts, trying again (attempt {N})")
            continue

        if vis >= VizLVL.RESULT_ONLY:
            vizualisation.display_end_result(
                best_fit_doorpost_a, best_fit_doorpost_b, post_vectors, pcd_orig
            )

        certainty = processor.determine_certainty_from_angle(post_vectors)
    return Response(success, poses, certainty)


def doorpost_pose_from_pointcloud_and_door_location_estimate_usecase(
    points: list[tuple[float, float, float]], door_location: tuple[float, float]
) -> Response:
    """
    Given a pointcloud and a door location, find the doorpost pose.
    """
    processor = PointcloudProcessor()
    cropped_points = processor.crop_pointcloud(points, door_location)

    response = doorpost_pose_from_cropped_pointcloud_usecase(cropped_points)
    return response
