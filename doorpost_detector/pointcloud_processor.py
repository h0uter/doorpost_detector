# ENTITY
import numpy as np
import numpy.typing as npt
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
import open3d as o3d
import logging

from doorpost_detector.utils.converters import npy2pcd


# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


class PointcloudProcessor:
    def __init__(self) -> None:

        # all hyper parameters for the pipeline
        self.nb_neigbours = 100
        self.std_ratio = 0.1
        self.line1_thresh = 0.1
        self.max_iteration = 1000
        self.dbscan_eps = 0.2
        self.dbscan_min_points = 10
        self.door_post_line_thresh = 0.01
        self.door_post_line_max_iteration = 1000

    def crop_pointcloud(
        self, pc_array: npt.NDArray, crop_target: tuple, crop_margin: float = 0.8
    ):

        cropped_pc = [
            point
            for point in pc_array
            if crop_target[0] - crop_margin <= point[0] <= crop_target[0] + crop_margin
            and crop_target[1] - crop_margin <= point[1] <= crop_target[1] + crop_margin
        ]

        return cropped_pc

    def remove_outliers_around_door_first_pass(
        self, pointcloud: o3d.geometry.PointCloud
    ) -> tuple:
        logging.debug(f"removing outliers_around_door_first_pass")
        cl, index = pointcloud.remove_statistical_outlier(
            nb_neighbors=self.nb_neigbours,
            std_ratio=self.std_ratio,
            print_progress=False,
        )
        # cl, index = pointcloud.remove_radius_outlier(nb_points=40, radius = 0.075)
        # cl, index = pointcloud.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.01)
        pointcloud_inliers = pointcloud.select_by_index(index)
        inlier_points = np.asarray(pointcloud_inliers.points)
        return inlier_points, pointcloud, index

    def fit_plane_to_U_shaped_door_frame(self, points: list) -> tuple:
        logging.debug(f"fit_plane_to_U_shaped_door_frame")

        plane1 = pyrsc.Plane()
        best_eq, best_inliers = plane1.fit(
            points, thresh=self.line1_thresh, maxIteration=self.max_iteration
        )

        set_difference = set(list(range(len(points)))) - set(best_inliers)
        outliers = list(set_difference)
        return best_inliers, outliers

    def remove_ground_plane_line(
        self, points: list, best_inliers: list
    ) -> o3d.geometry.PointCloud:
        logging.debug(f"removing the ground plane using spots height")

        dpoints = points[best_inliers]

        # change to expected hight of spot
        not_plane = dpoints[:, 2] > np.min(dpoints[:, 2]) + 0.1
        best_points = dpoints[not_plane]
        pointcloud = npy2pcd(best_points)
        return pointcloud

    def obtain_door_post_poses_using_clustering(
        self, pcd_small: o3d.geometry.PointCloud
    ) -> tuple:

        logging.debug(f"obtain_door_post_poses_using_clustering")

        labels = np.array(
            pcd_small.cluster_dbscan(
                eps=self.dbscan_eps,
                min_points=self.dbscan_min_points,
                print_progress=False,
            )
        )

        max_label = labels.max()

        logging.debug(f"pointcloud has {max_label + 1} clusters")

        possible_posts = []
        post_vectors = []

        cmap = plt.get_cmap("tab20")
        colors = cmap(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_small.colors = o3d.utility.Vector3dVector(colors[:, :3])
        color_array = np.asarray(pcd_small.colors)
        color_list = (
            color_array[:, 0] + 10 * color_array[:, 1] + 100 * color_array[:, 1]
        )
        colors = np.unique(color_list)
        for color in colors:
            temp = color_list == color
            ind = [i for i, x in enumerate(temp) if x]
            pcd_inlier = pcd_small.select_by_index(ind)
            points = np.asarray(pcd_inlier.points)
            door_post_line = pyrsc.Line()
            # TODO: use A vector to vizualize the fit
            # print(f"points: {points}")

            # fix a weird crash if points are too small due to random inlier detection
            if points.shape[0] > 2:
                A, B, best_inliers = door_post_line.fit(
                    points,
                    thresh=self.door_post_line_thresh,
                    maxIteration=self.door_post_line_max_iteration,
                )
            else:
                return False, False, False

            if np.abs(A[2]) > 0.9:
                possible_posts.append([B[0], B[1]])
                # print(f"possible_posts: {possible_posts[-1]}")
                post_vectors.append(A)

        logging.debug(f"possible_posts: {possible_posts}")

        clustered_pointcloud = pcd_small
        return possible_posts, clustered_pointcloud, post_vectors

    def determine_certainty_from_angle(self, post_vectors) -> tuple[float, float]:
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2) -> float:
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
        return certainty

    def find_best_fit_doorposts(self, possible_posts) -> tuple[tuple[float,float], tuple[float,float]]:
        best_fit_door_post_a, best_fit_door_post_b = None, None
        best_fit_door_width_error = float("Inf")
        
        for posta in possible_posts:
            for postb in possible_posts:
                door_width = np.linalg.norm(np.array(posta) - np.array(postb))
                logging.debug(f"for post {posta} and {postb} the door width is: {door_width}")

                # HACK: this is not a good way to get this width
                # get the doorposts for which the door width is as close to the standard size of a door (0.8) as possible
                door_width_error = np.abs(door_width - 0.8)
                if door_width_error < best_fit_door_width_error:
                    best_fit_door_width_error = door_width_error
                    best_fit_door_post_a = posta
                    if postb != posta:
                        best_fit_door_post_b = postb
        
        logging.debug(f"lowest error compared to std doorwidth of 0.8meter: {best_fit_door_width_error}, with posts {best_fit_door_post_a} and {best_fit_door_post_b}")

        return best_fit_door_post_a, best_fit_door_post_b