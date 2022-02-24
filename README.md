# Doorpost Detector
This package contains a pointcloud processing pipeline to estimate the poses of doorposts.

<img src="https://raw.githubusercontent.com/h0uter/doorpost_detector/develop/doorpost_detector_demo.png" alt="alt text" width="700" height="whatever">


## Quick start
`pip install -r requirements.txt`

`pip install -e .`


## To obtain doorpost poses from cropped pointcloud
```python
import doorpost_detector.api as dpd

response = dpd.doorpost_pose_from_cropped_pointcloud_usecase(points)

```


## To obtain doorpost poses from pointcloud and estimated door location
```python
import doorpost_detector.api as dpd

response = dpd.doorpost_pose_from_pointcloud_and_door_location_estimate_usecase(points, door_location)

```


## Response format
```python
@dataclass
class Response:
    success: bool
    poses: tuple[float, float, float, float] # x1,y1,x2,y2
    certainty: tuple[float, float]
```
