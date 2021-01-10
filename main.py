import numpy as np
from cv2 import cv2 # for pylint to work
from scipy.spatial.transform import Rotation as R
import helpers_plotting

## Constants
CAM_MATRIX = np.array([[5.3578e3, 0, 2.1329e3], [0, 5.3769e3, 2.3431e3], [
                      0, 0, 1]], dtype="double")  # [[fx 0 cx],[0 fy cy],[0 0 1]] where f in pixel units
# None # k1, k2, p1, p2(, k3(, k4, k5, k6)) where k radial, p tangential
DISTORTION = np.array(
    [-0.1632, 0.5933, 0.0077, -0.0012, -1.9754], dtype="double")
DIST_ZEROED = np.array([0, 0, 0, 0, 0], dtype="double")

WAND_LEDS = np.array([[9.0967378616333008, 128.44847106933594, 24.759185791015625], [-71.046150207519531, 8.5778627395629883, 25.214023590087891], [8.9318943023681641, 248.62664794921875, 25.382614135742188], [8.8888835906982422, 8.8440675735473633, 24.927968978881836], [168.46176147460938, 8.8073110580444336, 25.007959365844727]], dtype='double')

# Randomise 3D points and camera pose
vertices_3d = np.random.rand(10,3)*100
rand3 = np.random.rand(3,)
trans = np.random.rand(3,1)*100

# calculate rigid transform
rot_vec = rand3 / np.linalg.norm(rand3)
rot = R.from_rotvec(rot_vec)
rigid_transform = np.hstack((rot.as_matrix(), trans))

# Project to 2D image plane
uv_stack = np.array([[None, None]])
for vertex in vertices_3d:
    vertex = np.array([vertex]).T
    dot1 = np.dot(CAM_MATRIX, rigid_transform)
    dot2 = np.dot(dot1,np.vstack((vertex,1)))
    uv = dot2/dot2[2]
    uv_stack = np.vstack((uv_stack,uv[:2].T))

# Remove first entry
uv_stack = uv_stack[1:,:]
title = "test"
#helpers_plotting.plot_correlation(vertices_3d[:,0],vertices_3d[:,1],vertices_3d[:,2],uv_stack[:,0],uv_stack[:,1], title,show=True)
print(uv_stack)
# OpenCV call
retval, rvec, tvec, inliers = cv2.solvePnPRansac(vertices_3d, uv_stack, CAM_MATRIX, DIST_ZEROED)

# Analyse
print(f"rvec: {rvec}")
print(f"tvec: {tvec}")