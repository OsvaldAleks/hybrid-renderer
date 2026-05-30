import numpy as np

DATA_ROOT         = "./models"
TARGET_SPLAT_COUNT = 500_000

NEAR, FAR = 0.01, 200.0
FOV_Y     = 45.0
TIMING_WINDOW = 60

RENDER_HYBRID     = 0
RENDER_SPLAT_ONLY = 1
RENDER_MESH_ONLY  = 2

# Camera defaults
CAM_RADIUS = 4.0
CAM_THETA  = 178.2
CAM_PHI    = -0.4

# Splat zone defaults
CONE_INNER_ANGLE  = 10.0
CONE_OUTER_ANGLE  = 27.5
CONE_MAX_DEPTH    = 8.0
AUTO_DEPTH_SCALE  = 2.0
SPHERE_RADIUS     = 5.0
SPHERE_FADE       = 1.0

# DoF defaults
DOF_FOCAL_DISTANCE = 4.0
DOF_FOCAL_RANGE    = 0.3
DOF_MAX_COC        = 8.0
DOF_APERTURE       = 0.01

# Other render defaults
DEPTH_MARGIN       = 5.0
FREQ_BLEED         = 2.0
AMBIENT            = np.array([1.0, 1.0, 1.0], np.float32)
LIGHT_DIR          = np.array([0.5, 1.0, 0.7], np.float32)