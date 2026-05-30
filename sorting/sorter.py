import numpy as np
import torch

def _extract_view_direction(view_np):
    fwd = -view_np[:3, 2].copy()
    norm = np.linalg.norm(fwd)
    return fwd / norm

def _extract_cam_pos(view_np):
    R = view_np[:3, :3]
    t = view_np[3, :3]
    return -t @ R.T

def _sort(data, view_np):
    view_t = torch.from_numpy(view_np).cuda()
    pos_t = torch.from_numpy(data[:, :3]).cuda()
    ones_t = torch.ones(len(pos_t), 1, device='cuda')
    cam_z  = (torch.cat([pos_t, ones_t], dim=1) @ view_t)[:, 2]
    visible = cam_z < -0.05
    cam_z = cam_z[visible]
    data_t = torch.from_numpy(data).cuda()[visible]
    order = torch.argsort(cam_z, stable=True)
    return data_t[order].cpu().numpy()

def _build_camera_axes(cam_fwd):
    fwd = cam_fwd / (np.linalg.norm(cam_fwd) + 1e-12)
    world_up = np.array([0., -1., 0.], np.float32)
    right = np.cross(fwd, world_up)
    rn = np.linalg.norm(right)

    # Fallback for when camera is looking straingt up/down
    if rn < 1e-6:
        world_up = np.array([1., 0., 0.], np.float32)
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
    right /= rn
    up = np.cross(right, fwd)
    return fwd, right, up

class SplatSorter:
    angle_threshold = 0.005
    dist_threshold = 0.02
    cull_angle_threshold = 0.001
    cull_dist_threshold = 0.1

    def __init__(self):
        self._last_sorted = None
        self._last_culled = None
        self._last_indices = None
        self._last_n = 0
        self._last_dir = None
        self._last_pos = None
        self._last_cull_dir = None
        self._last_cull_pos = None
        self._fy = 2.414

    def flush(self):
        self._last_sorted = None
        self._last_culled = None
        self._last_indices = None
        self._last_n = 0
        self._last_dir = None
        self._last_pos = None
        self._last_cull_dir = None
        self._last_cull_pos = None

    def sort(self, scene, view,
             cone_mode=False,
             cam_fwd=None, cam_pos_w=None,
             cone_inner_deg=40.0, cone_outer_deg=60.0, cone_max_depth=8.0,
             data_override=None, aspect=1.0, fy=2.414):
        self._fy = fy

        view_np = np.array(view, dtype=np.float32)
        cam_dir = _extract_view_direction(view_np)
        cam_pos = _extract_cam_pos(view_np)

        if data_override is not None:
            data = data_override
            n = len(data)
            if (self._last_sorted is not None and self._last_n == n
                    and self._last_dir is not None and self._last_pos is not None):
                dir_ok = (1.0 - float(np.dot(cam_dir, self._last_dir))) <= self.angle_threshold
                pos_ok = np.linalg.norm(cam_pos - self._last_pos) <= self.dist_threshold
                if dir_ok and pos_ok:
                    return self._last_sorted
            if n == 0:
                return np.empty((0, scene.full_splats.shape[1]), np.float32)
            sd = _sort(data, view_np)
            self._last_sorted = sd;
            self._last_n = n
            self._last_dir = cam_dir.copy();
            self._last_pos = cam_pos.copy()
            return sd

        # Cull cache check
        need_cull = True
        if cone_mode and self._last_indices is not None:
            cdir_ok = (self._last_cull_dir is not None and
                       (1.0 - np.dot(cam_dir, self._last_cull_dir)) <= self.cull_angle_threshold)
            cpos_ok = (self._last_cull_pos is not None and
                       np.linalg.norm(cam_pos - self._last_cull_pos) <= self.cull_dist_threshold)
            need_cull = not (cdir_ok and cpos_ok)

        if need_cull:
            if cone_mode and cam_fwd is not None:
                indices = self._cone_cull_indices(
                    scene, cam_pos_w, cam_fwd,
                    cone_inner_deg, cone_outer_deg, cone_max_depth,
                    aspect=aspect)
            else:
                indices = np.arange(len(scene.current_splats), dtype=np.int32)

            src = scene.full_splats if cone_mode else scene.current_splats
            data = src[indices].copy()

            self._last_indices = indices
            self._last_culled = data
            self._last_cull_dir = cam_dir.copy()
            self._last_cull_pos = cam_pos.copy()
            self._last_sorted = None
        else:
            data = self._last_culled

        n = len(data)

        # Sort cache check
        if (self._last_sorted is not None and self._last_n == n
                and self._last_dir is not None and self._last_pos is not None):
            dir_ok = (1.0 - np.dot(cam_dir, self._last_dir)) <= self.angle_threshold
            pos_ok = np.linalg.norm(cam_pos - self._last_pos) <= self.dist_threshold
            if dir_ok and pos_ok:
                return self._last_sorted

        # Avoid empty buffer error
        if n == 0:
            empty = np.empty((0, scene.full_splats.shape[1]), np.float32)
            self._last_sorted = empty;
            self._last_n = 0
            return empty

        sorted_data = _sort(data, view_np)
        self._last_sorted = sorted_data
        self._last_n = n
        self._last_dir = cam_dir.copy()
        self._last_pos = cam_pos.copy()
        return sorted_data

    @staticmethod
    def _cone_cull_indices(scene, cam_pos, cam_fwd, inner_deg, outer_deg, max_depth,
                           aspect=1.0, fy=2.414):
        oy = float(np.tan(np.deg2rad(outer_deg)) * fy)

        # Project splats to view space
        fwd, right, up = _build_camera_axes(cam_fwd)
        d  = scene.splat_pos - cam_pos
        vz = d @ fwd
        vx = d @ right
        vy = d @ up

        depth_mask = (vz > 0.1) & (vz <= max_depth + 1.0)
        idx_d = np.where(depth_mask)[0]
        if len(idx_d) == 0:
            return np.empty(0, np.int32)

        vz2 = np.maximum(vz[idx_d], 1e-6)
        ndcx = vx[idx_d] / vz2 * fy / aspect
        ndcy = vy[idx_d] / vz2 * fy
        r = np.sqrt((ndcx/oy)**2 + (ndcy/oy)**2)
        return idx_d[r <= 1.0].astype(np.int32)