import numpy as np
import moderngl
from types import SimpleNamespace
from scipy.spatial import cKDTree

from config import TARGET_SPLAT_COUNT
from scene.loader import load_splats, load_obj_with_mtl, load_texture

# auto-detect sphere radius, so it contains roughly target_n splats
def auto_radius_for_target(splat_tree, center, target_n, total_n):
    if total_n <= target_n:
        all_pos = splat_tree.data
        dists = np.linalg.norm(all_pos - center, axis=1)
        r = float(np.percentile(dists, 99))
        return r * 0.85, r * 0.15

    lo, hi = 0.0, float(np.linalg.norm(splat_tree.data.max(axis=0) - splat_tree.data.min(axis=0)))
    for _ in range(32):
        mid = (lo + hi) / 2.0
        if len(splat_tree.query_ball_point(center, mid)) < target_n:
            lo = mid
        else:
            hi = mid
    inner = (lo + hi) / 2.0
    return inner, inner * 0.15

def load_scene(model_name, ply_path, obj_path, ctx):
    print(f"LOAGING {model_name}")
    s = SimpleNamespace()
    s.name = model_name

    s.full_splats = load_splats(ply_path)
    s.center = s.full_splats[:, :3].mean(axis=0)
    s.full_splats[:, :3] -= s.center
    s.splat_tree = cKDTree(s.full_splats[:, :3])
    s.splat_pos = np.ascontiguousarray(s.full_splats[:, :3])
    s.total_splats = len(s.full_splats)

    look_at = s.full_splats[:, :3].mean(axis=0)
    s.inner_radius, s.fade_width = auto_radius_for_target(
        s.splat_tree, look_at, TARGET_SPLAT_COUNT, s.total_splats)
    s.outer_radius = s.inner_radius + s.fade_width

    (s.interleaved_mesh, s.mesh_indices,
     s.interleaved_uv, s.mesh_uv_indices,
     s.mesh_raw_verts, mesh_tex_path) = load_obj_with_mtl(obj_path)

    s.interleaved_mesh[:, :3] -= s.center.astype(np.float32)
    s.interleaved_uv[:, :3] -= s.center.astype(np.float32)
    s.mesh_raw_verts -= s.center.astype(np.float32)

    s.mesh_vbo = ctx.buffer(s.interleaved_mesh.tobytes())
    s.mesh_ibo = ctx.buffer(s.mesh_indices.tobytes())
    s.mesh_uv_vbo = ctx.buffer(s.interleaved_uv.tobytes())
    s.mesh_uv_vert_count = len(s.interleaved_uv)
    s.mesh_texture = load_texture(ctx, mesh_tex_path)

    print(" - assigning mesh vertex colours from splats...")
    mesh_verts = s.interleaved_mesh[:, :3]
    splat_rgb = s.full_splats[:, 6:9]
    splat_alpha = s.full_splats[:, 9]
    splat_size = np.cbrt(s.full_splats[:, 3] * s.full_splats[:, 4] * s.full_splats[:, 5]).astype(np.float32)

    K = 16
    dists, idxs = s.splat_tree.query(mesh_verts, k=K, workers=-1)
    dists = np.maximum(dists, 1e-6).astype(np.float32)
    a = splat_alpha[idxs]
    sz = np.maximum(splat_size[idxs], 1e-4)
    w = a / (sz * dists ** 2)
    w /= w.sum(axis=1, keepdims=True)
    vert_col = np.clip((w[:, :, None] * splat_rgb[idxs]).sum(axis=1), 0., 1.).astype(np.float32)
    s.interleaved_mesh[:, 6:9] = vert_col
    s.mesh_vbo.write(s.interleaved_mesh.tobytes())

    s.current_splats = s.full_splats
    s.num_splats = s.total_splats
    s.base_splats = s.full_splats.copy()

    n_bytes = s.full_splats.astype('f4').tobytes()
    s.splat_vbo_a = ctx.buffer(n_bytes)
    s.splat_vbo_b = ctx.buffer(n_bytes)
    s.render_vbo = s.splat_vbo_a
    s.upload_vbo = s.splat_vbo_b
    return s