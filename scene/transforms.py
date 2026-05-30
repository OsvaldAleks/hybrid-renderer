import numpy as np
from scipy.spatial import cKDTree

# Cross pattern tiling
def apply_star_replication(s, ctx, n_rings=1):
    pos = s.full_splats[:, :3]
    step_x = (pos[:, 0].max() - pos[:, 0].min()) * 1.05
    step_z = (pos[:, 2].max() - pos[:, 2].min()) * 1.05

    offsets = []
    for ring in range(1, n_rings + 1):
        for dx, dz in [(ring, 0), (-ring, 0), (0, ring), (0, -ring)]:
            off = np.array([dx * step_x, 0.0, dz * step_z], dtype=np.float32)
            if not any(np.allclose(off, o) for o in offsets):
                offsets.append(off)

    splat_copies = [s.full_splats] + [np.concatenate([s.full_splats[:, :3] + off,
                                                       s.full_splats[:, 3:]], axis=1)
                                      for off in offsets]
    tiled_splats = np.concatenate(splat_copies, axis=0)

    mesh_copies = [s.interleaved_mesh]
    for off in offsets:
        c = s.interleaved_mesh.copy(); c[:, :3] += off; mesh_copies.append(c)
    tiled_mesh = np.concatenate(mesh_copies, axis=0)

    n_verts_per = len(s.interleaved_mesh)
    idx_copies = [s.mesh_indices] + [s.mesh_indices + n_verts_per * (i + 1)
                                       for i in range(len(offsets))]
    tiled_indices = np.concatenate(idx_copies, axis=0).astype(np.int32)

    s.full_splats = tiled_splats
    s.base_splats = tiled_splats.copy()
    s.splat_pos = np.ascontiguousarray(tiled_splats[:, :3])
    s.total_splats = len(tiled_splats)
    s.current_splats = tiled_splats
    s.num_splats = len(tiled_splats)
    s.splat_tree = cKDTree(s.splat_pos)
    s.interleaved_mesh = tiled_mesh
    s.mesh_indices = tiled_indices
    s.mesh_vbo = ctx.buffer(tiled_mesh.tobytes())
    s.mesh_ibo = ctx.buffer(tiled_indices.tobytes())

    nb = tiled_splats.astype('f4').tobytes()
    s.splat_vbo_a = ctx.buffer(nb)
    s.splat_vbo_b = ctx.buffer(nb)
    s.render_vbo = s.splat_vbo_a
    s.upload_vbo = s.splat_vbo_b

    print(f"[Star] {len(offsets)+1} instances -> {s.total_splats:,} splats, "
          f"{len(tiled_indices)//3:,} triangles")
    return s

def apply_model_rotation(s, rot_mat3):
    from scipy.spatial.transform import Rotation as R

    base = s.base_splats
    rotated = base.copy()
    glsl_rot = rot_mat3.T

    rotated[:, :3] = base[:, :3] @ glsl_rot.T

    q_model = R.from_matrix(glsl_rot).as_quat()  # [x,y,z,w]
    mx, my, mz, mw = float(q_model[0]), float(q_model[1]), float(q_model[2]), float(q_model[3])

    sx, sy, sz, sw = base[:,10], base[:,11], base[:,12], base[:,13]
    rotated[:,10] = mw*sx + mx*sw + my*sz - mz*sy
    rotated[:,11] = mw*sy - mx*sz + my*sw + mz*sx
    rotated[:,12] = mw*sz + mx*sy - my*sx + mz*sw
    rotated[:,13] = mw*sw - mx*sx - my*sy - mz*sz

    s.full_splats = rotated
    s.current_splats = rotated
    s.splat_pos = np.ascontiguousarray(rotated[:, :3])
    s.splat_tree = cKDTree(s.splat_pos)
    s.num_splats = len(rotated)
    nb = rotated.astype('f4').tobytes()
    s.splat_vbo_a.write(nb)
    s.splat_vbo_b.write(nb)

# 1st cull what's not in sphere
def cull_splats_sphere(s, center_pt, radius):
    idxs = s.splat_tree.query_ball_point(center_pt, radius)
    if not idxs:
        return np.empty((0, s.full_splats.shape[1]), dtype=np.float32)
    return s.full_splats[idxs].copy()
    
# then update the buffers, so they contain only what's in the sphere
def update_splat_buffers(s, new_splats):
    if len(new_splats) == 0:
        new_splats = np.zeros((1, s.full_splats.shape[1]), dtype=np.float32)
    s.current_splats = new_splats
    s.num_splats = len(new_splats)
    s._last_rendered_sorted = None