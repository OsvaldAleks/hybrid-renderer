import os
import glob
import numpy as np
from plyfile import PlyData
from PIL import Image

def discover_models(root):
    models = []
    for folder in sorted(os.listdir(root)):
        fpath = os.path.join(root, folder)
        if not os.path.isdir(fpath):
            continue
        plys = glob.glob(os.path.join(fpath, "refined_ply", "*.ply"))
        objs = glob.glob(os.path.join(fpath, "refined_mesh", "*.obj"))
        if plys and objs:
            models.append((folder, plys[0], objs[0]))
    return models

def load_splats(filename):
    plydata = PlyData.read(filename)
    vert = plydata['vertex']
    x = np.asarray(vert['x'], dtype=np.float32)
    y = np.asarray(vert['y'], dtype=np.float32)
    z = np.asarray(vert['z'], dtype=np.float32)
    scales = np.exp(np.stack(
        [np.asarray(vert[f'scale_{i}'], dtype=np.float32) for i in range(3)], axis=-1))
    alpha = 1.0 / (1.0 + np.exp(-np.asarray(vert['opacity'], dtype=np.float32)))
    features_dc = np.stack(
        [np.asarray(vert[f'f_dc_{i}'], dtype=np.float32) for i in range(3)], axis=-1)
    color = 1.0 / (1.0 + np.exp(-features_dc))
    quat_wxyz = np.stack(
        [np.asarray(vert[f'rot_{i}'], dtype=np.float32) for i in range(4)], axis=-1)
    quat_wxyz /= np.where(
        (n := np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)) == 0, 1.0, n)
    data = np.stack([x, y, z,
                     scales[:, 0], scales[:, 1], scales[:, 2],
                     color[:, 0], color[:, 1], color[:, 2], alpha,
                     quat_wxyz[:, 1], quat_wxyz[:, 2],
                     quat_wxyz[:, 3], quat_wxyz[:, 0]], axis=-1)
    return data.astype(np.float32)

def load_obj_with_mtl(obj_path):
    vertices, uvs_list, faces_raw = [], [], []
    mtllib_name = None
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('vt '):
                p = line.split()
                uvs_list.append([float(p[1]), float(p[2])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                for i in range(1, len(parts) - 1):
                    tri = []
                    for token in [parts[0], parts[i], parts[i+1]]:
                        vals = token.split('/')
                        vi  = int(vals[0]) - 1
                        uvi = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else -1
                        tri.append((vi, uvi))
                    faces_raw.append(tri)
            elif line.startswith('mtllib '):
                mtllib_name = line.split()[1]

    n_verts  = len(vertices)
    vert_pos = np.array(vertices, dtype=np.float32)

    vi_arr = np.array([[f[0][0], f[1][0], f[2][0]] for f in faces_raw], dtype=np.int32)
    v0 = vert_pos[vi_arr[:, 0]].astype(np.float64)
    v1 = vert_pos[vi_arr[:, 1]].astype(np.float64)
    v2 = vert_pos[vi_arr[:, 2]].astype(np.float64)
    fn = np.cross(v1 - v0, v2 - v0)
    vert_norm = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(vert_norm, vi_arr[:, 0], fn)
    np.add.at(vert_norm, vi_arr[:, 1], fn)
    np.add.at(vert_norm, vi_arr[:, 2], fn)
    norm_len = np.linalg.norm(vert_norm, axis=1, keepdims=True)
    vert_norm = (vert_norm / np.maximum(norm_len, 1e-8)).astype(np.float32)

    vert_col = np.full((n_verts, 3), 0.7, dtype=np.float32)
    interleaved_col = np.hstack([vert_pos, vert_norm, vert_col]).astype(np.float32)

    uvs_arr = np.array(uvs_list, dtype=np.float32) if uvs_list else np.zeros((1, 2), dtype=np.float32)
    pos_uv, norm_uv, uv_uv = [], [], []
    for face in faces_raw:
        (vi0, uvi0), (vi1, uvi1), (vi2, uvi2) = face
        for vi, uvi in [(vi0, uvi0), (vi1, uvi1), (vi2, uvi2)]:
            pos_uv.append(vert_pos[vi])
            norm_uv.append(vert_norm[vi])
            uv_uv.append(uvs_arr[uvi] if uvi >= 0 else [0., 0.])
    interleaved_uv = np.hstack([
        np.array(pos_uv,  dtype=np.float32),
        np.array(norm_uv, dtype=np.float32),
        np.array(uv_uv,   dtype=np.float32),
    ]).astype(np.float32)
    indices_uv = np.arange(len(pos_uv), dtype=np.int32)

    tex_path = None
    if mtllib_name:
        obj_dir  = os.path.dirname(obj_path) or '.'
        mtl_path = os.path.join(obj_dir, mtllib_name)
        if os.path.exists(mtl_path):
            with open(mtl_path) as mtl:
                for line in mtl:
                    if line.strip().startswith('map_Kd '):
                        candidate = os.path.join(obj_dir, line.strip().split()[1])
                        if os.path.exists(candidate):
                            tex_path = candidate

    return (interleaved_col, vi_arr.ravel(),
            interleaved_uv, indices_uv,
            vert_pos, tex_path)

def load_texture(ctx, tex_path):
    import moderngl
    if tex_path:
        img = Image.open(tex_path).convert('RGBA').transpose(Image.FLIP_TOP_BOTTOM)
        t = ctx.texture((img.width, img.height), 4, np.array(img).tobytes())
        t.repeat_x = t.repeat_y = True
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    else:
        t = ctx.texture((1, 1), 4, np.array([200, 200, 200, 255], dtype=np.uint8).tobytes())
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return t