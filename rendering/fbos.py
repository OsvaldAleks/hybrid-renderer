import numpy as np
import moderngl
from types import SimpleNamespace

def mk_rgba_fbo(ctx, w, h, clamp=False):
    t = ctx.texture((w, h), 4, dtype='f4')
    t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        t.repeat_x = t.repeat_y = False
    return ctx.framebuffer(color_attachments=[t]), t

def mk_r_fbo(ctx, w, h, clamp=False):
    t = ctx.texture((w, h), 1, dtype='f4')
    t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        t.repeat_x = t.repeat_y = False
    return ctx.framebuffer(color_attachments=[t]), t

def mk_mesh_fbo(ctx, w, h):
    ct = ctx.texture((w, h), 4, dtype='f4')
    ct.filter = (moderngl.LINEAR, moderngl.LINEAR)
    dt = ctx.depth_texture((w, h))
    return ctx.framebuffer(color_attachments=[ct], depth_attachment=dt), ct, dt

def build_fbos(ctx, w, h):
    f  = SimpleNamespace()
    qw = max(w // 4, 1)
    qh = max(h // 4, 1)

    f.mesh_fbo, f.mesh_color, f.mesh_depth = mk_mesh_fbo(ctx, w, h)
    f.mesh_resolved_fbo, f.mesh_resolved = mk_rgba_fbo(ctx, w, h)
    f.dof_out_fbo, f.dof_out_tex = mk_rgba_fbo(ctx, w, h)
    f.coc_fbo, f.coc_tex = mk_r_fbo(ctx, w, h)
    f.kaw_a_fbo, f.kaw_a = mk_rgba_fbo(ctx, qw, qh, clamp=True)
    f.kaw_b_fbo, f.kaw_b = mk_rgba_fbo(ctx, qw, qh, clamp=True)
    f.mesh_lf2_fbo, f.mesh_lf2 = mk_rgba_fbo(ctx, qw, qh, clamp=True)
    f.splat_lf_fbo, f.splat_lf = mk_rgba_fbo(ctx, qw, qh, clamp=True)
    f.splat_lf2_fbo, f.splat_lf2= mk_rgba_fbo(ctx, qw, qh, clamp=True)
    f.splat_fbo, f.splat_tex= mk_rgba_fbo(ctx, w, h)

    f.dummy_depth = ctx.depth_texture((1, 1), data=np.ones(1, dtype=np.float32).tobytes())
    f.dummy_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return f