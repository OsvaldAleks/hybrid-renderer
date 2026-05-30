import time
import numpy as np
import moderngl

from config import NEAR, FAR, RENDER_HYBRID, RENDER_SPLAT_ONLY, RENDER_MESH_ONLY
from rendering.shaders import make_splat_vao

def pass_mesh(ctx, sh, fbos, scene, mesh_vao, mesh_color_mode,
              model_mat, view, proj, ambient, light_dir, width, height):
    fbos.mesh_fbo.use()
    ctx.clear(0.1, 0.1, 0.1, 1.0)
    ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True

    prog = sh.mesh_prog if mesh_color_mode == 'vertex' else sh.mesh_uv_prog
    prog['model'].write(model_mat.astype('f4').tobytes())
    prog['view'].write(view.astype('f4').tobytes())
    prog['proj'].write(proj.astype('f4').tobytes())
    prog['ambient'].write(ambient.tobytes())
    prog['light_dir'].write(light_dir.tobytes())
    prog['diffuse_str'].value = 0.0
    if mesh_color_mode == 'uv':
        scene.mesh_texture.use(location=0)
        prog['tex'].value = 0
        mesh_vao.render(moderngl.TRIANGLES, vertices=scene.mesh_uv_vert_count)
    else:
        mesh_vao.render(moderngl.TRIANGLES)

    ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
    fbos.mesh_resolved_fbo.use()
    ctx.viewport = (0, 0, width, height)
    fbos.mesh_color.use(location=0)
    sh.blit_prog['src'] = 0
    sh.blit_vao.render(moderngl.TRIANGLES)

def pass_dof(ctx, sh, fbos, render_mode, dof_enabled, freq_blend_enabled, eff_focal, focal_range, max_coc, width, height, qw, qh):
    mesh_color_src = fbos.mesh_resolved
    post_mesh_tex = mesh_color_src
    lf_tex = None
    need_blit_mesh = render_mode != RENDER_HYBRID

    ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)

    if dof_enabled:
        fbos.coc_fbo.use()
        ctx.viewport = (0, 0, width, height)
        fbos.mesh_depth.use(location=0)
        sh.coc_prog['depth_tex'] = 0
        sh.coc_prog['focal_distance'] = eff_focal
        sh.coc_prog['focal_range'] = focal_range
        sh.coc_prog['max_coc'] = max_coc
        sh.coc_prog['near'] = NEAR
        sh.coc_prog['far'] = FAR
        sh.coc_vao.render(moderngl.TRIANGLES)

        for src_tex, dst_fbo, offset in [
            (mesh_color_src, fbos.kaw_a_fbo, 0.5),
            (fbos.kaw_a, fbos.kaw_b_fbo, 1.5),
            (fbos.kaw_b, fbos.kaw_a_fbo, 2.5),
            (fbos.kaw_a, fbos.kaw_b_fbo, 4.5),
        ]:
            dst_fbo.use(); ctx.viewport = (0, 0, qw, qh)
            src_tex.use(location=0)
            sh.kawase_prog['color_tex'] = 0
            sh.kawase_prog['texel'] = (1.0/qw, 1.0/qh)
            sh.kawase_prog['kawase_offset'] = offset
            sh.kawase_vao.render(moderngl.TRIANGLES)
        dof_blur_tex = fbos.kaw_b

        fbos.dof_out_fbo.use()
        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.1, 0.1, 0.1, 1.0)
        ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        mesh_color_src.use(location=0)
        dof_blur_tex.use(location=1)
        fbos.coc_tex.use(location=2)
        sh.dof_prog['sharp_tex'] = 0
        sh.dof_prog['blur_tex'] = 1
        sh.dof_prog['coc_tex'] = 2
        sh.dof_prog['max_coc'] = max_coc
        sh.dof_vao.render(moderngl.TRIANGLES)

        if render_mode == RENDER_MESH_ONLY:
            ctx.screen.use(); ctx.viewport = (0, 0, width, height)
            ctx.clear(0.1, 0.1, 0.1, 1.0)
            fbos.dof_out_tex.use(location=0)
            sh.blit_prog['src'] = 0
            sh.blit_vao.render(moderngl.TRIANGLES)
            need_blit_mesh = False
        else:
            fbos.mesh_resolved_fbo.use(); ctx.viewport = (0, 0, width, height)
            fbos.dof_out_tex.use(location=0)
            sh.blit_prog['src'] = 0
            sh.blit_vao.render(moderngl.TRIANGLES)
            mesh_color_src = fbos.mesh_resolved
            post_mesh_tex  = fbos.mesh_resolved

    if render_mode == RENDER_HYBRID and freq_blend_enabled:
        for src_tex, dst_fbo, offset in [
            (mesh_color_src, fbos.kaw_a_fbo, 0.5),
            (fbos.kaw_a, fbos.kaw_b_fbo, 1.5),
        ]:
            dst_fbo.use(); ctx.viewport = (0, 0, qw, qh)
            src_tex.use(location=0)
            sh.kawase_prog['color_tex'] = 0
            sh.kawase_prog['texel'] = (1.0/qw, 1.0/qh)
            sh.kawase_prog['kawase_offset'] = offset
            sh.kawase_vao.render(moderngl.TRIANGLES)
        lf_tex = fbos.kaw_b

    if need_blit_mesh and render_mode == RENDER_MESH_ONLY:
        ctx.screen.use(); ctx.viewport = (0, 0, width, height)
        ctx.clear(0.1, 0.1, 0.1, 1.0)
        ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        post_mesh_tex.use(location=0); sh.blit_prog['src'] = 0
        sh.blit_vao.render(moderngl.TRIANGLES)

    return post_mesh_tex, lf_tex

def pass_splats(ctx, sh, fbos, scene, render_mode, follow_camera, view, proj, cam_fwd, cam_pos, cone_inner_angle, cone_outer_angle, cone_max_depth, width, height, proj_np, hybrid_sorter, full_sorter, render_vao_obj, upload_vao_obj):
    ctx.enable(moderngl.BLEND)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.depth_mask = False

    if render_mode == RENDER_HYBRID:
        fbos.splat_fbo.use()
        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        sh.splat_prog['premult_output'] = 1
    else:
        sh.splat_prog['premult_output'] = 0

    fbos.mesh_depth.use(location=2)
    sh.splat_prog['mesh_depth_tex'] = 2
    sh.splat_prog['view'].write(view.astype('f4').tobytes())
    sh.splat_prog['proj'].write(proj.astype('f4').tobytes())

    if render_mode == RENDER_SPLAT_ONLY:
        sorted_data = full_sorter.sort(scene, view, data_override=scene.full_splats)
        fbos.dummy_depth.use(location=2)
        sh.splat_prog['mesh_depth_tex'] = 2
    elif follow_camera:
        sorted_data = hybrid_sorter.sort(
            scene, view, cone_mode=True,
            cam_fwd=cam_fwd, cam_pos_w=cam_pos,
            cone_inner_deg=cone_inner_angle,
            cone_outer_deg=cone_outer_angle,
            cone_max_depth=cone_max_depth,
            aspect=float(width) / float(height),
            fy=float(proj_np[1, 1]))
    else:
        sorted_data = hybrid_sorter.sort(scene, view, cone_mode=False)

    n_draw = len(sorted_data)
    if n_draw > 0:
        if sorted_data is not getattr(scene, '_last_rendered_sorted', None):
            buf = sorted_data.astype('f4', copy=False)
            scene.upload_vbo.write(memoryview(buf))
            scene.render_vbo, scene.upload_vbo = scene.upload_vbo, scene.render_vbo
            render_vao_obj = make_splat_vao(ctx, sh, scene.render_vbo)
            upload_vao_obj = make_splat_vao(ctx, sh, scene.upload_vbo)
            scene._last_rendered_sorted = sorted_data
        render_vao_obj.render(moderngl.POINTS, vertices=n_draw)

    scene.num_splats = n_draw
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    return render_vao_obj, upload_vao_obj, n_draw

def pass_composite(ctx, sh, fbos, post_mesh_tex, lf_tex, freq_blend_enabled, freq_bleed, freq_debug, width, height, qw, qh):
    ctx.disable(moderngl.BLEND); ctx.disable(moderngl.DEPTH_TEST)
    ctx.screen.use()
    ctx.viewport = (0, 0, width, height)
    ctx.clear(0.1, 0.1, 0.1, 1.0)

    if freq_blend_enabled and lf_tex is not None:
        for src_tex, dst_fbo, offset in [
            (fbos.splat_tex, fbos.kaw_a_fbo, 0.5),
            (fbos.kaw_a, fbos.splat_lf_fbo, 1.5),
        ]:
            dst_fbo.use(); ctx.viewport = (0, 0, qw, qh)
            src_tex.use(location=0)
            sh.kawase_prog['color_tex'] = 0; sh.kawase_prog['texel'] = (1.0/qw, 1.0/qh)
            sh.kawase_prog['kawase_offset'] = offset; sh.kawase_vao.render(moderngl.TRIANGLES)

        for src_tex, dst_fbo, offset in [
            (fbos.splat_lf, fbos.kaw_a_fbo, 2.5),
            (fbos.kaw_a, fbos.kaw_b_fbo, 4.5),
            (fbos.kaw_b, fbos.kaw_a_fbo, 8.5),
            (fbos.kaw_a, fbos.splat_lf2_fbo, 16.5),
        ]:
            dst_fbo.use(); ctx.viewport = (0, 0, qw, qh)
            src_tex.use(location=0)
            sh.kawase_prog['color_tex'] = 0
            sh.kawase_prog['kawase_offset'] = offset
            sh.kawase_vao.render(moderngl.TRIANGLES)

        ctx.screen.use(); ctx.viewport = (0, 0, width, height)
        ctx.clear(0.1, 0.1, 0.1, 1.0)
        post_mesh_tex.use(location=0)
        fbos.splat_tex.use(location=1)
        fbos.splat_lf.use(location=2)
        fbos.splat_lf2.use(location=3)
        sh.freq_prog['freq_aware'] = 1
        sh.freq_prog['freq_bleed'] = freq_bleed
        sh.freq_prog['debug_freq'] = 1 if freq_debug else 0
    else:
        post_mesh_tex.use(location=0)
        fbos.splat_tex.use(location=1)
        fbos.splat_tex.use(location=2)
        fbos.splat_tex.use(location=3)
        sh.freq_prog['freq_aware'] = 0
        sh.freq_prog['freq_bleed'] = 1.0
        sh.freq_prog['debug_freq'] = 0

    sh.freq_prog['mesh_tex'] = 0
    sh.freq_prog['splat_tex'] = 1
    sh.freq_prog['splat_lf1_tex'] = 2
    sh.freq_prog['splat_lf2_tex'] = 3
    sh.freq_vao.render(moderngl.TRIANGLES)

def pass_zone_overlay(ctx, sh, follow_camera, fy, fx, cone_inner_angle, cone_outer_angle, sphere_radius, sphere_fade, fixed_center, view, proj, width, height):
    ctx.screen.use()
    ctx.viewport = (0, 0, width, height)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.disable(moderngl.DEPTH_TEST)

    if follow_camera:
        tan_i = float(np.tan(np.deg2rad(cone_inner_angle)))
        tan_o = float(np.tan(np.deg2rad(cone_outer_angle)))
        sh.zone_vis_prog['inner_x'] = tan_i * fy
        sh.zone_vis_prog['inner_y'] = tan_i * fy
        sh.zone_vis_prog['outer_x'] = tan_o * fy
        sh.zone_vis_prog['outer_y'] = tan_o * fy
        sh.zone_vis_prog['center_ndc'] = (0.0, 0.0)
    else:
        view_np = np.array(view,  dtype=np.float32)
        proj_np = np.array(proj,  dtype=np.float32)
        center_h = np.array([*fixed_center, 1.0], dtype=np.float32)
        clip = center_h @ view_np @ proj_np
        cx_ndc = float(clip[0] / clip[3]) if abs(clip[3]) > 1e-6 else 0.0
        cy_ndc = float(clip[1] / clip[3]) if abs(clip[3]) > 1e-6 else 0.0
        depth = max(abs(float((center_h @ view_np)[2])), 0.01)
        sh.zone_vis_prog['inner_x'] = float(sphere_radius * fx / depth)
        sh.zone_vis_prog['inner_y'] = float(sphere_radius * fy / depth)
        sh.zone_vis_prog['outer_x'] = float((sphere_radius + sphere_fade) * fx / depth)
        sh.zone_vis_prog['outer_y'] = float((sphere_radius + sphere_fade) * fy / depth)
        sh.zone_vis_prog['center_ndc'] = (cx_ndc, cy_ndc)

    sh.zone_vis_vao.render(moderngl.TRIANGLES)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA