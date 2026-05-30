import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import moderngl
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

from config import (
    DATA_ROOT, NEAR, FAR, RENDER_HYBRID, RENDER_SPLAT_ONLY, RENDER_MESH_ONLY,
    TIMING_WINDOW, CONE_INNER_ANGLE, CONE_OUTER_ANGLE, CONE_MAX_DEPTH,
    AUTO_DEPTH_SCALE, SPHERE_RADIUS, SPHERE_FADE,
    DOF_FOCAL_DISTANCE, DOF_FOCAL_RANGE, DOF_MAX_COC, DOF_APERTURE,
    DEPTH_MARGIN, FREQ_BLEED, AMBIENT, LIGHT_DIR,
)
from scene.loader import discover_models
from scene.scene import load_scene
from scene.transforms import apply_star_replication, apply_model_rotation, \
                             cull_splats_sphere, update_splat_buffers
from sorting.sorter import SplatSorter, _extract_view_direction
from rendering.shaders import build_shaders, make_mesh_vao, make_mesh_vao_uv, make_splat_vao
from rendering.fbos import build_fbos
from rendering.passes import (pass_mesh, pass_dof, pass_splats,
                                pass_composite, pass_zone_overlay)
from camera.camera import (init as cam_init, mouse_move_cb, scroll_cb,
                               get_view_proj, build_model_mat,
                               model_yaw, model_pitch)
import camera.camera as cam
from camera.depth_reader import CentreDepthReader

# Init GLFW + ModernGL + ImGui
ALL_MODELS = discover_models(DATA_ROOT)
if not ALL_MODELS:
    raise RuntimeError(f"No models found under {DATA_ROOT}")

if not glfw.init():
    raise RuntimeError("GLFW init failed")
window = glfw.create_window(1280, 720, "Hybrid Renderer", None, None)
glfw.make_context_current(window)

ctx = moderngl.create_context()
ctx.enable(moderngl.DEPTH_TEST)
ctx.depth_func = '<='
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

imgui.create_context()
impl = GlfwRenderer(window)
io   = imgui.get_io()
io.display_size = glfw.get_framebuffer_size(window)

cam_init(io)
glfw.set_cursor_pos_callback(window, mouse_move_cb)
glfw.set_scroll_callback(window, scroll_cb)

# Build GPU resources
sh   = build_shaders(ctx)
width, height = glfw.get_framebuffer_size(window)
fbos = build_fbos(ctx, width, height)

depth_reader = CentreDepthReader()

# App state
current_model_idx  = next((i for i, m in enumerate(ALL_MODELS) if m[0] == 'truck'), 0)
pending_reload     = True
scene              = None
mesh_vao_obj       = None
render_vao_obj     = None
upload_vao_obj     = None

hybrid_sorter = SplatSorter()
full_sorter   = SplatSorter()

follow_camera      = True
fixed_center       = np.zeros(3, np.float32)
cone_inner_angle   = CONE_INNER_ANGLE
cone_outer_angle   = CONE_OUTER_ANGLE
cone_max_depth     = CONE_MAX_DEPTH
auto_depth         = True
auto_depth_scale   = AUTO_DEPTH_SCALE
sphere_radius      = SPHERE_RADIUS
sphere_fade        = SPHERE_FADE
last_sphere_params = (-1.0, -1.0)
center_changed     = False
center_depth       = None
last_follow_camera = True
last_cone_params   = (cone_inner_angle, cone_outer_angle, cone_max_depth)

dof_enabled     = True
focal_distance  = DOF_FOCAL_DISTANCE
focal_range     = DOF_FOCAL_RANGE
max_coc         = DOF_MAX_COC
aperture        = DOF_APERTURE
depth_margin    = DEPTH_MARGIN

render_mode        = RENDER_HYBRID
freq_blend_enabled = False
freq_bleed         = FREQ_BLEED
freq_debug         = False

model_mat      = build_model_mat()
last_model_rot = (cam.model_yaw + 1.0, cam.model_pitch)
ambient        = AMBIENT.copy()
light_dir      = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
debug_uv       = False
show_zone      = False
mesh_color_mode = 'vertex'
star_rings      = 0

t_frame = np.zeros(TIMING_WINDOW); t_mesh  = np.zeros(TIMING_WINDOW)
t_dof   = np.zeros(TIMING_WINDOW); t_splat = np.zeros(TIMING_WINDOW)
t_freq  = np.zeros(TIMING_WINDOW)
t_idx   = 0
t_prev  = time.perf_counter()
last_fb_size   = (width, height)
frame_count    = 0
last_time_wall = time.time()

# Main loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    impl.process_inputs()

    width, height = glfw.get_framebuffer_size(window)
    ctx.viewport  = (0, 0, width, height)
    qw = max(width  // 4, 1)
    qh = max(height // 4, 1)

    # Rebuild fbos if window size changed
    if (width, height) != last_fb_size:
        fbos = build_fbos(ctx, width, height)
        last_fb_size = (width, height)

    proj, view, cam_pos = get_view_proj(width, height)

    model_mat = build_model_mat()
    if scene is not None and (cam.model_yaw, cam.model_pitch) != last_model_rot:
        apply_model_rotation(scene, np.array(model_mat)[:3, :3])
        hybrid_sorter.flush(); full_sorter.flush()
        last_model_rot = (cam.model_yaw, cam.model_pitch)

    if pending_reload:
        for _s in (hybrid_sorter, full_sorter):
            _s.flush()
        name, ply_path, obj_path = ALL_MODELS[current_model_idx]
        scene = load_scene(name, ply_path, obj_path, ctx)
        if star_rings > 0:
            scene = apply_star_replication(scene, ctx, n_rings=star_rings)
        mesh_vao_obj   = make_mesh_vao(ctx, sh, scene) if mesh_color_mode == 'vertex' \
                         else make_mesh_vao_uv(ctx, sh, scene)
        render_vao_obj = make_splat_vao(ctx, sh, scene.render_vbo)
        upload_vao_obj = make_splat_vao(ctx, sh, scene.upload_vbo)
        last_sphere_params = (-1.0, -1.0)
        fixed_center[:] = 0.0; follow_camera = True
        cone_max_depth  = float(scene.inner_radius)
        sphere_radius   = float(scene.outer_radius)
        sphere_fade     = float(scene.fade_width)
        center_depth    = None; pending_reload = False
        last_follow_camera = True
        last_cone_params   = (cone_inner_angle, cone_outer_angle, cone_max_depth)
        last_model_rot     = (cam.model_yaw + 1.0, cam.model_pitch)
        scene._last_rendered_sorted = None
        print(f"LOADED {scene.total_splats:,} splats")

    cam_fwd = _extract_view_direction(np.array(view, dtype=np.float32))
    cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd) + 1e-12)

    # ImGui controls
    imgui.new_frame()
    imgui.set_next_window_position(10, 10, imgui.ONCE)
    imgui.set_next_window_size(310, 0, imgui.ONCE)
    imgui.begin("Controls", True)

    imgui.text("Model")
    imgui.push_item_width(-1)
    model_names = [m[0] for m in ALL_MODELS]
    changed_model, new_idx = imgui.combo("##model", current_model_idx, model_names)
    imgui.pop_item_width()
    if changed_model and new_idx != current_model_idx:
        current_model_idx = new_idx; pending_reload = True

    imgui.push_item_width(185)
    _, cam.model_yaw   = imgui.drag_float("Yaw##rot",   cam.model_yaw,   0.5, -180.0, 180.0, "%.1f deg")
    _, cam.model_pitch = imgui.drag_float("Pitch##rot", cam.model_pitch, 0.5,  -89.0,  89.0, "%.1f deg")
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button("Reset##rot"):
        cam.model_yaw = 0.0; cam.model_pitch = 0.0
    if imgui.is_item_hovered():
        imgui.set_tooltip("Right-click drag to rotate")
    imgui.separator()

    imgui.text("Mode")
    if imgui.radio_button("Hybrid##m",    render_mode == RENDER_HYBRID):     render_mode = RENDER_HYBRID
    imgui.same_line()
    if imgui.radio_button("Splat##m",     render_mode == RENDER_SPLAT_ONLY): render_mode = RENDER_SPLAT_ONLY
    imgui.same_line()
    if imgui.radio_button("Mesh##m",      render_mode == RENDER_MESH_ONLY):  render_mode = RENDER_MESH_ONLY
    imgui.separator()

    open_mesh, _ = imgui.collapsing_header("Mesh Shading")
    if open_mesh:
        imgui.push_item_width(160)
        _, ambient[0] = imgui.slider_float("Ambient##m", ambient[0], 0.0, 1.0)
        imgui.pop_item_width()
        ambient[1] = ambient[2] = ambient[0]
        imgui.separator()
        imgui.text("Colour mode:")
        imgui.same_line()
        prev_mode = mesh_color_mode
        if imgui.radio_button("Vertex (splat)##cm", mesh_color_mode == 'vertex'):
            mesh_color_mode = 'vertex'
        imgui.same_line()
        if imgui.radio_button("UV (texture)##cm", mesh_color_mode == 'uv'):
            mesh_color_mode = 'uv'
        if mesh_color_mode != prev_mode and scene is not None:
            mesh_vao_obj = make_mesh_vao(ctx, sh, scene) if mesh_color_mode == 'vertex' \
                           else make_mesh_vao_uv(ctx, sh, scene)

    open_zone, _ = imgui.collapsing_header("Splat Zone")
    if open_zone:
        _, follow_camera = imgui.checkbox("Follow camera", follow_camera)
        if follow_camera:
            imgui.push_item_width(175)
            _, cone_inner_angle = imgui.slider_float("Inner angle##cull", cone_inner_angle, 5.0, 85.0)
            _, cone_outer_angle = imgui.slider_float("Outer angle##cull", cone_outer_angle, 5.0, 90.0)
            cone_outer_angle = max(cone_outer_angle, cone_inner_angle + 1.0)
            imgui.pop_item_width()
            _, auto_depth = imgui.checkbox("Auto depth##cull", auto_depth)
            if auto_depth:
                imgui.push_item_width(175)
                _, auto_depth_scale = imgui.slider_float("Depth scale##cull", auto_depth_scale, 1.0, 5.0)
                imgui.pop_item_width()
                d_str = f"{center_depth:.2f}" if center_depth else "--"
                imgui.text(f"Centre depth: {d_str}  cone: {cone_max_depth:.2f}")
            else:
                imgui.push_item_width(175)
                _, cone_max_depth = imgui.slider_float("Max depth##cull", cone_max_depth, 0.5, 30.0)
                imgui.pop_item_width()
        else:
            imgui.text("Center")
            imgui.push_item_width(75)
            cx, fixed_center[0] = imgui.drag_float("X##c", fixed_center[0], 0.05, -20, 20)
            imgui.same_line()
            cy, fixed_center[1] = imgui.drag_float("Y##c", fixed_center[1], 0.05, -20, 20)
            imgui.same_line()
            cz, fixed_center[2] = imgui.drag_float("Z##c", fixed_center[2], 0.05, -20, 20)
            imgui.pop_item_width()
            center_changed = cx or cy or cz
            imgui.push_item_width(175)
            _, sphere_radius = imgui.slider_float("Radius##sr", sphere_radius, 0.5, 20.0)
            _, sphere_fade   = imgui.slider_float("Fade##sr",   sphere_fade,   0.1,  5.0)
            imgui.pop_item_width()
        imgui.text(f"Visible: {scene.num_splats:,} / {scene.total_splats:,}")
        imgui.separator()
        _, show_zone = imgui.checkbox("Show zone overlay", show_zone)
        _, debug_uv  = imgui.checkbox("Show UV as colour", debug_uv)

    open_occ, _ = imgui.collapsing_header("Depth Occlusion")
    if open_occ:
        imgui.push_item_width(180)
        _, depth_margin = imgui.slider_float("Margin##o", depth_margin, 0.01, 10.0)
        imgui.pop_item_width()

    open_dof, _ = imgui.collapsing_header("Depth of Field")
    if open_dof:
        _, dof_enabled = imgui.checkbox("Enable DoF", dof_enabled)
        if dof_enabled:
            imgui.push_item_width(180)
            if auto_depth and follow_camera:
                d_str = f"{eff_focal:.2f}" if center_depth else "--"
                imgui.text(f"Focal dist (auto): {d_str}")
            elif follow_camera:
                _, focal_distance = imgui.drag_float("Focal dist##d", focal_distance, 0.1, 0.5, 50.0)
            else:
                imgui.text(f"Focal dist (auto): {np.linalg.norm(cam_pos - fixed_center):.2f}")
            _, focal_range   = imgui.drag_float("Sharp zone##d", focal_range,  0.05, 0.0, 5.0)
            chg_ap, aperture = imgui.drag_float("Aperture##d",   aperture,     0.005, 0.0, 0.5)
            if chg_ap: max_coc = aperture * 20.0
            imgui.pop_item_width()
        imgui.text("Full-res CoC, 4-pass 1/4-res Kawase blur")

    open_freq, _ = imgui.collapsing_header("Splat -> Mesh Transition")
    if open_freq:
        if render_mode != RENDER_HYBRID:
            imgui.text("(Hybrid mode only)")
        else:
            _, freq_blend_enabled = imgui.checkbox("Freq-domain blend##freq", freq_blend_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip("On: sharp HF, smooth LF.\nOff: linear alpha blend.")
            if freq_blend_enabled:
                imgui.push_item_width(175)
                _, freq_bleed = imgui.slider_float("Bleed##freq", freq_bleed, 0.5, 16.0, "%.1f")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("How far LF blend bleeds past cone edge.")
                imgui.pop_item_width()
            _, freq_debug = imgui.checkbox("Debug: show HF band##freq", freq_debug)

    open_adv, _ = imgui.collapsing_header("Sort & Debug")
    if open_adv:
        imgui.push_item_width(160)
        _, hybrid_sorter.angle_threshold = imgui.drag_float(
            "Sort angle##s", hybrid_sorter.angle_threshold, 0.001, 0.0, 0.2, "%.3f")
        _, hybrid_sorter.dist_threshold  = imgui.drag_float(
            "Sort dist##s",  hybrid_sorter.dist_threshold,  0.005, 0.0, 2.0, "%.3f")
        _, hybrid_sorter.cull_angle_threshold = imgui.drag_float(
            "Cull angle##s", hybrid_sorter.cull_angle_threshold, 0.005, 0.0, 0.5, "%.3f")
        _, hybrid_sorter.cull_dist_threshold  = imgui.drag_float(
            "Cull dist##s",  hybrid_sorter.cull_dist_threshold,  0.05,  0.0, 5.0, "%.2f")
        imgui.pop_item_width()
        full_sorter.angle_threshold      = hybrid_sorter.angle_threshold
        full_sorter.dist_threshold       = hybrid_sorter.dist_threshold
        full_sorter.cull_angle_threshold = hybrid_sorter.cull_angle_threshold
        full_sorter.cull_dist_threshold  = hybrid_sorter.cull_dist_threshold

        imgui.separator()
        imgui.text("Star Replication (stress test)")
        imgui.push_item_width(120)
        chg_rings, new_rings = imgui.slider_int("Rings##star", star_rings, 0, 3)
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Cross pattern (no diagonals):\n"
                "0=off  1=+4 (N/S/E/W)=5 total\n"
                "2=+8=9 total  3=+12=13 total\nReloads on change.")
        if chg_rings and new_rings != star_rings:
            star_rings = new_rings; pending_reload = True
        n_inst = (2*star_rings+1)**2 if star_rings > 0 else 1
        if scene:
            imgui.text(f"Instances: {n_inst}  Splats: {scene.total_splats:,}")

    imgui.end()

    # Spatial culling / cache invalidation
    if follow_camera != last_follow_camera:
        hybrid_sorter.flush()
        last_sphere_params = (-1.0, -1.0)
    last_follow_camera = follow_camera

    if not follow_camera:
        cur_sphere = (sphere_radius, sphere_fade, *fixed_center)
        if cur_sphere != last_sphere_params or center_changed:
            outer_r  = sphere_radius + sphere_fade
            filtered = cull_splats_sphere(scene, fixed_center, outer_r)
            update_splat_buffers(scene, filtered)
            render_vao_obj     = make_splat_vao(ctx, sh, scene.render_vbo)
            upload_vao_obj     = make_splat_vao(ctx, sh, scene.upload_vbo)
            last_sphere_params = cur_sphere
    else:
        cur_cone = (cone_inner_angle, cone_outer_angle, cone_max_depth)
        if cur_cone != last_cone_params:
            hybrid_sorter.flush()
            last_cone_params = cur_cone
    center_changed = False

    # Performance panel
    af  = t_frame.mean(); am = t_mesh.mean()
    ad  = t_dof.mean();   asp = t_splat.mean(); afq = t_freq.mean()
    fps = 1000.0 / af if af > 0 else 0.0
    r   = min(1.0, max(0.0, 1.0 - (fps-25)/25))
    g   = min(1.0, max(0.0,       (fps-25)/25))
    cnt = scene.total_splats if render_mode == RENDER_SPLAT_ONLY else scene.num_splats

    imgui.set_next_window_position(10, height - 185, imgui.ALWAYS)
    imgui.set_next_window_size(265, 175, imgui.ALWAYS)
    imgui.begin("Performance", True,
                imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS)
    imgui.text_colored(f"FPS   {fps:6.1f}", r, g, 0.2, 1.0)
    imgui.text(         f"Frame {af:6.2f} ms")
    imgui.separator()
    imgui.text(["Hybrid","Splat Only","Mesh Only"][render_mode])
    if render_mode != RENDER_MESH_ONLY:
        imgui.text(f"Splat  {asp:6.2f} ms  {cnt:,} pts")
    if render_mode != RENDER_SPLAT_ONLY:
        imgui.text(f"Mesh   {am:6.2f} ms")
        if dof_enabled:
            imgui.text(f"DoF    {ad:6.2f} ms")
        if freq_blend_enabled and render_mode == RENDER_HYBRID:
            imgui.text(f"FreqBl {afq:6.2f} ms")
    imgui.end()

    # Splat zone uniforms
    proj_np = np.array(proj, dtype=np.float32)
    fy = float(proj_np[1, 1])
    fx = float(proj_np[0, 0])

    sh.splat_prog['debug_uv']             = 1 if debug_uv else 0
    sh.splat_prog['premult_output']       = 0
    sh.splat_prog['screen_size']          = (float(width), float(height))
    sh.splat_prog['depth_margin'].value   = depth_margin
    sh.splat_prog['near'].value           = NEAR
    sh.splat_prog['far'].value            = FAR
    sh.splat_prog['fade_max_depth'].value = cone_max_depth \
        if follow_camera and render_mode != RENDER_SPLAT_ONLY else 1e9

    if render_mode == RENDER_SPLAT_ONLY:
        sh.splat_prog['fade_inner_x'].value = 1e9
        sh.splat_prog['fade_inner_y'].value = 1e9
        sh.splat_prog['fade_outer_x'].value = 1e10
        sh.splat_prog['fade_outer_y'].value = 1e10
    elif follow_camera:
        tan_i = float(np.tan(np.deg2rad(cone_inner_angle)))
        tan_o = float(np.tan(np.deg2rad(cone_outer_angle)))
        sh.splat_prog['fade_inner_x'].value = tan_i * fy
        sh.splat_prog['fade_inner_y'].value = tan_i * fy
        sh.splat_prog['fade_outer_x'].value = tan_o * fy
        sh.splat_prog['fade_outer_y'].value = tan_o * fy
    else:
        view_np  = np.array(view, dtype=np.float32)
        center_h = np.array([*fixed_center, 1.0], dtype=np.float32)
        depth    = max(abs(float((center_h @ view_np)[2])), 0.01)
        sh.splat_prog['fade_inner_x'].value = float(sphere_radius * fx / depth)
        sh.splat_prog['fade_inner_y'].value = float(sphere_radius * fy / depth)
        sh.splat_prog['fade_outer_x'].value = float((sphere_radius + sphere_fade) * fx / depth)
        sh.splat_prog['fade_outer_y'].value = float((sphere_radius + sphere_fade) * fy / depth)

    # Auto depth / effective focal
    if auto_depth and follow_camera and render_mode != RENDER_SPLAT_ONLY:
        d = depth_reader.read(ctx, sh.quad_vbo, sh.quad_ibo,
                              fbos.mesh_depth, width, height, NEAR, FAR)
        if d is not None:
            center_depth   = d
            cone_max_depth = float(np.clip(d * auto_depth_scale, 1.0, FAR * 0.9))
    eff_focal = center_depth if (auto_depth and follow_camera and center_depth) \
                else (focal_distance if follow_camera
                      else float(np.linalg.norm(cam_pos - fixed_center)))

    # Render passes
    t0 = time.perf_counter()
    if render_mode != RENDER_SPLAT_ONLY:
        pass_mesh(ctx, sh, fbos, scene, mesh_vao_obj, mesh_color_mode,
                  model_mat, view, proj, ambient, light_dir, width, height)
    t_mesh[t_idx] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if render_mode == RENDER_SPLAT_ONLY:
        ctx.screen.use(); ctx.viewport = (0, 0, width, height)
        ctx.clear(0.1, 0.1, 0.1, 1.0)
        ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        post_mesh_tex, lf_tex = fbos.mesh_resolved, None
    else:
        post_mesh_tex, lf_tex = pass_dof(
            ctx, sh, fbos, render_mode, dof_enabled, freq_blend_enabled,
            eff_focal, focal_range, max_coc, width, height, qw, qh)
    t_dof[t_idx] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if render_mode != RENDER_MESH_ONLY:
        render_vao_obj, upload_vao_obj, _ = pass_splats(
            ctx, sh, fbos, scene, render_mode, follow_camera,
            view, proj, cam_fwd, cam_pos,
            cone_inner_angle, cone_outer_angle, cone_max_depth,
            width, height, proj_np,
            hybrid_sorter, full_sorter,
            render_vao_obj, upload_vao_obj)
    t_splat[t_idx] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if render_mode == RENDER_HYBRID:
        pass_composite(ctx, sh, fbos, post_mesh_tex, lf_tex,
                       freq_blend_enabled, freq_bleed, freq_debug,
                       width, height, qw, qh)
    t_freq[t_idx] = (time.perf_counter() - t0) * 1000.0

    if show_zone and render_mode != RENDER_SPLAT_ONLY:
        pass_zone_overlay(ctx, sh, follow_camera, fy, fx,
                          cone_inner_angle, cone_outer_angle,
                          sphere_radius, sphere_fade,
                          fixed_center, view, proj,
                          width, height)

    imgui.render()
    impl.render(imgui.get_draw_data())
    glfw.swap_buffers(window)

    t_now = time.perf_counter()
    t_frame[t_idx] = (t_now - t_prev) * 1000.0
    t_idx  = (t_idx + 1) % TIMING_WINDOW
    t_prev = t_now

    frame_count += 1
    now = time.time()
    if now - last_time_wall >= 1.0:
        fps_disp = frame_count / (now - last_time_wall)
        mode_str = ["Hybrid","Splat","Mesh"][render_mode]
        fb_str   = "+FreqBl" if (freq_blend_enabled and render_mode == RENDER_HYBRID) else ""
        glfw.set_window_title(window,
            f"Hybrid | {scene.name} | FPS:{fps_disp:.1f} | "
            f"{mode_str}{fb_str} | DoF:{'ON' if dof_enabled else 'OFF'}")
        frame_count    = 0
        last_time_wall = now

impl.shutdown()
glfw.terminate()