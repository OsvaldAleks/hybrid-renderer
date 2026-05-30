import numpy as np
import moderngl
from types import SimpleNamespace

_QUAD_VS = '''
    #version 330
    in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
    void main(){ gl_Position=vec4(in_pos,0,1); v_uv=in_uv; }
'''

_MESH_VERT = '''
    #version 330
    uniform mat4 model, view, proj;
    in vec3 in_pos; in vec3 in_normal; in vec3 in_color;
    out vec3 v_normal; out vec3 v_world_pos; out vec3 v_color;
    void main() {
        vec4 wp = model * vec4(in_pos, 1.0);
        v_world_pos = wp.xyz;
        v_normal = mat3(transpose(inverse(model))) * in_normal;
        v_color = in_color;
        gl_Position = proj * view * wp;
    }
'''

_MESH_FRAG = '''
    #version 330
    uniform vec3 ambient; uniform vec3 light_dir; uniform float diffuse_str;
    in vec3 v_normal; in vec3 v_world_pos; in vec3 v_color;
    out vec4 f_color;
    void main() {
        vec3 N     = normalize(v_normal);
        float diff = max(dot(N, normalize(light_dir)), 0.0);
        vec3 color = v_color * (ambient + diff * diffuse_str);
        f_color    = vec4(color, 1.0);
    }
'''

_MESH_UV_VERT = '''
    #version 330
    uniform mat4 model, view, proj;
    in vec3 in_pos; in vec3 in_normal; in vec2 in_uv;
    out vec3 v_normal; out vec3 v_world_pos; out vec2 v_uv;
    void main() {
        vec4 wp     = model * vec4(in_pos, 1.0);
        v_world_pos = wp.xyz;
        v_normal    = mat3(transpose(inverse(model))) * in_normal;
        v_uv        = in_uv;
        gl_Position = proj * view * wp;
    }
'''

_MESH_UV_FRAG = '''
    #version 330
    uniform sampler2D tex;
    uniform vec3 ambient; uniform vec3 light_dir; uniform float diffuse_str;
    in vec3 v_normal; in vec3 v_world_pos; in vec2 v_uv;
    out vec4 f_color;
    void main() {
        vec4 base  = texture(tex, v_uv);
        vec3 N     = normalize(v_normal);
        float diff = max(dot(N, normalize(light_dir)), 0.0);
        vec3 color = base.rgb * (ambient + diff * diffuse_str);
        f_color    = vec4(color, 1.0);
    }
'''

_SPLAT_VERT = '''
    #version 330
    in vec3 in_pos; in vec3 in_scale; in vec4 in_color; in vec4 in_quat;
    out vec3 v_scale; out vec4 v_color; out vec4 v_quat;
    void main() {
        v_scale = in_scale; v_quat = in_quat; v_color = in_color;
        gl_Position = vec4(in_pos, 1.0);
    }
'''

_SPLAT_GEOM = '''
    #version 330
    layout(points) in;
    layout(triangle_strip, max_vertices=4) out;
    uniform mat4 view, proj;
    uniform float fade_inner_x, fade_inner_y, fade_outer_x, fade_outer_y;
    uniform float fade_max_depth;
    in  vec4  v_color[]; in  vec3  v_scale[]; in  vec4  v_quat[];
    out vec4  g_color; out vec2  g_uv; out float g_view_depth;

    vec3 rot_quat(vec3 v, vec4 q) {
        vec3 u = q.xyz; float w = q.w;
        return 2.0*dot(u,v)*u + (w*w - dot(u,u))*v + 2.0*w*cross(u,v);
    }
    vec2 proj_delta(vec3 vpos, vec3 offset, vec4 clip_c) {
        vec4 c2 = proj * vec4(vpos + offset, 1.0);
        return c2.xy / c2.w - clip_c.xy / clip_c.w;
    }

    void main() {
        vec4 cam_pos = view * gl_in[0].gl_Position;
        if (cam_pos.z >= -0.05) return;
        vec4 clip_c  = proj * cam_pos;
        if (clip_c.w <= 0.0) return;

        vec2 ndc = clip_c.xy / clip_c.w;
        vec2 e_outer = vec2(ndc.x / max(fade_outer_x, 0.001),
                            ndc.y / max(fade_outer_y, 0.001));
        float r_outer = length(e_outer);
        if (r_outer >= 1.0) return;

        float inner_frac = 0.5 * (fade_inner_x / max(fade_outer_x, 0.001)
                                + fade_inner_y / max(fade_outer_y, 0.001));
        float t = clamp((r_outer - inner_frac) / max(1.0 - inner_frac, 0.001), 0.0, 1.0);
        float ang_fade = 1.0 - t * t * (3.0 - 2.0 * t);

        float view_depth  = -cam_pos.z;
        float depth_start = fade_max_depth * 0.7;
        float td = clamp((view_depth - depth_start) / max(fade_max_depth - depth_start, 0.001), 0.0, 1.0);
        float dep_fade = 1.0 - td * td * (3.0 - 2.0 * td);
        float fade = ang_fade * dep_fade;

        vec4 q  = v_quat[0];
        vec3 sc = v_scale[0];
        vec3 v0 = (view * vec4(rot_quat(vec3(1,0,0), q) * sc.x, 0.0)).xyz;
        vec3 v1 = (view * vec4(rot_quat(vec3(0,1,0), q) * sc.y, 0.0)).xyz;
        vec3 v2 = (view * vec4(rot_quat(vec3(0,0,1), q) * sc.z, 0.0)).xyz;

        vec3 vp = cam_pos.xyz;
        vec2 d0 = proj_delta(vp, v0, clip_c);
        vec2 d1 = proj_delta(vp, v1, clip_c);
        vec2 d2 = proj_delta(vp, v2, clip_c);

        float cxx = d0.x*d0.x + d1.x*d1.x + d2.x*d2.x;
        float cxy = d0.x*d0.y + d1.x*d1.y + d2.x*d2.y;
        float cyy = d0.y*d0.y + d1.y*d1.y + d2.y*d2.y;
        float T = cxx + cyy;
        float D = cxx*cyy - cxy*cxy;
        float S = sqrt(max(0.0, T*T*0.25 - D));
        float l1 = min(T*0.5+S, 16.0/9.0);
        float l2 = min(T*0.5-S, 16.0/9.0);
        if (l1 < 1e-6) return;

        vec2 e1  = normalize(vec2(cxy, l1 - cxx + 1e-6));
        vec2 e2  = vec2(-e1.y, e1.x);
        vec2 ax1 = 3.0*sqrt(max(0.0,l1))*e1;
        vec2 ax2 = 3.0*sqrt(max(0.0,l2))*e2;

        vec2 corners[4] = vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
        for (int i = 0; i < 4; i++){
            vec2 off = corners[i].x*ax1 + corners[i].y*ax2;
            gl_Position = clip_c;
            gl_Position.xy += off * clip_c.w;
            g_color = v_color[0];
            g_color.a *= fade;
            g_uv = corners[i] * 3.0;
            g_view_depth = -cam_pos.z;
            EmitVertex();
        }
        EndPrimitive();
    }
'''

_SPLAT_FRAG = '''
    #version 330
    in  vec4  g_color; in  vec2  g_uv; in  float g_view_depth;
    out vec4  f_color;
    uniform int       debug_uv;
    uniform int       premult_output;
    uniform sampler2D mesh_depth_tex;
    uniform vec2      screen_size;
    uniform float     depth_margin, near, far;
    float linearize(float d){
        return 2.0*near*far/(far+near-(2.0*d-1.0)*(far-near));
    }
    void main(){
        if (debug_uv==1){
            f_color=vec4(g_uv.x*0.33+0.5,g_uv.y*0.33+0.5,0.0,1.0); return;
        }
        float r2=dot(g_uv,g_uv);
        if(r2>9.0) discard;
        float ga=g_color.a*exp(-0.5*r2);
        if(ga<0.001) discard;
        vec2  uv2=gl_FragCoord.xy/screen_size;
        float md=texture(mesh_depth_tex,uv2).r;
        if(md<1.0){
            float ml=linearize(md);
            float bias=depth_margin*0.1;
            ga*=(1.0-smoothstep(-bias,depth_margin-bias,g_view_depth-ml));
        }
        if(ga<0.001) discard;
        if (premult_output == 1)
            f_color = vec4(g_color.rgb * ga, ga);
        else
            f_color = vec4(g_color.rgb, ga);
    }
'''

# circle of confusion radius for each pixel
_COC_FRAG = '''
    #version 330
    uniform sampler2D depth_tex;
    uniform float focal_distance, focal_range, max_coc, near, far;
    in vec2 v_uv; out float f_coc;
    float lin(float d){ return 2.0*near*far/(far+near-(2.0*d-1.0)*(far-near)); }
    void main(){
        float l = lin(texture(depth_tex, v_uv).r);
        float r = l - focal_distance;
        f_coc = (abs(r) < focal_range) ? 0.0
              : sign(r) * min((abs(r)-focal_range)*(max_coc/max(focal_distance,0.1)), max_coc);
    }
'''

_KAWASE_FRAG = '''
    #version 330
    uniform sampler2D color_tex;
    uniform vec2  texel;
    uniform float kawase_offset;
    in vec2 v_uv; out vec4 f_color;
    void main(){
        float o = kawase_offset;
        vec4 acc = texture(color_tex, v_uv + vec2( o,  o) * texel);
             acc += texture(color_tex, v_uv + vec2(-o,  o) * texel);
             acc += texture(color_tex, v_uv + vec2( o, -o) * texel);
             acc += texture(color_tex, v_uv + vec2(-o, -o) * texel);
        f_color = acc * 0.25;
    }
'''

# uses CoC to blend mesh with blurred version
_DOF_COMPOSITE_FRAG = '''
    #version 330
    uniform sampler2D sharp_tex, blur_tex, coc_tex;
    uniform float max_coc;
    in vec2 v_uv; out vec4 f_color;
    void main(){
        float coc = abs(texture(coc_tex, v_uv).r);
        float t = clamp(coc / max(max_coc, 0.5), 0.0, 1.0);
        vec3 sharp = texture(sharp_tex, v_uv).rgb;
        vec3 blurred = texture(blur_tex,  v_uv).rgb;
        f_color = vec4(mix(sharp, blurred, smoothstep(0.0, 1.0, t)), 1.0);
    }
'''

_FREQ_COMPOSITE_FRAG = '''
    #version 330
    uniform sampler2D mesh_tex;
    uniform sampler2D splat_tex;
    uniform sampler2D splat_lf1_tex, splat_lf2_tex;
    uniform int freq_aware;
    uniform int debug_freq;
    uniform float freq_bleed;
    in vec2 v_uv; out vec4 f_color;

    void main(){
        vec3 mesh = texture(mesh_tex, v_uv).rgb;
        vec4 sp = texture(splat_tex, v_uv);
        float alpha = clamp(sp.a, 0.0, 1.0);
        vec3 sp_pm  = clamp(sp.rgb, vec3(0.0), vec3(alpha));

        if (debug_freq == 1){
            vec3 s_lf1 = texture(splat_lf1_tex, v_uv).rgb;
            f_color = vec4((sp_pm - s_lf1) * 2.0 + 0.5, 1.0); return;
        }

        if (freq_aware == 0){
            f_color = vec4(clamp(sp_pm + mesh * (1.0 - alpha), 0.0, 1.0), 1.0);
            return;
        }

        vec4 lf1_s = texture(splat_lf1_tex, v_uv);
        vec4 lf2_s = texture(splat_lf2_tex, v_uv);

        float w_hf = alpha;
        float w_mf = clamp(lf1_s.a * freq_bleed * 2.0, 0.0, 1.0);
        float w_lf = clamp(lf2_s.a * freq_bleed * 4.0, 0.0, 1.0);

        vec3 result = sp_pm + mesh * (1.0 - w_hf);
        float bleed_lf = clamp(w_lf - w_hf, 0.0, 1.0);
        float bleed_mf = clamp(w_mf - w_hf, 0.0, 1.0);
        result = mix(result, mesh, bleed_lf * 0.85);
        result = mix(result, mesh, bleed_mf * 0.6);

        f_color = vec4(clamp(result, 0.0, 1.0), 1.0);
    }
'''

# copy one FBO's colour attachment to another
_BLIT_FRAG = '''
    #version 330
    uniform sampler2D src; in vec2 v_uv; out vec4 f_color;
    void main(){ f_color=texture(src,v_uv); }
'''

_ZONE_VIS_FRAG = '''
    #version 330
    uniform float inner_x, inner_y, outer_x, outer_y;
    uniform vec2  center_ndc;
    in vec2 v_uv; out vec4 f_color;
    void main(){
        vec2 ndc  = v_uv * 2.0 - 1.0;
        vec2 d    = ndc - center_ndc;
        vec2 e_outer = vec2(d.x / max(outer_x, 0.0001), d.y / max(outer_y, 0.0001));
        vec2 e_inner = vec2(d.x / max(inner_x, 0.0001), d.y / max(inner_y, 0.0001));
        float r_outer = length(e_outer);
        float r_inner = length(e_inner);
        float ring_w  = 0.03;
        float a_inner = smoothstep(ring_w, 0.0, abs(r_inner - 1.0));
        float a_outer = smoothstep(ring_w, 0.0, abs(r_outer - 1.0));
        float in_zone = step(1.0, r_inner) * step(r_outer, 1.0);
        if (r_outer > 1.0 + ring_w) discard;
        vec4 col = vec4(0.0);
        col += vec4(1.0, 0.85, 0.0, 0.80) * a_inner;
        col += vec4(0.0, 0.85, 1.0, 0.80) * a_outer;
        col += vec4(0.4, 0.75, 1.0, 0.06) * in_zone;
        f_color = col;
    }
'''

def build_shaders(ctx):
    s = SimpleNamespace()

    s.mesh_prog = ctx.program(vertex_shader=_MESH_VERT,    fragment_shader=_MESH_FRAG)
    s.mesh_uv_prog = ctx.program(vertex_shader=_MESH_UV_VERT, fragment_shader=_MESH_UV_FRAG)
    s.splat_prog = ctx.program(vertex_shader=_SPLAT_VERT,   geometry_shader=_SPLAT_GEOM, fragment_shader=_SPLAT_FRAG)
    s.coc_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_COC_FRAG)
    s.kawase_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_KAWASE_FRAG)
    s.dof_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_DOF_COMPOSITE_FRAG)
    s.freq_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_FREQ_COMPOSITE_FRAG)
    s.blit_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_BLIT_FRAG)
    s.zone_vis_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_ZONE_VIS_FRAG)

    # quad geometry shared by all full-screen passes
    qv = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
    qi = np.array([0,1,2,2,1,3], dtype='i4')
    s.quad_vbo = ctx.buffer(qv.tobytes())
    s.quad_ibo = ctx.buffer(qi.tobytes())

    def _quad_vao(prog):
        return ctx.vertex_array(prog, [(s.quad_vbo, '2f 2f', 'in_pos', 'in_uv')], s.quad_ibo)

    s.coc_vao = _quad_vao(s.coc_prog)
    s.kawase_vao = _quad_vao(s.kawase_prog)
    s.dof_vao = _quad_vao(s.dof_prog)
    s.freq_vao = _quad_vao(s.freq_prog)
    s.blit_vao = _quad_vao(s.blit_prog)
    s.zone_vis_vao = _quad_vao(s.zone_vis_prog)

    return s

def make_mesh_vao(ctx, sh, scene):
    return ctx.vertex_array(
        sh.mesh_prog,
        [(scene.mesh_vbo, '3f 3f 3f', 'in_pos', 'in_normal', 'in_color')],
        scene.mesh_ibo)

def make_mesh_vao_uv(ctx, sh, scene):
    return ctx.vertex_array(
        sh.mesh_uv_prog,
        [(scene.mesh_uv_vbo, '3f 3f 2f', 'in_pos', 'in_normal', 'in_uv')])

def make_splat_vao(ctx, sh, vbo):
    return ctx.vertex_array(
        sh.splat_prog,
        [(vbo, '3f 3f 4f 4f', 'in_pos', 'in_scale', 'in_color', 'in_quat')])