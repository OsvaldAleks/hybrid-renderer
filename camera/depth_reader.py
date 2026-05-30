import numpy as np
import moderngl

class CentreDepthReader:
    def __init__(self):
        self._prog = None
        self._fbo  = None
        self._tex  = None
        self._vao  = None

    def _init(self, ctx, quad_vbo, quad_ibo):
        self._prog = ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
                void main(){ gl_Position=vec4(in_pos,0,1); v_uv=in_uv; }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D depth_tex;
                uniform vec2 centre_uv, texel;
                uniform float near, far;
                in vec2 v_uv; out float f_depth;
                float lin(float d){
                    return 2.0*near*far/(far+near-(2.0*d-1.0)*(far-near));
                }
                void main(){
                    float depths[9]; int k=0;
                    for(int dy=-1;dy<=1;dy++)
                        for(int dx=-1;dx<=1;dx++){
                            float d=texture(depth_tex,
                                centre_uv+vec2(float(dx),float(dy))*texel*4.0).r;
                            depths[k++]=d;
                        }
                    for(int i=0;i<9;i++)
                        for(int j=i+1;j<9;j++)
                            if(depths[j]<depths[i]){
                                float tmp=depths[i];depths[i]=depths[j];depths[j]=tmp;
                            }
                    float med=depths[4];
                    f_depth = (med >= 0.9999) ? -1.0 : lin(med);
                }
            ''')
        self._tex = ctx.texture((1, 1), 1, dtype='f4')
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._fbo = ctx.framebuffer(color_attachments=[self._tex])
        self._vao = ctx.vertex_array(
            self._prog, [(quad_vbo, '2f 2f', 'in_pos', 'in_uv')], quad_ibo)
        self._ctx = ctx

    def read(self, ctx, quad_vbo, quad_ibo, depth_tex, w, h, near, far):
        if self._prog is None:
            self._init(ctx, quad_vbo, quad_ibo)
        self._fbo.use()
        depth_tex.use(location=0)
        self._prog['depth_tex'] = 0
        self._prog['centre_uv'] = (0.5, 0.5)
        self._prog['texel'] = (1.0 / w, 1.0 / h)
        self._prog['near'] = near
        self._prog['far'] = far
        self._vao.render(moderngl.TRIANGLES)
        val = np.frombuffer(self._tex.read(), dtype=np.float32)[0]
        return None if val < 0.0 else float(val)