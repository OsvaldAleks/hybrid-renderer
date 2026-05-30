import numpy as np
import glfw
from pyrr import Matrix44

from config import NEAR, FAR, FOV_Y, CAM_RADIUS, CAM_THETA, CAM_PHI

radius    = CAM_RADIUS
theta     = CAM_THETA
phi       = CAM_PHI
last_mouse = None

model_yaw   = 0.0
model_pitch = 0.0

_io = None

def init(io):
    global _io
    _io = io

def mouse_move_cb(win, x, y):
    global last_mouse, theta, phi, model_yaw, model_pitch
    if last_mouse is None:
        last_mouse = (x, y); return
    if _io and _io.want_capture_mouse:
        last_mouse = (x, y); return
    dx, dy = x - last_mouse[0], y - last_mouse[1]
    last_mouse = (x, y)
    if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
        theta += dx * 0.005
        phi    = np.clip(phi - dy * 0.005, -np.pi/2+0.01, np.pi/2-0.01)
    if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
        model_yaw   += dx * 0.5
        model_pitch  = np.clip(model_pitch - dy * 0.5, -89.0, 89.0)

def scroll_cb(win, xoff, yoff):
    global radius
    if _io and _io.want_capture_mouse:
        return
    radius = max(0.1, radius * (1 - yoff * 0.1))

def get_view_proj(w, h):
    aspect = w / h
    cam = np.array([np.cos(phi)*np.sin(theta)*radius,
                    np.sin(phi)*radius,
                    np.cos(phi)*np.cos(theta)*radius], dtype=np.float32)
    view = Matrix44.look_at(cam, (0, 0, 0), (0, -1, 0))
    proj = Matrix44.perspective_projection(FOV_Y, aspect, NEAR, FAR)
    return proj, view, cam

def build_model_mat():
    ry = Matrix44.from_y_rotation(np.deg2rad(model_yaw))
    rx = Matrix44.from_x_rotation(np.deg2rad(model_pitch))
    return (rx * ry).astype('f4')