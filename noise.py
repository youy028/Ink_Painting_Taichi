import taichi as ti
import numpy as np
import taichi_glsl as ts

ti.init(arch=ti.gpu)

nx = 256
ny = 256
blk_txt = ti.field(dtype=ti.f32, shape=(nx, ny))
pic = ti.field(dtype=ti.f32, shape=(nx, ny))


@ti.func
def smooth_lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + ts.scalar.smoothstep(frac) * (vr - vl)


@ti.func
def noise(i, j):
    return ts.fract(ti.sin(i * 127.1 + j * 311.7) * 4378.245233)


@ti.func
def interp_noise(i: ti.f32, j: ti.f32):
    fract_p_i = ts.fract(i)
    fract_p_j = ts.fract(j)
    int_p_i = ti.floor(i)
    int_p_j = ti.floor(j)

    v1 = noise(int_p_i, int_p_j)
    v2 = noise(int_p_i + 1, int_p_j)
    v3 = noise(int_p_i, int_p_j + 1)
    v4 = noise(int_p_i + 1, int_p_j + 1)

    return smooth_lerp(smooth_lerp(v1, v2, fract_p_i),
                       smooth_lerp(v3, v4, fract_p_i),
                       fract_p_j)


@ti.func
def fbm(x: ti.f32, y: ti.f32):
    persistence = 0.5
    total = 0.0
    for i in range(10):
        freq = 3 ** i
        amp = persistence ** i
        total += interp_noise(x * freq, y * freq) * amp
    return ts.scalar.clamp(total * 0.5)


@ti.kernel
def init():
    for i, j in ti.ndrange(nx, ny):
        # pic[i, j] = noise(i * 0.05, j * 0.05)
        pic[i, j] = ts.scalar.smoothstep(fbm(ti.cast(i * 0.05, ti.f32), ti.cast(j * 0.05, ti.f32)) ** 3.5)


if __name__ == '__main__':
    init()
    gui = ti.GUI('lbm solver', (nx, ny))
    pic_np = pic.to_numpy()
    while True:
        gui.set_image(pic_np)
        gui.show()
