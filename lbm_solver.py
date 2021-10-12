import taichi as ti
import taichi_glsl as ts
import numpy as np
import noise

ti.init(arch=ti.gpu)


@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)


@ti.data_oriented
class lbm_solver:
    def __init__(self,
                 nx,  # domain size
                 ny,
                 niu,  # viscosity of fluid
                 steps=60000):  # total steps to run
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.relax = 0.5
        self.evap_b = 0.00005  # reduce f_i's that bounce back due to pinning at evap_b during streaming
        self.evap_s = 0.002  # overall evaporation rate
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))  # Fluid Density
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))  # Velocity
        self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_mid = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.blk_txt = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.blk = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.w = ti.field(dtype=ti.f32, shape=9)  # Weight
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))  # 9 Directions
        self.steps = steps

        arr = np.array([4.0 / 9.0,
                        1.0 / 9.0,
                        1.0 / 9.0,
                        1.0 / 9.0,
                        1.0 / 9.0,
                        1.0 / 36.0,
                        1.0 / 36.0,
                        1.0 / 36.0,
                        1.0 / 36.0],
                       dtype=np.float32)
        self.w.from_numpy(arr)
        arr = np.array([[0, 0],  # Center     0
                        [1, 0],  # Down       1
                        [0, 1],  # Right      2
                        [-1, 0],  # Top       3
                        [0, -1],  # Left      4
                        [1, 1],  # DownRight  5
                        [-1, 1],  # TopRight  6
                        [-1, -1],  # TopLeft  7
                        [1, -1]],  # DownLeft 8
                       dtype=np.int32)
        self.e.from_numpy(arr)

    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + \
             ti.cast(self.e[k, 1], ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0
        advect = ts.scalar.smoothstep(self.rho[i, j], 0.0, 0.25)
        return self.w[k] * (self.rho[i, j] + advect * (3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv))

    @ti.kernel
    def init(self):
        for i, j in self.blk_txt:
            if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                self.blk_txt[i, j] = 100.0
            else:
                # self.blk_txt[i, j] = noise.fbm(ti.cast(i / 25.0, ti.f32), ti.cast(j / 25.0, ti.f32)) ** 3
                self.blk_txt[i, j] = ts.scalar.smoothstep(noise.fbm(ti.cast(i * 0.05, ti.f32), ti.cast(j * 0.05, ti.f32)) ** 3.5)

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            a = i - (self.nx / 2)
            b = j - (self.ny / 2)
            if (a ** 2) + (b ** 2) < 60 ** 2:
                self.rho[i, j] = 1.0
            else:
                self.rho[i, j] = 0.0

            self.mask[i, j] = 0.0
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_mid[i, j][k] = self.f_new[i, j][k]
                self.f_old[i, j][k] = self.f_new[i, j][k]

                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.blk[i, j][k] = ts.scalar.clamp((self.blk_txt[ip, jp] + self.blk_txt[i, j]) / 2.0)

    @ti.kernel
    def collide(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_mid[i, j][k] = lerp(self.f_old[ip, jp][k], self.f_eq(ip, jp, k), 0.5)

    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.f_new[i, j][0] = self.f_mid[i, j][0]  # Center

            self.f_new[i, j][1] = lerp(self.f_mid[i - 1, j][1], self.f_mid[i, j][3], self.blk[i, j][1])  # Down
            self.f_new[i, j][3] = lerp(self.f_mid[i + 1, j][3], self.f_mid[i, j][1], self.blk[i, j][3])  # Top
            self.f_new[i, j][2] = lerp(self.f_mid[i, j - 1][2], self.f_mid[i, j][4], self.blk[i, j][2])  # Right
            self.f_new[i, j][4] = lerp(self.f_mid[i, j + 1][4], self.f_mid[i, j][2], self.blk[i, j][4])  # Left

            self.f_new[i, j][5] = lerp(self.f_mid[i - 1, j - 1][5], self.f_mid[i, j][7], self.blk[i, j][5])  # DownRight
            self.f_new[i, j][7] = lerp(self.f_mid[i + 1, j + 1][7], self.f_mid[i, j][5], self.blk[i, j][7])  # TopLeft
            self.f_new[i, j][6] = lerp(self.f_mid[i + 1, j - 1][6], self.f_mid[i, j][8], self.blk[i, j][6])  # TopRight
            self.f_new[i, j][8] = lerp(self.f_mid[i - 1, j + 1][8], self.f_mid[i, j][6], self.blk[i, j][8])  # DownLeft

            self.f_new[i, j] = max(self.f_new[i, j] - (self.blk[i, j] > 0.98) * self.evap_b, 0)

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += (ti.cast(self.e[k, 0], ti.f32) * self.f_new[i, j][k])
                self.vel[i, j][1] += (ti.cast(self.e[k, 1], ti.f32) * self.f_new[i, j][k])
            self.rho[i, j] = max(0, self.rho[i, j] - self.evap_s)

    @ti.kernel
    def sum_rho(self) -> ti.f32:
        rho_sum = 0.0
        for i, j in self.rho:
            rho_sum += self.rho[i, j]
        return rho_sum

    def solve(self):
        gui = ti.GUI('lbm solver', (self.nx, self.ny))
        self.init()
        for i in range(self.steps):
            self.collide()
            self.stream()
            self.update_macro_var()
            rho_img = self.rho.to_numpy()
            gui.set_image(rho_img)
            gui.show()
            if (i % 1000 == 0):
                print('Step: {:}'.format(i))
                # ti.imwrite((img[:,:,0:3]*255).astype(np.uint8), 'fig/karman_'+str(i).zfill(6)+'.png')
                print((self.sum_rho()))

    def pass_to_py(self):
        return self.vel.to_numpy()[:, :, 0]


if __name__ == '__main__':
    flow_case = 1
    lbm = lbm_solver(512, 512, 0.01)
    lbm.solve()
