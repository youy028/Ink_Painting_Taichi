import taichi as ti
import numpy as np
import taichi_glsl as ts
import cv2

ti.init(arch=ti.gpu)

@ti.func
def lerp(vl, vr, frac):
    return vl + ts.scalar.clamp(frac) * (vr - vl)

@ti.kernel
def sum(x: ti.template()) -> ti.f32:
    x_sum = 0.0
    for i, j in x:
        x_sum += x[i, j]
    return x_sum

@ti.data_oriented
class lbm_solver:
    def __init__(self,
                 nx, # domain size
                 ny,
                 steps = 60000): # total steps to run
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny

        self.rho_surf = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.rho_old = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.rho_new = ti.field(dtype=ti.f32, shape=(nx, ny))

        self.pigment_surf = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.pigment_old = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.pigment_new = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.fixture = ti.field(dtype=ti.f32, shape=(nx, ny))

        self.xuan = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.lines = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.blk_txt = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.perm = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.pinning_threshold = 0.4

        self.paint_rho = 0.5
        self.paint_pig = 0.5
        self.paint_radius = 20

        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))
        self.steps = steps
        self.w.from_numpy(np.array([4.0 / 9.0,
                                    1.0 / 9.0,
                                    1.0 / 9.0,
                                    1.0 / 9.0,
                                    1.0 / 9.0,
                                    1.0 / 36.0,
                                    1.0 / 36.0,
                                    1.0 / 36.0,
                                    1.0 / 36.0],
                                   dtype=np.float32))
        self.e.from_numpy(np.array([[0, 0],  # Center     0
                                    [1, 0],  # Down       1
                                    [0, 1],  # Right      2
                                    [-1, 0],  # Top       3
                                    [0, -1],  # Left      4
                                    [1, 1],  # DownRight  5
                                    [-1, 1],  # TopRight  6
                                    [-1, -1],  # TopLeft  7
                                    [1, -1]],  # DownLeft 8
                                   dtype=np.int32))
        xuan_image = cv2.imread('xuan_paper.png', 0)
        xuan_scaled = cv2.resize(xuan_image, (self.nx, self.ny))
        xuan_arr = (np.asarray(xuan_scaled) / 255.0).astype('float32')
        self.xuan.from_numpy(xuan_arr)

        lines_image = cv2.imread('lines.png', 0)
        lines_scaled = cv2.resize(lines_image, (self.nx, self.ny))
        lines_arr = (np.asarray(lines_scaled) / 255.0).astype('float32')
        self.lines.from_numpy(lines_arr)

        self.evap_b = 0.00005 * 0.1   # reduce f_i's that bounce back due to pinning at evap_b during streaming
        self.evap_s = 0.00015 * 0.1   # overall evaporation rate


    @ti.func # compute equilibrium distribution function
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + ti.cast(self.e[k, 1], ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2.0 + self.vel[i, j][1]**2.0
        advect = ts.scalar.smoothstep(self.rho_new[i, j], 0.0, 0.8)
        return self.w[k] * (self.rho_new[i, j] + advect * (3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv))

    @ti.kernel
    def init(self):
        for i, j in self.rho_new:
            # generate xuan using procedurally
            # self.xuan[i, j] = noise.fbm(ti.cas t(i * 0.2, ti.f32), ti.cast(j * 0.2, ti.f32)) ** 2
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.fixture[i, j] = 0.0

            self.rho_surf[i, j] = 0.0
            self.rho_old[i, j] = 0.0
            self.rho_new[i, j] = 0.0

            self.pigment_surf[i, j] = 0.0
            self.pigment_old[i, j] = 0.0
            self.pigment_new[i, j] = 0.0

            # a = i - (self.nx / 2)
            # b = j - (self.ny / 2)
            # if (a ** 2) + (b ** 2) < 100 ** 2 and i > (self.nx / 2):
            #     self.rho_surf[i, j] = 0.1
            # if (a ** 2) + (b ** 2) < 50 ** 2:
            #     self.rho_surf[i, j] = 0.5
            #     self.pigment_surf[i, j] = 0.5

        for i, j in self.rho_new:
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]


    @ti.kernel
    def update_block(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            pinning = self.rho_new[i, j] == 0.0

            # didn't implement glue so I used self.rho_new[i, j] as glue
            pinning_threshold = 0.05 + 0.2 * self.fixture[i, j] + \
                0.3 * lerp(self.xuan[i, j], self.lines[i, j], ts.scalar.smoothstep(0, 0.5, 0.5)) #self.rho_new[i, j]

            for k in ti.static(range(1, 5)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                pinning = pinning and self.f_old[ip, jp][k] < pinning_threshold
            for k in ti.static(range(5, 9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                pinning = pinning and self.f_old[ip, jp][k] < pinning_threshold * 1.414213562 # sqrt(2) = 1.414213562
            if pinning:
                self.blk_txt[i, j] = 100.0
            else:
                self.blk_txt[i, j] = self.xuan[i, j] + 0.3 * self.fixture[i, j]
                # self.blk_txt[i, j] = 0

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.perm[i, j][k] = 1 - ts.scalar.clamp(0.5 * self.blk_txt[ip, jp] + 0.5 * self.blk_txt[i, j])

    @ti.kernel
    def collide_and_stream(self): # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_new[i, j][k] = 0.5 * self.f_old[ip, jp][k] + 0.5 * self.f_eq(ip, jp, k)

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.f_old[i, j] = self.f_new[i, j]

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.f_new[i, j][1] = lerp(self.f_old[i, j][3], self.f_old[i - 1, j][1], self.perm[i, j][1])  # Down
            self.f_new[i, j][2] = lerp(self.f_old[i, j][4], self.f_old[i, j - 1][2], self.perm[i, j][2])  # Right
            self.f_new[i, j][3] = lerp(self.f_old[i, j][1], self.f_old[i + 1, j][3], self.perm[i, j][3])  # Top
            self.f_new[i, j][4] = lerp(self.f_old[i, j][2], self.f_old[i, j + 1][4], self.perm[i, j][4])  # Left

            self.f_new[i, j][5] = lerp(self.f_old[i, j][7], self.f_old[i - 1, j - 1][5], self.perm[i, j][5])  # DownRight
            self.f_new[i, j][6] = lerp(self.f_old[i, j][8], self.f_old[i + 1, j - 1][6], self.perm[i, j][6])  # TopRight
            self.f_new[i, j][7] = lerp(self.f_old[i, j][5], self.f_old[i + 1, j + 1][7], self.perm[i, j][7])  # TopLeft
            self.f_new[i, j][8] = lerp(self.f_old[i, j][6], self.f_old[i - 1, j + 1][8], self.perm[i, j][8])  # DownLeft

            # additional evaporation at pinning edges
            self.f_new[i, j] = max(self.f_new[i, j] - (self.perm[i, j] > 0.999) * self.evap_b, 0)


    @ti.kernel
    def update_macro_var(self): # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho_old[i, j] = self.rho_new[i, j]
            self.rho_new[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho_new[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += (ti.cast(self.e[k, 0], ti.f32) * self.f_new[i, j][k])
                self.vel[i, j][1] += (ti.cast(self.e[k, 1], ti.f32) * self.f_new[i, j][k])
            # global evaporation
            self.rho_new[i, j] = max(0.0, self.rho_new[i, j] - self.evap_s)

    @ti.kernel
    def seep(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            rho_seep_amt = min(0.01, self.rho_surf[i, j])
            self.rho_new[i, j] += rho_seep_amt
            self.rho_surf[i, j] -= rho_seep_amt

            pig_seep_amt = min(0.01, self.pigment_surf[i, j])
            self.pigment_new[i, j] += pig_seep_amt
            self.pigment_surf[i, j] -= pig_seep_amt


    @ti.kernel
    def apply_bc(self): # impose boundary conditions
        # corners
        self.apply_bc_core(0, 0, 1, 1)
        self.apply_bc_core(self.nx - 1, 0, self.nx - 2, 1)
        self.apply_bc_core(0, self.ny - 1, 1, self.ny - 2)
        self.apply_bc_core(self.nx - 1, self.ny - 1, self.nx - 2, self.ny - 2)

        # left and right
        for j in ti.ndrange(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in ti.ndrange(1, self.nx - 1):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(i, 0, i, 1)


    @ti.func
    def apply_bc_core(self, ibc, jbc, inb, jnb):
        self.vel[ibc, jbc][0] = 0
        self.vel[ibc, jbc][1] = 0
        self.rho_new[ibc, jbc] = self.rho_new[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc, jbc][k] = self.f_eq(ibc, jbc, k) - self.f_eq(inb, jnb, k) + self.f_old[inb, jnb][k]


    @ti.kernel
    def deposit_pigment(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            water_loss = self.rho_old[i, j] - self.rho_new[i, j]
            if self.rho_old[i, j] > 0.0 and water_loss > 0.0:
                fix_factor = water_loss / self.rho_old[i, j]
                self.fixture[i, j] += fix_factor * self.pigment_new[i, j]
                self.pigment_new[i, j] -= fix_factor * self.pigment_new[i, j]


    @ti.kernel
    def advect_pigment(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # percent_old = 1.0
            # pig_surround = 0.0
            # self.pigment_old[i, j] = self.pigment_new[i, j]
            # for k in ti.static(range(1, 9)):
            #     percent_old -= self.f_new[i, j][k]
            #     ip = i - self.e[k, 0]
            #     jp = j - self.e[k, 1]
            #     pig_surround += self.pigment_old[ip, jp] * self.f_new[ip, jp][k]
            # self.pigment_new[i, j] = (percent_old * self.pigment_old[i, j] + pig_surround)


            self.pigment_old[i, j] = self.pigment_new[i, j]
            if self.rho_new[i, j] <= 0.00001:
                self.pigment_new[i, j] = 0.0
            elif self.rho_old[i, j] <= 0.00001:
                self.pigment_new[i, j] = 0.0
                for k in ti.static(range(1, 9)):
                    ip = i - self.e[k, 0]
                    jp = j - self.e[k, 1]
                    self.pigment_new[i, j] += self.f_new[i, j][k] * self.pigment_old[ip, jp]
                self.pigment_new[i, j] /= self.rho_new[i, j]
            else:
                ip = i - self.vel[i, j][0]
                jp = j - self.vel[i, j][1]
                self.pigment_new[i, j] = ts.sampling.bilerp(self.pigment_old, ts.vec2(ip, jp))

                # # edge case ?
                # ip_inc = ti.cast(self.vel[i, j][0], ti.i32)
                # jp_inc = ti.cast(self.vel[i, j][1], ti.i32)
                # if self.rho_new[i + ip_inc, j + jp_inc] == 0.0 or \
                #    self.rho_new[i,          j + jp_inc] == 0.0 or \
                #    self.rho_new[i + ip_inc, j] == 0.0:
                #     self.pigment_new[i, j] = self.pigment_old[i, j]
                # else:
                #     self.pigment_new[i, j] = ts.sampling.bilerp(self.pigment_old, ts.vec2(ip, jp))


    @ti.kernel
    def paint(self, pos_x: ti.i32, pos_y: ti.i32, paint_radius: ti.i32, paint_rho: ti.f32, paint_pig: ti.f32):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            a = i - pos_x
            b = j - pos_y
            if (a ** 2) + (b ** 2) < paint_radius ** 2:
                self.rho_surf[i, j] = paint_rho
                self.pigment_surf[i, j] = paint_pig


    def solve(self):
        gui = ti.GUI('lbm solver', (self.nx, self.ny))
        self.init()
        print('water mass', "start", sum(self.rho_new))
        print('ink mass', "start", sum(self.pigment_new))
        for i in range(self.steps):
            gui.get_event()
            if i % 10 == 0:
                if gui.is_pressed(ti.GUI.UP):
                    self.paint_pig = min(1.0, self.paint_pig + 0.1)
                    print("paint_rho: %s, paint_pig %s  " % (self.paint_rho, self.paint_pig))
                if gui.is_pressed(ti.GUI.DOWN):
                    self.paint_pig = max(0.0, self.paint_pig - 0.1)
                    print("paint_rho: %s, paint_pig %s  " % (self.paint_rho, self.paint_pig))
                if gui.is_pressed(ti.GUI.RIGHT):
                    self.paint_rho = min(1.0, self.paint_rho + 0.1)
                    print("paint_rho: %s, paint_pig %s  " % (self.paint_rho, self.paint_pig))
                if gui.is_pressed(ti.GUI.LEFT):
                    self.paint_rho = max(0.0, self.paint_rho - 0.1)
                    print("paint_rho: %s, paint_pig %s  " % (self.paint_rho, self.paint_pig))
                if gui.is_pressed('d'):
                    self.paint_radius = self.paint_radius + 5
                    print("paint_radius: %s" % self.paint_radius)
                if gui.is_pressed('a'):
                    self.paint_radius = max(self.paint_radius - 5, 5)
                    print("paint_radius: %s" % self.paint_radius)

            if gui.is_pressed(' '):
                pos = gui.get_cursor_pos()
                pos_x = int(ts.clamp(pos[0]) * self.nx)
                pos_y = int(ts.clamp(pos[1]) * self.ny)
                self.paint(pos_x, pos_y, self.paint_radius, self.paint_rho, self.paint_pig)

            self.update_block()
            self.collide_and_stream()
            self.update_macro_var()
            self.seep()
            self.apply_bc()

            self.deposit_pigment()
            self.advect_pigment()

            ink_img = self.pigment_new.to_numpy()
            fix_img = self.fixture.to_numpy()
            final_ink = 1.0 - fix_img - ink_img
            gui.set_image(final_ink)
            # gui.set_image(self.rho_new.to_numpy())
            gui.show()


if __name__ == '__main__':
    lbm = lbm_solver(512, 512)
    lbm.solve()