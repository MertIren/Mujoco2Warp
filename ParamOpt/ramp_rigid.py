import numpy as np

import warp as wp
import warp.optim
import warp.sim
import warp.sim.render

from pxr import Usd, UsdGeom, Gf
import math

def quad_mesh_to_triangles(face_idxs, vertex_count):

    tri_idxs = []
    
    offset = 0
    for count in vertex_count:
        assert(count == 3 or count == 4)
        if count == 3:
            v0 = face_idxs[offset]
            v1 = face_idxs[offset + 1]
            v2 = face_idxs[offset + 2]

            offset += 3
            tri_idxs.extend([v0, v1, v2])


        elif count == 4:
            v0 = face_idxs[offset]
            v1 = face_idxs[offset + 1]
            v2 = face_idxs[offset + 2]
            v3 = face_idxs[offset + 3]

            offset += 4
            tri_idxs.extend([v0, v1, v2])
            tri_idxs.extend([v0, v2, v3])
            
            tri_idxs.extend([v1, v2, v3])
            tri_idxs.extend([v0, v1, v3])

    assert (offset == face_idxs.shape[0])

    return np.array(tri_idxs)


@wp.kernel
def assign_param(params: wp.array(dtype=wp.float32), tet_materials: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    tet_materials[tid] = params[tid]


@wp.kernel
def com_kernel(particle_q: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    point = particle_q[tid]
    a = point / wp.float32(particle_q.shape[0])

    # Atomically add the point coordinates to the accumulator
    wp.atomic_add(com, 0, a)


@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.transformf), target: wp.vec3, loss: wp.array(dtype=float)):
    tid = wp.tid()
    delta = wp.transform_get_translation(pos[tid]) - target
    loss[tid] = wp.dot(delta, delta)

@wp.kernel
def enforce_constraint_kernel(lower_bound: wp.float32, upper_bound: wp.float32, x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if x[tid] < lower_bound:
        x[tid] = lower_bound
    elif x[tid] > upper_bound:
        x[tid] = upper_bound


class Example:
    def __init__(
        self,
        stage_path="example_softbody_properties.usd",
        material_behavior="anisotropic",
        verbose=False,
    ):
        self.verbose = verbose
        self.material_behavior = material_behavior

        # seconds
        sim_duration = 2.5

        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 10
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        # self.train_rate = 1e7
        self.train_rate = 1e3

        self.losses = []

        self.hard_lower_bound = wp.float32(-10.0)
        self.hard_upper_bound = wp.float32(4e6)
        # self.hard_upper_bound = wp.float32(2e5)


        # Create FEM model.
        # self.cell_dim = 2
        # self.cell_size = 0.1
        # center = self.cell_size * self.cell_dim * 0.5
        # self.grid_origin = wp.vec3(-0.5, 1.0, -center)
        self.create_model()

        # self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = wp.vec3(6.0, 0.5, 0.0)
        # self.target = wp.vec3(-20.0, 30.0, 0.0)

        # Initialize material parameters
        # if self.material_behavior == "anisotropic":
        #     # Different Lame parameters for each tet
        #     self.material_params = wp.array(
        #         self.model.tet_materials.numpy()[:, :2].flatten(),
        #         dtype=wp.float32,
        #         requires_grad=True,
        #     )
        # else:
        #     # Same Lame parameters for all tets
        #     self.material_params = wp.array(
        #         self.model.tet_materials.numpy()[0, :2].flatten(),
        #         dtype=wp.float32,
        #         requires_grad=True,
        #     )

        print(f"Shape is: {self.model.shape_materials.kf.shape}")
        print(self.model.shape_materials.kf)
        self.material_params = wp.array(
                self.model.shape_materials.kf,
                dtype=wp.float32,
                requires_grad=True,
        )


        self.optimizer = wp.optim.SGD(
            [self.material_params],
            lr=self.train_rate,
            nesterov=False,
        )

        # self.com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True)
        # self.pos_error = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0])


        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None

        # capture forward/backward passes
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def create_model(self):
        builder = wp.sim.ModelBuilder()


        asset_stage = Usd.Stage.Open("/home/miren/Documents/ParamOpt/assets/bear.usd")

        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bear"))
        points = geom.GetPointsAttr().Get()

        xform = Gf.Matrix4f(geom.ComputeLocalToWorldTransform(0.0))
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        self.points = [wp.vec3(point) for point in points]
        self.tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        ke = 1.0e5
        kf = 500.0
        kd = 250.0
        # mu = 0.2

        # b = builder.add_body(origin=wp.transform((0, 5, 0), wp.quat_identity()))
        # builder.add_shape_box(
        #     body=b,
        #     pos=wp.vec3(0),
        #     hx = 0.3, hy = 0.3, hz = 0.3,
        #     # rot=wp.quat_rpy(0.0, 0.0, math.pi*-0.6),
        #     rot=wp.quat_identity(),
        #     ke=ke,
        #     kf=kf,
        #     kd=kd,
        #     # mu=mu
        # )

        


        self.scale = 0.8
        self.ke = 1.0e5
        self.kd = 250.0
        self.kf = 500.0

        # boxes
        # for i in range(self.num_bodies):
        b = builder.add_body(origin=wp.transform((0.0, 3.0, 0.0), wp.quat_rpy(0.0, 0.0, math.pi*0.9)))

        builder.add_shape_box(
            # body=b,
            body=b,
            pos=wp.vec3(0.0, 0.0, 0.0),
            hx=0.5 * self.scale,
            hy=0.2 * self.scale,
            hz=0.2 * self.scale,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
        )
        builder.add_shape_box(
            body=-1,
            pos=wp.vec3(0.0, 2.0, 0.0),
            hx=0.5,
            hy=15.0,
            hz=2.0,
            rot=wp.quat_rpy(0.0, 0.0, math.pi*-0.6),
            ke=self.ke,
            kf=self.kf,
            kd=self.kd,
            # mu=mu,
        )

        # initial spin
        # for i in range(len(builder.body_qd)):
        #     builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)



        # use `requires_grad=True` to create a model for differentiable simulation
        # self.builder = builder
        self.model = builder.finalize(requires_grad=True)
        self.model.shape_materials.kf.requires_grad = True
        self.model.ground = True



        # self.model.soft_contact_ke = ke
        # self.model.soft_contact_kf = kf
        # self.model.soft_contact_kd = kd
        # self.model.soft_contact_mu = mu
        # self.model.soft_contact_margin = 1.0
        # self.model.soft_contact_restitution = 1.0

    def forward(self):
        wp.launch(
            kernel=assign_param,
            dim=self.model.shape_count,
            inputs=(self.material_params,),
            outputs=(self.model.shape_materials.kf,),
        )
        # run control loop
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()

            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # Update loss
        # Compute the center of mass for the last time step.
        # print(f"Particle count: {self.model.particle_count}")
        # wp.launch(
        #     kernel=com_kernel,
        #     dim=self.model.particle_count,
        #     inputs=(self.states[-1].particle_q,),
        #     outputs=(self.com,),
        # )

        # calculate loss
        # print()
        # wp.launch(
        #     kernel=loss_kernel,
        #     dim=1,
        #     inputs=(
        #         self.target,
        #         self.com,
        #     ),
        #     outputs=(self.pos_error, self.loss),
        # )

        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].body_q, self.target, self.loss])
        return self.loss

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)

            if self.verbose:
                self.log_step()

            self.optimizer.step([self.material_params.grad])

            wp.launch(
                kernel=enforce_constraint_kernel,
                dim=self.material_params.shape[0],
                inputs=(
                    self.hard_lower_bound,
                    self.hard_upper_bound,
                ),
                outputs=(self.material_params,),
            )

            self.losses.append(self.loss.numpy()[0])

            # clear grads for next iteration
            self.tape.zero()
            self.loss.zero_()
            # self.com.zero_()
            # self.pos_error.zero_()

            self.iter = self.iter + 1

    def log_step(self):
        x = self.material_params.numpy()
        x_grad = self.material_params.grad.numpy()

        print(f"Iter: {self.iter} Loss: {self.loss.numpy()[0]}")



        print(f"KF Values: {x}")
        print(f"KF Gradients: {x_grad}")
        # print(
        #     f"Max Mu: {np.max(x[:, 0])}, Min Mu: {np.min(x[:, 0])}, "
        #     f"Max Lambda: {np.max(x[:, 1])}, Min Lambda: {np.min(x[:, 1])}"
        # )

        # print(
        #     f"Max Mu Grad: {np.max(x_grad[:, 0])}, Min Mu Grad: {np.min(x_grad[:, 0])}, "
        #     f"Max Lambda Grad: {np.max(x_grad[:, 1])}, Min Lambda Grad: {np.min(x_grad[:, 1])}"
        # )

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            # draw trajectory
            # traj_verts = [np.mean(self.states[0].particle_q.numpy(), axis=0).tolist()]
            for i in range(0, self.sim_steps, self.sim_substeps):
                # traj_verts.append(np.mean(self.states[i].particle_q.numpy(), axis=0).tolist())

                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i])
                self.renderer.render_box(
                    pos=self.target,
                    rot=wp.quat_identity(),
                    extents=(0.2, 0.2, 0.2),
                    name="target",
                    color=(0.0, 0.0, 0.0),
                )
                # self.renderer.render_line_strip(
                #     vertices=traj_verts,
                #     color=wp.render.bourke_color_map(0.0, self.losses[0], self.losses[-1]),
                #     radius=0.02,
                #     name=f"traj_{self.iter - 1}",
                # )
                self.renderer.end_frame()

                from pxr import Gf, UsdGeom

                # particles_prim = self.renderer.stage.GetPrimAtPath("/root/particles")
                # particles = UsdGeom.Points.Get(self.renderer.stage, particles_prim.GetPath())
                # particles.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)], time=self.renderer.time)

                self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_softbody_properties.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=300,
        help="Total number of training iterations.",
    )
    parser.add_argument(
        "--material_behavior",
        default="anisotropic",
        choices=["anisotropic", "isotropic"],
        help="Set material behavior to be Anisotropic or Isotropic.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out additional status messages during execution.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            material_behavior=args.material_behavior,
            verbose=args.verbose,
        )

        # replay and optimize
        for i in range(args.train_iters):
            example.step()
            if i == 0 or i % 5 == 0 or i == args.train_iters - 1:
                example.render()

        if example.renderer:
            example.renderer.save()
