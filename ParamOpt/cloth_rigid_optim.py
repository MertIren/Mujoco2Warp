# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Cloth
#
# Shows how to use Warp to optimize the initial velocities of a piece of
# cloth such that its center of mass hits a target after a specified time.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import math

import warp as wp
import warp.sim
import warp.optim
import warp.sim.render

from pxr import Usd, UsdGeom, Gf
import numpy as np

@wp.kernel
def assign_param(params: wp.array(dtype=wp.vec3), model_params: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    model_params[tid] = params[0]

@wp.kernel
def com_kernel(positions: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    # compute center of mass
    wp.atomic_add(com, 0, positions[tid] / wp.float32(positions.shape[0]))


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), target: wp.vec3, pos_error: wp.array(dtype=float), loss: wp.array(dtype=float)):
    # sq. distance to target
    diff = com[0] - target
    pos_error[0] = wp.dot(diff, diff)
    norm = pos_error[0]
    loss[0] = norm

@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, stage_path="example_cloth_throw.usd", verbose=False):
        self.verbose = verbose

        # seconds
        sim_duration = 2.0

        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 32
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_rate = 0.01

        self.create_model()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # print(f"Shape 1: {self.velocities.numpy().shape}, Shape 2: {self.model.particle_qd.numpy().shape}")

        

        self.target = (8.0, 0.0, 0.0)
        self.com = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
        self.pos_error = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        
        # self.velocities = wp.zeros(1, dtype=wp.vec3, requires_grad=True)

        self.velocities = wp.array(
            self.states[0].particle_qd.numpy()[0],
            dtype=wp.vec3,
            requires_grad=True
        )
        self.optimizer = wp.optim.SGD(
            [self.velocities],
            lr=self.train_rate,
            nesterov=False,
        )

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=4.0)
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
        # builder.default_particle_radius = 0.01

        dim_x = 24
        dim_y = 24


        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.1, 0.1, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
            # rot = wp.quat_identity(),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e1,
            # tri_lift=10.0,
            # tri_drag=5.0,
        )

        builder.add_shape_box(
            body=-1,
            pos=wp.vec3(8.0, -2.0, 0.0),
            rot=wp.quat_identity(),
            hx=2.0, hy=1.0, hz=2.0,
            ke=1.0e2,
            kd=1.0e2,
            kf=1.0e1,
            # friction=1.0,
            # restitution=0.5,
        )

        # asset_stage = Usd.Stage.Open("/home/miren/Documents/ParamOpt/assets/bunny.usd")
        # mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))

        

        # points = mesh_geom.GetPointsAttr().Get()
        # xform = Gf.Matrix4f(mesh_geom.ComputeLocalToWorldTransform(0.0))

        # for i in range(len(points)):
        #     points[i] = xform.Transform(points[i])

        # indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        # bunny = wp.sim.Mesh(points, indices)
        # builder.add_shape_mesh( 
        #     body = -1,
        #     mesh = bunny,
        #     rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5),
        #     scale = (2.0, 2.0, 2.0),
        #     pos = wp.vec3(3.0, -2.0, 0.0),
        #     ke = 1.0e2,
        #     kf = 1.0e2,
        #     kd = 1.0e1,
        # )


        self.model = builder.finalize(requires_grad=True)
        self.model.ground = False



    def forward(self):
        # print(self.model.particle_qd.numpy)
        wp.launch(
            kernel=assign_param,
            dim=self.model.particle_count,
            inputs=(self.velocities,),
            outputs=(self.states[0].particle_qd,),
        )

        # run control loop
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()

            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # compute loss on final state
        self.com.zero_()
        wp.launch(
            com_kernel,
            dim=self.model.particle_count,
            inputs=[self.states[-1].particle_q,],
            outputs=(self.com,),
        )
        wp.launch(loss_kernel, dim=1, inputs=[self.com, self.target,], outputs=(self.pos_error, self.loss,),)

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)

            # gradient descent step
            # x = self.states[0].particle_qd

            if self.verbose:
                self.log_step()
        
            self.optimizer.step([self.velocities.grad])


            # wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

            # clear grads for next iteration
            self.tape.zero()
            self.loss.zero_()
            self.com.zero_()
            self.pos_error.zero_()

            self.iter = self.iter + 1

    def log_step(self):
        x = self.velocities.numpy()
        x_grad = self.velocities.grad.numpy()

        print(f"Iter: {self.iter} Loss: {self.loss}")
        # print(f"Velocities shape: {self.states[0].particle_qd.numpy().shape}")
        # print(f"Particle Velocities: {self.states[0].particle_qd.numpy()}")

        print(f"Max velocity: {np.max(x)}, Min velocity: {np.min(x)}")
        print(f"Max grad: {np.max(x_grad)}, Min grad: {np.min(x_grad)}")



    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            # draw trajectory
            traj_verts = [self.states[0].particle_q.numpy().mean(axis=0)]

            for i in range(0, self.sim_steps, self.sim_substeps):
                traj_verts.append(self.states[i].particle_q.numpy().mean(axis=0))

                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i])
                self.renderer.render_box(
                    pos=self.target,
                    rot=wp.quat_identity(),
                    extents=(0.1, 0.1, 0.1),
                    name="target",
                    color=(1.0, 0.0, 0.0),
                )
                self.renderer.render_line_strip(
                    vertices=traj_verts,
                    color=wp.render.bourke_color_map(0.0, 269.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_{self.iter - 1}",
                )
                self.renderer.end_frame()

                self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth_throw.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=64, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        # replay and optimize
        for i in range(args.train_iters):
            example.step()
            if i % 1 == 0 or i == args.train_iters-1:
                example.render()

        if example.renderer:
            example.renderer.save()
