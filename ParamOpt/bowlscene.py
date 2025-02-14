# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#############################################################################
# Example Ray Cast
#
# Shows how to use the built-in wp.Mesh data structure and wp.mesh_query_ray()
# function to implement a basic ray-tracer.
#
##############################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom, Gf

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import math

def make_cube(size):
    size = 1
    points = []
    indices = []
    idx = 0

    for z in range(1):
        for y in range(1):
            for x in range(size):
                points.append([x, y, z])
                points.append([x + 1, y, z])
                points.append([x, y + 1, z])
                points.append([x, y, z + 1])
                points.append([x + 1, y + 1, z])
                points.append([x + 1, y, z + 1])
                points.append([x, y + 1, z + 1])
                points.append([x + 1, y + 1, z + 1])

                A = idx
                B = idx + 1
                C = idx + 2
                D = idx + 3
                E = idx + 4
                F = idx + 5
                G = idx + 6
                H = idx + 7
                indices.extend([
                    [A, B, C, D],
                    [B, C, F, H],
                    [B, D, E, C],
                    [G, E, C, D],
                    [B, E, H, C],
                    [G, H, E, C]
                ])
    return (np.array(points), np.array(indices))
                



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

    assert (offset == face_idxs.shape[0])

    return np.array(tri_idxs)


class Example:
    def __init__(self, stage_path = "bowl_scene.usd"):
        self.sim_duration = 5
        self.verbose = True

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        frame_steps = int(self.sim_duration / self.frame_dt)

        self.sim_substeps = 120
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0
        self.sim_time = 0


        # asset_stage = Usd.Stage.Open("/home/mert-iren/Documents/Projects/Mujoco2Warp/mujoco2warp/lib/python3.12/site-packages/warp/examples/assets/bunny.usd")
        # asset_stage = Usd.Stage.Open("/home/mert-iren/Documents/Projects/Mujoco2Warp/assets/bunny.usd")
        asset_stage = Usd.Stage.Open("/home/miren/Documents/ParamOpt/assets/bowl/Bowl.geom.usd")
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/Bowl/Geom/Bowl"))
        # mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))
            # print(f"AAAA {mesh_geom}")

        points = mesh_geom.GetPointsAttr().Get()
        xform = Gf.Matrix4f(mesh_geom.ComputeLocalToWorldTransform(0.0))
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])



        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()
        vertex_count = np.array(mesh_geom.GetFaceVertexCountsAttr().Get()).flatten()


        indices = quad_mesh_to_triangles(indices, vertex_count)
        bowl = wp.sim.Mesh(points, indices)


        ke = 1.0e3
        kf = 0.0
        kd = 1.0e0
        mu = 0.2


        builder = wp.sim.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_mesh( 
            body = -1,
            mesh = bowl,
            rot = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi*-0.5),
            scale = (0.5, 0.5, 0.5),
            # pos= wp.vec3(5, -5, 30),
            pos = wp.vec3(0, 10, 0),
            thickness=1e-01,
            ke = ke,
            kf = kf,
            kd = kd,
            mu = mu
        )        
    
        asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bear.usd"))

        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bear"))
        points = geom.GetPointsAttr().Get()

        xform = Gf.Matrix4f(geom.ComputeLocalToWorldTransform(0.0))
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        self.points = [wp.vec3(point) for point in points]
        self.tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        self.cell_dim = 2
        self.cell_size = 0.5
        center = self.cell_size * self.cell_dim * 0.5
        self.grid_origin = wp.vec3(0, 1.0, -center)

        builder.default_particle_radius = 0.0005

        # total_mass = 0.2
        # num_particles = (self.cell_dim + 1) ** 3
        # particle_mass = total_mass / num_particles
        # particle_density = particle_mass / (self.cell_size**3)
        # print(f"NUM PARTICLES ISSS: {num_particles}")

        # young_mod = 1.5 * 1e4
        # poisson_ratio = 0.3
        # k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        # k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))


        # print(f"Density: {particle_density}, k_mu: {k_mu}, k_lambda: {k_lambda}")

        # builder.add_soft_grid(
        #     pos=wp.vec3(3, 25, 4),
        #     rot=wp.quat_identity(),
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     dim_x=self.cell_dim,
        #     dim_y=self.cell_dim,
        #     dim_z=self.cell_dim,
        #     cell_x=self.cell_size,
        #     cell_y=self.cell_size,
        #     cell_z=self.cell_size,
        #     density=particle_density,
        #     k_mu=k_mu,
        #     k_lambda=k_lambda,
        #     k_damp=2.0,
        #     tri_ke=1e-4,
        #     tri_ka=1e-4,
        #     tri_kd=1e-4,
        #     tri_drag=0.0,
        #     tri_lift=0.0,
        #     fix_bottom=False,
        # )

        cell_dim = 20
        cell_size = 2.0 / cell_dim

        center = cell_size * cell_dim * 0.5

        builder.add_soft_grid(
            pos=wp.vec3(-5*center, 20.0, -5*center),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=10.0,
            # fix_bottom=True,
            # fix_top=True,
            k_mu=1000.0,
            k_lambda=5000.0,
            k_damp=0.0,
        )


        # builder.add_soft_mesh(
        #     pos=wp.vec3(5.0, 25, 3.0),
        #     rot=wp.quat_identity(),
        #     scale=0.5,
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     vertices=self.points,
        #     indices=self.tet_indices,
        #     density=1.0,
        #     k_mu=2000.0,
        #     k_lambda=2000.0,
        #     k_damp=2.0,
        #     tri_ke=0.0,
        #     tri_ka=1e-8,
        #     tri_kd=0.0,
        #     tri_drag=0.0,
        #     tri_lift=0.0,
        # )


        # b = builder.add_body()

        # builder.add_shape_box(
        #     body=b,
        #     pos=wp.vec3(0, 4, 0),
        #     hx=0.2, hy=0.2, hz=0.2,
        #     ke = ke,
        #     kf = kf,
        #     kd = kd,
        #     mu = mu
        #     # density=self.density_value,
        #     # kf=0.05
        # )

        # builder.add_shape_box(
        #     body=-1,
        #     pos=wp.vec3(2.0, 1.0, 0.0),
        #     hx=0.25,
        #     hy=1.0,
        #     hz=1.0,
        #     ke=ke,
        #     kf=kf,
        #     kd=kd,
        #     mu=mu,
        # )


        self.model = builder.finalize(requires_grad=True)
        # self.control = self.model.control()


        # radii = wp.zeros(self.model.particle_count, dtype=float)
        # radii.fill_(0.05)
        # self.model.particle_radius = radii


        # BRING BACK
        # self.model.soft_contact_ke = 2.0e3
        # self.model.soft_contact_kd = 0.1
        # self.model.soft_contact_kf = 10.0
        # self.model.soft_contact_mu = 0.7
        self.model.ground = True

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu

        # self.model = builder.finalize()

        # self.model.soft_contact_ke = 2.0e3
        # self.model.soft_contact_kd = 0.1
        # self.model.soft_contact_kf = 10.0
        # self.model.soft_contact_mu = 0.7


        # self.model.soft_contact_ke = ke
        # self.model.soft_contact_kf = kf
        # self.model.soft_contact_kd = kd
        # self.model.soft_contact_mu = mu
        # self.model.soft_contact_margin = 0.001
        # self.model.soft_contact_restitution = 1.0

        self.integrator = wp.sim.SemiImplicitIntegrator()


        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()


        # ???
        # wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        print(self.model.tet_materials)

        with wp.ScopedTimer("render", active=True):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="bowl_scene.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(example.fps*example.sim_duration):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()

