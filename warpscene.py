
import warp as wp
import warp.sim
import warp.sim.render
import time
import warp.optim

@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.transformf), target: wp.vec3, loss: wp.array(dtype=float)):
    delta = wp.transform_get_translation(pos[0]) - target
    # loss[0] = wp.dot(delta, delta)
    loss[0] = wp.dot(wp.vec3(1.0, 0.0, 0.0), wp.transform_get_translation(pos[0]))

@wp.kernel
def step_kernel(x: wp.array(dtype=float), grad: wp.array(dtype=float), alpha: float):
    tid = wp.tid()

    x[tid] = x[tid] - grad[tid] * alpha

class Scene:
    def __init__(self):
        # Simulation Parameters
        sim_duration = 5.0
        self.verbose = True

        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        # Learning Rate for Density Optimization
        self.train_rate = 1.0

        # Build Simulation
        builder = wp.sim.ModelBuilder()

        b = builder.add_body()
        # Initial Density
        self.density_value = 1000

        builder.add_shape_box(
            body=b,
            pos=(0, 0.3, 0),
            hx=0.3, hy=0.3, hz=0.3,
            density=self.density_value,
            kf=0.05
        )

        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True
        self.model.body_mass.requires_grad=True

        print(self.model.body_mass)

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # Target Position for Optimization
        self.target = (-8.75, 0.0, 0.0)
        # self.target = (0.0, 0.0, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        # self.state = self.model.state()

        wp.sim.collide(self.model, self.states[0])
        # wp.sim.collide(self.model, self.state)
        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, "Sim")

        self.use_cuda_graph = wp.get_device().is_cuda

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def forward(self):
        """Simulate 5 seconds of motion and compute loss."""
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])

            self.states[i].clear_forces()
            self.states[i].body_f.assign(
                [[0.0, 0.0, 0.0, -1000.0, 0.0, 0.0]] 
            )

            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

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
                self.tape.backward(self.loss)

            # Gradient Descent Step

            x = self.model.body_mass
            wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])
            x_grad = self.tape.gradients[self.model.body_mass]

            if self.verbose:
                print(f"Iter: {self.iter} Loss: {self.loss}")
                print(f"    x: {x} g: {x_grad}")

            self.tape.zero()
            self.iter += 1


    def render(self, i):
        """Render the simulation."""
        with wp.ScopedTimer("render"):
            for i in range(0, self.sim_steps, self.sim_substeps):
                self.renderer.begin_frame(self.render_time)
                # self.renderer.render(self.states[i])
                self.renderer.render(self.states[i])
                # print(self.states[i].body_f)
                # print(self.states[i].body_f)
                if i % 16 == 0:
                    print(self.states[i].body_q)

                self.renderer.end_frame()
                self.render_time += self.frame_dt

if __name__ == "__main__":
    iter = 2
    with wp.ScopedDevice(wp.get_cuda_devices()[0]):
        scene = Scene()
        for i in range(iter):
            scene.step()
            if i % 16 == 0:
                scene.render(i)
