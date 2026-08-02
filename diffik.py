import mujoco
import mujoco.viewer
import numpy as np
import time
import multiprocessing as mp
from collections import deque

def plot_process(q: mp.Queue):
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    import queue as queue_mod

    n = 2000  # samples shown
    t_buf = deque(maxlen=n)
    bufs = [deque(maxlen=n) for _ in range(6)]
    labels = ["gx", "gy", "gz", "ax", "ay", "az"]

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    lines = []
    for i in range(3):
        lines.append(ax1.plot([], [], label=labels[i])[0])
    for i in range(3, 6):
        lines.append(ax2.plot([], [], label=labels[i])[0])
    ax1.set_ylabel("gyro [rad/s]"); ax1.legend(loc="upper right")
    ax2.set_ylabel("accel [m/s²]"); ax2.set_xlabel("time [s]"); ax2.legend(loc="upper right")

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    while True:
        # Drain everything queued since the last redraw.
        drained = False
        try:
            while True:
                # Block briefly for the first item, then grab the rest non-blocking.
                item = q.get(block=not drained, timeout=0.03)
                if item is None:
                    return
                t, vals = item
                t_buf.append(t)
                for i in range(6):
                    bufs[i].append(vals[i])
                drained = True
        except queue_mod.Empty:
            pass

        if drained:
            for i, ln in enumerate(lines):
                ln.set_data(t_buf, bufs[i])
            ax1.relim(); ax1.autoscale_view()
            ax2.relim(); ax2.autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.03)

# https://mathcurve.com/courbes3d.gb/noeuds/noeudenhuit.shtml

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 1.0 / 840.0     # matches the IMU frequency

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    import sys
    import os
    if sys.platform == "darwin":
        mp.set_start_method("spawn", force=True)
        mp.set_executable(os.path.join(sys.prefix, "bin", "python3"))

    queue = mp.Queue()
    proc = mp.Process(target=plot_process, args=(queue,), daemon=True)
    proc.start()

    # Load the model and data.
    #model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    model = mujoco.MjModel.from_xml_path("mycobot_280/scene.xml")
    data = mujoco.MjData(model)

    # Override the simulation timestep.
    model.opt.timestep = dt

    # End-effector site we wish to control, in this case a site attached to the last
    # link (wrist_3_link) of the robot.
    site_id = model.site("attachment_site").id

    gyro_adr = model.sensor("imu_gyro").adr[0]
    accel_adr = model.sensor("imu_acc").adr[0]

    # Name of bodies we wish to apply gravity compensation to.
    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    # Note that actuator names are the same as joint names in this case.
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("calibrate").id

    # Mocap body we will control with our mouse.
    mocap_id = model.body("target").mocapid[0]
    tag_site_id = model.site("tag_down_right").id

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Define a trajectory for the end-effector site to follow.
    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        z = 0.2
        return np.array([x, y, z])

    def trefoil(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        x = r * np.cos(2 * np.pi * f * t) + 2 * r * np.cos(4 * np.pi * f * t) + h
        y = 5 * r * np.sin(2 * np.pi * f * t) - 5 * 2 * r * np.sin(4 * np.pi * f * t) + k
        z = 2 * 2 * r * np.sin(3 * np.pi * f * t) + 0.15
        return np.array([x, y, z])

    def lookat_quat(eye, target, axis="z", up=np.array([0.0, 0.0, -1.0])):
        """Quaternion that points the site's given local axis from eye toward target.
        axis: 'x', 'y', 'z', '-x', '-y', '-z'"""
        gaze = target - eye
        gaze = gaze / np.linalg.norm(gaze)
        if axis.startswith("-"):
            gaze = -gaze
        if abs(np.dot(gaze, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])

        a = np.cross(up, gaze); a /= np.linalg.norm(a)
        b = np.cross(gaze, a)

        ax = axis[-1]
        if ax == "z":
            cols = [a, b, gaze]        # x, y, z columns
        elif ax == "x":
            cols = [gaze, a, b]
        else:  # 'y'
            cols = [b, gaze, a]

        mat = np.stack(cols, axis=1).flatten()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return quat

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        cam_id = model.camera("wrist_cam").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.opt.sitegroup[1] = 0
        viewer.opt.sitegroup[2] = 0

        while viewer.is_running():
            step_start = time.time()

            # Set the target position of the end-effector site.
            #data.mocap_pos[mocap_id] = circle(data.time, 0.1, 0.5, 0.0, 0.5)
            #data.mocap_pos[mocap_id] = trefoil(data.time, 0.075, 0.4, 0.0, 0.2)
            data.mocap_pos[mocap_id] = trefoil(data.time, 0.01,  0.2, 0.0, 0.2)
            #data.mocap_pos[mocap_id] = circle(data.time, 0.03,  0.15, 0.0, 0.2)
            data.mocap_quat[mocap_id] = lookat_quat(
                data.mocap_pos[mocap_id],
                data.site(tag_site_id).xpos,
                axis="x",
            )   

            # Position error.
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal.
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)

            # in the loop, after mj_step:
            gyro = data.sensordata[gyro_adr:gyro_adr + 3]
            accel = data.sensordata[accel_adr:accel_adr + 3]

            imu = np.concatenate([
                gyro,
                accel,
            ])
            queue.put_nowait((data.time, imu))

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        queue.put(None)


if __name__ == "__main__":
    main()
