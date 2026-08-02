import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time

from constants import *
from motions import trefoil

# https://mathcurve.com/courbes3d.gb/noeuds/noeudenhuit.shtml

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Simulation timestep in seconds.
dt: float = 1.0 / 840.0  # matches the IMU frequency

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0


def setup_model(
    fp: str, dt: float = 1.0 / 840.0, gravity_compensation: bool = True
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(fp)
    data = mujoco.MjData(model)
    model.opt.timestep = dt  # sim timestep = IMU frequency.
    body_ids = [model.body(name).id for name in BODY_NAMES]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0
    return model, data


def camera_calibrate():
    pass


def zero_bias(model, t: float, disable_t: float = 10.0, transition_t: float = 1.0):
    if t < transition_t:
        return model.key("zero_bias").ctrl
    elif t < disable_t + transition_t:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        return model.key("zero_bias").ctrl
    else:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        return model.key("home").ctrl


def ellipsoid_fit():
    pass


def main() -> None:
    # load simulated model of cobot and initialize data
    fp = "./mycobot_280/scene.xml"
    model, data = setup_model(fp)

    # get id of the end effector we want to control
    ee_id = model.site(END_EFFECTOR_NAME).id

    # grab the accelerometer and gyroscope
    gyro_adr = model.sensor(GYRO_NAME).adr[0]
    accel_adr = model.sensor(ACCEL_NAME).adr[0]

    # get all ids for DOFs that we control
    dof_ids = np.array([model.joint(name).id for name in JOINT_NAMES])
    # get all ids for actuators that we control (same as DOFs)
    actuator_ids = np.array([model.actuator(name).id for name in JOINT_NAMES])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    # Mocap body we will control with our mouse.
    mocap_id = model.body("target").mocapid[0]
    tag_ee_id = model.site("tag_down_right").id

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    def lookat_quat(eye, target, axis="z", up=np.array([0.0, 0.0, -1.0])):
        """Quaternion that points the site's given local axis from eye toward target.
        axis: 'x', 'y', 'z', '-x', '-y', '-z'"""
        gaze = target - eye
        gaze = gaze / np.linalg.norm(gaze)
        if axis.startswith("-"):
            gaze = -gaze
        if abs(np.dot(gaze, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])

        a = np.cross(up, gaze)
        a /= np.linalg.norm(a)
        b = np.cross(gaze, a)

        ax = axis[-1]
        if ax == "z":
            cols = [a, b, gaze]  # x, y, z columns
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

        global_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Set the target position of the end-effector site.
            data.mocap_pos[mocap_id] = trefoil(data.time)
            data.mocap_quat[mocap_id] = lookat_quat(
                data.mocap_pos[mocap_id],
                data.site(tag_ee_id).xpos,
                axis="x",
            )

            # Position error.
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(ee_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(ee_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], ee_id)

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
            reference_time = time.time() - global_time
            data.ctrl[actuator_ids] = q[
                actuator_ids
            ]  # zero_bias(model, reference_time)

            # Step the simulation.
            mujoco.mj_step(model, data)

            # in the loop, after mj_step:
            gyro = data.sensordata[gyro_adr : gyro_adr + 3]
            accel = data.sensordata[accel_adr : accel_adr + 3]

            imu = np.concatenate(
                [
                    gyro,
                    accel,
                ]
            )

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
