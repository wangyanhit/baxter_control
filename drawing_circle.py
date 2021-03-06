#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Modified Baxter RSDK Joint Torque Example: task springs
"""

import argparse

import rospy

from dynamic_reconfigure.server import (
    Server,
)
from std_msgs.msg import (
    Empty,
)
from baxter_core_msgs.msg import  (
    SEAJointState,
)
import baxter_interface

from baxter_examples.cfg import (
    DrawingCircleExampleConfig,
)
from baxter_interface import CHECK_VERSION

from baxter_pykdl import baxter_kinematics

from time import time

import numpy as np

from copy import deepcopy

import csv

from pyquaternion import Quaternion


class JointSprings(object):
    """
    Virtual Joint Springs class for torque example.

    @param limb: limb on which to run joint springs example
    @param reconfig_server: dynamic reconfigure server

    JointSprings class contains methods for the joint torque example allowing
    moving the limb to a neutral location, entering torque mode, and attaching
    virtual springs.
    """
    def __init__(self, limb, reconfig_server):
        self._dyn = reconfig_server

        # control parameters
        self._rate = 100.0  # Hz
        self._missed_cmds = 20.0  # Missed cycles before triggering timeout
        self.joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
        # define start and end time to calcualte velocity
        self._end = time()
        self._start = time()
        # create our limb instance
        self._limb = baxter_interface.Limb(limb)
        self._start_angles = dict()
        # self._dof_names = {'x': 1, 'y': 2, 'z': 3}
        self._dof_names = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._start_pos = [0, 0, 0]
        self._start_q = Quaternion(axis=[0, 1, 0], angle=3.14/2)
        self._kin = baxter_kinematics('left')
        # gravity compensation
        self._gravity_comp_effort = dict()
        self._gravity_comp_effort_ok = False
        gravity_comp_topic = '/robot/limb/left/gravity_compensation_torques'
        self._gravity_comp_sub = rospy.Subscriber(
            gravity_comp_topic,
            SEAJointState,
            self._on_gravity_comp,
            queue_size=1,
            tcp_nodelay=True)

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")
        self.safe_file = open('mani.csv', 'w')

    def _on_gravity_comp(self, msg):
        #print('gravity compensation effort received!')
        self._gravity_comp_effort_ok = True
        #print(msg.name)
        #print(msg.gravity_model_effort)
        for idx, name in enumerate(msg.name):
            if name in self._gravity_comp_effort:
                self._gravity_comp_effort[name] = msg.gravity_model_effort[idx]

    def _update_parameters(self):
        for dof in self._dof_names:
            self._springs[dof] = self._dyn.config[dof[-2:] + '_spring_stiffness']
            self._damping[dof] = self._dyn.config[dof[-2:] + '_damping_coefficient']

    def _update_forces(self):
        """
        Calculates the current angular difference between the start position
        and the current joint positions applying the joint torque spring forces
        as defined on the dynamic reconfigure server.
        """
        # get latest spring constants
        self._update_parameters()

        # disable cuff interaction
        self._pub_cuff_disable.publish()

        # create our command dict
        cmd = dict()
        # record current angles/velocities
        cur_pos = self._kin.forward_position_kinematics()[0:3]
        q = self._limb.endpoint_pose()['orientation']
        cur_q = Quaternion([q[3], q[0], q[1], q[2]])
        self._end = time()
        #cur_vel = (cur_pos - self._last_pos)/(-self._start + self._end)
        #print "endpoint velocity:"
        #print self._limb.endpoint_velocity()['linear']
        cur_linear_vel = self._limb.endpoint_velocity()['linear']
        cur_angular_vel = self._limb.endpoint_velocity()['angular']
        self._start = time()
        self._last_pos = cur_pos
        J_trans = self._kin.jacobian_transpose()
        # calculate manipulability
        angles = self._limb.joint_angles()
        # angles_raw = self._limb.joint_angles()
        # angles = [0, 0, 0, 0, 0, 0, 0]
        # for idx, name in enumerate(self.joint_names):
        #     angles[idx] = angles_raw[name]
        J = self._kin.jacobian()
        M = np.sqrt(np.linalg.det(np.matmul(J, J_trans)))
        #print "M is: "
        #print M
        Mq = [0, 0, 0, 0, 0, 0, 0]
        delta_q = 0.1
        for idx, name in enumerate(self.joint_names):
            angles_temp = deepcopy(angles)
            angles_temp[name] = angles[name] + delta_q
            J_hat = self._kin.jacobian(angles_temp)
            J_trans_hat = self._kin.jacobian_transpose(angles_temp)
            M_temp = np.sqrt(np.linalg.det(np.matmul(J_hat, J_trans_hat)))
            Mq[idx] = (M_temp - M) / (delta_q)
        M_limit = 0.07
        M_limit = 1
        tau_singularity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lamda = 5
        if M < M_limit:
            k = 1/(M + 0.000001) - 1/(M_limit + 0.00001)
            for i in range(7):
                tau_singularity[i] = lamda * Mq[i] * k
        # print("tau_singularity is: ")
        # print(tau_singularity)
        # calculate current forces
        f_cmd = np.matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

        for i in range(0, 3):
            # spring portion
            position_error = cur_pos[i] - self._start_pos[i]
            # print("{}'s error: {}\n".format(self._dof_names[i], position_error))
            f_cmd[i, 0] = -self._springs[self._dof_names[i]] * position_error
            # damping portion
            f_cmd[i, 0] -= self._damping[self._dof_names[i]] * cur_linear_vel[i]
        q_error = cur_q * self._start_q.conjugate
        theta = np.arccos(q_error[0])
        axis = np.array([q_error[1], q_error[2], q_error[3]])
        angle_error = axis*theta

        for i in range(3):
            #angle_error
            f_cmd[i+3, 0] = -self._springs[self._dof_names[i+3]] * angle_error[i]
            f_cmd[i+3, 0] -= self._damping[self._dof_names[i+3]] * cur_angular_vel[i]


        torque_cmd = np.matmul(J_trans, f_cmd)
        # gravity compensation portion
        for joint in self._start_angles.keys():
            cmd[joint] = 0
            if self._gravity_comp_effort_ok and joint in self._gravity_comp_effort.keys():
                cmd[joint] = self._gravity_comp_effort[joint] * 0.01
        # command new joint torques
        tau_null_space = np.matmul(np.identity(7) - np.matmul(J_trans, np.transpose(np.linalg.pinv(J))), np.transpose(tau_singularity))
        tau_null_space = np.transpose(tau_null_space)
        for idx, name in enumerate(self.joint_names):
            cmd[name] += torque_cmd[idx, 0] + tau_singularity[idx]
            #cmd[name] += torque_cmd[idx, 0] + tau_null_space[idx]
            #cmd[name] += torque_cmd[idx, 0]
            # print(torque_cmd[idx, 0])
        # print(self._gravity_comp_effort)
        # print(cmd)
        self._limb.set_joint_torques(cmd)
        # save file
        save = [M, tau_singularity[0], tau_singularity[1], tau_singularity[2], tau_singularity[3], tau_singularity[4],
                tau_singularity[5], tau_singularity[6]]
        self.safe_file = open('mani.csv', 'a')
        with self.safe_file:
            writer = csv.writer(self.safe_file)
            writer.writerow(list(save))

    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

    def attach_springs(self):
        """
        Switches to joint torque mode and attached joint springs to current
        joint positions.
        """
        # record initial joint angles
        self._start_angles = self._limb.joint_angles()
        self._start_pos = self._kin.forward_position_kinematics()[0:3]
        self._last_pos = self._start_pos
        self._start = time()
        # set control rate
        control_rate = rospy.Rate(self._rate)

        # for safety purposes, set the control rate command timeout.
        # if the specified number of command cycles are missed, the robot
        # will timeout and disable
        self._limb.set_command_timeout((1.0 / self._rate) * self._missed_cmds)

        # loop at specified rate commanding new joint torques
        while not rospy.is_shutdown():
            if not self._rs.state().enabled:
                rospy.logerr("Task torque example failed to meet "
                             "specified control rate timeout.")
                break
            self._update_forces()
            control_rate.sleep()

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()


def main():
    """Modified RSDK Joint Torque Example: Task Springs

    Moves the specified limb to a neutral location and enters
    torque control mode, attaching virtual springs (Hooke's Law)
    to each joint maintaining the start position.

    Run this example on the specified limb and interact by
    grabbing, pushing, and rotating each joint to feel the torques
    applied that represent the virtual springs attached.
    You can adjust the spring constant and damping coefficient
    for each joint using dynamic_reconfigure.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-l', '--limb', dest='limb', required=True, choices=['left', 'right'],
        help='limb on which to attach joint springs'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("rsdk_drawing_circle_%s" % (args.limb,))
    dynamic_cfg_srv = Server(DrawingCircleExampleConfig,
                             lambda config, level: config)
    js = JointSprings(args.limb, dynamic_cfg_srv)
    # register shutdown callback
    rospy.on_shutdown(js.clean_shutdown)
    js.move_to_neutral()
    js.attach_springs()


if __name__ == "__main__":
    main()
