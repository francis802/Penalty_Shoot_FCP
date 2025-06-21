from agent.Base_Agent import Base_Agent
from behaviors.custom.Kick.Kick_Generator import Kick_Generator
import numpy as np

class Kick():
    def __init__(self, base_agent: Base_Agent) -> None:
        self.world = base_agent.world
        self.ik = base_agent.inv_kinematics
        self.description = "Kick (Skill-Set-Primitive)"
        self.auto_head = True

        nao_specs = self.ik.NAO_SPECS
        self.leg_length = nao_specs[1] + nao_specs[3]  # upper leg + lower leg
        sample_time = self.world.robot.STEPTIME
        max_ankle_z = nao_specs[5]

        # Initialize Kick Generator
        self.kick_generator = Kick_Generator(sample_time, max_ankle_z)
        self.kicking_leg_is_left = True  # Default, can be toggled externally

    def execute(self, reset, ts_total=15, z_peak=0.05, forward_max=0.05, lateral_dev=0.05, z_max=0.8):
        if reset:
            max_extension = self.leg_length * z_max
            self.kick_generator.reset(
                ts_total, z_peak, forward_max, lateral_dev,
                max_leg_extension=max_extension,
                is_left_active=self.kicking_leg_is_left
            )

        # Get target foot positions
        lfy, lfz, rfy, rfz = self.kick_generator.get_target_positions()

        # Get joint angles from generator
        joint_angles = self.kick_generator.get_joint_angles()

        # Use joint angles for support leg to enforce leaning
        if self.kicking_leg_is_left:
            # Left leg is active (kicking), right leg is support
            support_leg_angles = [  # Right leg joint values
                joint_angles['LLegRoll'],        # id=5 (mirrored)
                joint_angles['LLegPitch'],       # id=6
                joint_angles['RLegPitch'],       # id=7
                joint_angles['LKneePitch'],      # id=8
                joint_angles['RKneePitch'],      # id=9
                joint_angles['RAnklePitch'],     # id=11
            ]
            joint_ids = [5, 6, 7, 8, 9, 11]
        else:
            # Right leg is active (kicking), left leg is support
            support_leg_angles = [
                -joint_angles['LLegRoll'],       # mirror roll
                joint_angles['LLegPitch'],
                joint_angles['RLegPitch'],
                joint_angles['LKneePitch'],
                joint_angles['RKneePitch'],
                joint_angles['RAnklePitch'],
            ]
            joint_ids = [5, 6, 7, 8, 9, 11]

        # Apply joint angles for leaning
        self.world.robot.set_joints_target_position_direct(joint_ids, np.array(support_leg_angles))

        # Apply IK to the kicking leg for foot placement
        if self.kicking_leg_is_left:
            indices, self.values_l, error_codes = self.ik.leg((0, lfy, lfz), (0, 0, 0), True, dynamic_pose=False)
        else:
            indices, self.values_r, error_codes = self.ik.leg((0, rfy, rfz), (0, 0, 0), False, dynamic_pose=False)

        for i in error_codes:
            print(f"Joint {i} is out of range!" if i != -1 else "Position is out of reach!")
        self.world.robot.set_joints_target_position_direct(indices, self.values_l if self.kicking_leg_is_left else self.values_r)

        # Fixed arms
        arm_indices = [14, 16, 18, 20]
        arm_values = np.array([-80, 20, 90, 0])
        self.world.robot.set_joints_target_position_direct(arm_indices, arm_values)
        self.world.robot.set_joints_target_position_direct([i + 1 for i in arm_indices], arm_values)

        return self.kick_generator.kick_complete

    def is_ready(self):
        ''' Returns True if Kick Behavior is ready to start under current game/robot conditions '''
        return self.kick_generator.kick_complete
