import math
import numpy as np

class Kick_Generator:
    GRAVITY = 9.81
    Z0 = 0.2  # COM height

    def __init__(self, sample_time, max_ankle_z) -> None:
        self.sample_time = sample_time
        self.max_ankle_z = max_ankle_z
        self.kick_complete = False
        self.state_is_left_active = True

    def reset(self, ts_total, z_peak, forward_max, lateral_dev, max_leg_extension, is_left_active=True):
        self.ts_total = ts_total
        self.forward_reach = forward_max
        self.lateral_dev = lateral_dev
        self.max_leg_extension = max_leg_extension
        self.swing_height = min(z_peak, self.max_ankle_z - (-max_leg_extension))
        self.state_is_left_active = is_left_active
        self.state_current_ts = 0
        self.kick_complete = False

    def get_target_positions(self):
        if self.kick_complete:
            return 0, 0, 0, 0

        if not hasattr(self, "ts_total") or self.ts_total <= 1:
            raise ValueError("Kick_Generator not reset properly: ts_total <= 1")

        time_delta = self.state_current_ts * self.sample_time
        step_time = self.ts_total * self.sample_time
        progress = self.state_current_ts / self.ts_total
        self.external_progress = progress

        # COM trajectory lateral component
        W = math.sqrt(self.Z0 / self.GRAVITY)
        y0 = self.lateral_dev
        com_y = y0 + y0 * (math.sinh((step_time - time_delta) / W) + math.sinh(time_delta / W)) / math.sinh(-step_time / W)

        # Foot swing trajectory (sine-based)
        swing_y = self.forward_reach * math.sin(math.pi * progress)
        swing_z = self.swing_height * math.sin(math.pi * progress)

        # Initial foot Z
        z0 = min(-self.max_leg_extension, self.max_ankle_z)

        # Kick joint modeling (math-parametrized imitation of slot motion)
        if progress < 0.5:
            lean_progress = progress / 0.5

            self.LLegRoll = -10 + 5 * math.sin(math.pi * lean_progress)
            self.LLegPitch = 40 - 60 * lean_progress
            self.RLegPitch = 65
            self.LKneePitch = -60 + 10 * math.sin(math.pi * lean_progress)
            self.RKneePitch = -115 + 15 * math.sin(math.pi * lean_progress)
            self.RAnklePitch = 10 + 20 * math.sin(math.pi * lean_progress)
        else:
            kick_progress = (progress - 0.5) / 0.5

            self.LLegRoll = -5
            self.LLegPitch = -25
            self.RLegPitch = 80 + 20 * math.sin(math.pi * kick_progress)
            self.LKneePitch = 0
            self.RKneePitch = 0
            self.RAnklePitch = 25

        self.state_current_ts += 1
        if self.state_current_ts >= self.ts_total:
            self.kick_complete = True

        if self.state_is_left_active:
            return com_y + swing_y, z0 + swing_z, -com_y, z0
        else:
            return com_y, z0, -com_y + swing_y, z0 + swing_z

    def get_joint_angles(self):
        return {
            'LLegRoll': self.LLegRoll,
            'LLegPitch': self.LLegPitch,
            'RLegPitch': self.RLegPitch,
            'LKneePitch': self.LKneePitch,
            'RKneePitch': self.RKneePitch,
            'RAnklePitch': self.RAnklePitch
        }
