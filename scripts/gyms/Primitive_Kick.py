from agent.Base_Agent import Base_Agent as Agent
from behaviors.custom.Kick.Kick import Kick
from behaviors.custom.Step.Step import Step
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os
import gymnasium as gym
import numpy as np
import math

class Primitive_Kick(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0
        self.max_episode_steps = 400  # ~8 seconds

        # Initialize behaviors
        self.kick_obj : Kick = self.player.behavior.get_custom_behavior_object("Kick")
        self.step_obj : Step = self.player.behavior.get_custom_behavior_object("Step")
        
        # State space (Basic_Run obs + ball info + kick state)
        obs_size = 81  # 70 base + 5 step + 6 kick
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_size, -np.inf, np.float32),
            high=np.full(obs_size, np.inf, np.float32),
            dtype=np.float32
        )

        # Action space (22 for walking (22 joints - 2 head joints + 2 walking params), 3 for kick params when close)
        self.no_of_actions = act_size = 25
        MAX = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(
            low=np.full(act_size, -MAX, np.float32),
            high=np.full(act_size, MAX, np.float32),
            dtype=np.float32
        )

        # Behavior defaults
        self.step_default_dur = 7
        self.step_default_z_span = 0.035
        self.step_default_z_max = 0.70
        self.kick_default_dur = 15
        self.kick_default_z_peak = 0.05
        self.kick_default_forward = 0.05
        self.kick_default_lateral = 0.05

        # Initial ball position
        self.ball_pos = np.zeros(3, np.float32)

    def observe(self, init=False):
        r = self.player.world.robot
        w = self.player.world

        # Basic_Run observations (first 70 elements)
        self.obs[0] = self.step_counter / 100
        self.obs[1] = r.loc_head_z * 3
        self.obs[2] = r.loc_head_z_vel / 2
        self.obs[3] = r.imu_torso_orientation / 50
        self.obs[4] = r.imu_torso_roll / 15
        self.obs[5] = r.imu_torso_pitch / 15
        self.obs[6:9] = r.gyro / 100
        self.obs[9:12] = r.acc / 10
        self.obs[12:18] = r.frp.get('lf', (0,0,0,0,0,0))
        self.obs[18:24] = r.frp.get('rf', (0,0,0,0,0,0))
        self.obs[15:18] /= 100
        self.obs[21:24] /= 100
        self.obs[24:44] = r.joints_position[2:22] / 100
        self.obs[44:64] = r.joints_speed[2:22] / 6.1395

        # Behavior state
        if init or not hasattr(self, 'in_kick_range'):
            self.obs[64] = self.step_default_dur / 10
            self.obs[65] = self.step_default_z_span * 20
            self.obs[66] = self.step_default_z_max
            self.obs[67] = 1  # step progress
            self.obs[68] = 1  # left leg active
        else:
            self.obs[64] = self.step_obj.step_generator.ts_per_step / 10
            self.obs[65] = self.step_obj.step_generator.swing_height * 20
            self.obs[66] = self.step_obj.step_generator.max_leg_extension / self.step_obj.leg_length
            self.obs[67] = self.step_obj.step_generator.external_progress
            self.obs[68] = float(self.step_obj.step_generator.state_is_left_active)

        # Kick parameters (indices 69-74)
        if init or not hasattr(self, 'in_kick_range') or not self.in_kick_range:
            self.obs[69:75] = 0
        else:
            self.obs[69] = self.kick_obj.kick_generator.ts_total / 20
            self.obs[70] = self.kick_obj.kick_generator.swing_height * 20
            self.obs[71] = self.kick_obj.kick_generator.forward_reach * 20
            self.obs[72] = self.kick_obj.kick_generator.lateral_dev * 20
            self.obs[73] = self.kick_obj.kick_generator.external_progress
            self.obs[74] = float(self.kick_obj.kick_generator.kick_complete)

        # Ball observations (75-80)
        ball_rel = w.ball_cheat_abs_pos - r.cheat_abs_pos
        self.obs[75:78] = ball_rel  # relative position
        self.obs[78:81] = w.ball_cheat_abs_vel  # velocity

        return self.obs

    def sync(self):
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def reset(self, seed=None):
        self.step_counter = 0
        r = self.player.world.robot
        
        # Reset robot and ball positions
        self.player.scom.unofficial_beam((10, 0, r.beam_height), 0)
        self.ball_pos = np.array([12, 0, 0.042])
        self.player.scom.unofficial_move_ball(self.ball_pos)
        
        # Stabilize
        for _ in range(25):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()

        # Initialize tracking variables
        self.last_ball_dist = np.linalg.norm(self.ball_pos[:2] - r.cheat_abs_pos[:2])
        self.act = np.zeros(self.no_of_actions, np.float32)
        self.in_kick_range = False
        self.kick_started = False

        return self.observe(True), {}

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

    def step(self, action):
        r = self.player.world.robot
        w = self.player.world
        b_pos = w.ball_cheat_abs_pos
        ball_rel = b_pos[:2] - r.cheat_abs_pos[:2]
        b_vel = w.ball_cheat_abs_vel[:2]


        # Exponential moving average for smooth actions
        self.act = 0.4 * self.act + 0.6 * action

        # Calculate ball distance and determine if in kicking range
        ball_dist = np.linalg.norm(ball_rel)

        self.in_kick_range = ball_dist < 0.3  # 30cm threshold for kicking

        obj_in_use = None
        if self.in_kick_range:
            # Start kick sequence
            self.kick_started = True
            
            # We use max(min()) instead of clip() for two reasons:
            # 1. Clearer intent when we have asymmetric bounds
            # 2. Better handling of edge cases in RL training
            # Extract kick parameters from action (last 4 elements)
            kick_z = max(0.02, min(0.08, self.kick_default_z_peak + self.act[22] / 100))
            kick_fwd = max(0.02, min(0.10, self.kick_default_forward + self.act[23] / 100))
            kick_lat = np.clip(self.kick_default_lateral + self.act[24] / 100, -0.03, 0.03)

            # Execute kick with current parameters
            self.player.behavior.execute("Kick", self.kick_default_dur, kick_z, kick_fwd, kick_lat)
            obj_in_use = self.kick_obj

            #TODO maybe treat here the other joint actions

        elif not self.in_kick_range and self.kick_started:

            self.player.behavior.execute("Zero_Bent_Knees")

        else:
            if self.step_counter == 0:
                '''
                The first time step will change the parameters of the next footstep
                It uses default parameters so that the agent can anticipate the next generated pose
                Reason: the agent decides the parameters during the previous footstep
                '''
                self.player.behavior.execute("Step", self.step_default_dur, self.step_default_z_span, self.step_default_z_max)
            else:
                # Walk toward ball using Step behavior
                step_zsp = np.clip(self.step_default_z_span + self.act[20]/300, 0, 0.07)
                step_zmx = np.clip(self.step_default_z_max + self.act[21]/30, 0.6, 0.9)
                
                self.player.behavior.execute("Step", self.step_default_dur, step_zsp, step_zmx)
            obj_in_use = self.step_obj
            
        # Add action as residuals to Step and Kick behavior
        if obj_in_use is not None:
            new_action = self.act[:20] * 2
            new_action[[0,2,4,6,8,10]] += obj_in_use.values_l
            new_action[[1,3,5,7,9,11]] += obj_in_use.values_r
            new_action[12] -= 90
            new_action[13] -= 90
            new_action[16] += 90
            new_action[17] += 90
            new_action[18] += 90
            new_action[19] += 90
        
            r.set_joints_target_position_direct(
                slice(2,22), new_action, harmonize=False
            )

        self.sync()
        self.step_counter += 1

        ball_speed = np.linalg.norm(b_vel)

        # Reward calculation
        reward = 0
        if self.kick_started:
            # Reward based on ball velocity after kick
            reward = 5.0 * ball_speed
        else:
            reward = self.last_ball_dist - ball_dist  # Reward for reducing distance
            self.last_ball_dist = ball_dist

            if self.in_kick_range:
                reward += 1.0  # bonus for reaching kick range

        # Terminal conditions
        terminal = (
            (r.cheat_abs_pos[2] < 0.3 and ball_speed < 0.01) or  # Fell down
            self.step_counter >= self.max_episode_steps or
            (self.kick_started and ball_speed < 0.01 and not self.in_kick_range)  # Ball stopped moving after kick
        )

        return self.observe(), reward, terminal, False, {}

# Keep the same Train class as before
class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)


    def train(self, args):

        #--------------------------------------- Learning parameters
        n_envs = min(16, os.cpu_count())
        n_steps_per_env = 1024  # RolloutBuffer is of size (n_steps_per_env * n_envs)
        minibatch_size = 64    # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 30000000
        learning_rate = 3e-4
        folder_name = f'Basic_Kick_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        #--------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return Primitive_Kick( self.ip , self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False )
            return thunk

        servers = Server( self.server_p, self.monitor_p_1000, n_envs+1 ) #include 1 extra server for testing

        env = SubprocVecEnv( [init_env(i) for i in range(n_envs)] )
        eval_env = SubprocVecEnv( [init_env(n_envs)] )

        try:
            if "model_file" in args: # retrain
                model = PPO.load( args["model_file"], env=env, device="cpu", n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate )
            else: # train new model
                model = PPO( "MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate, device="cpu" )

            model_path = self.learn_model( model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env*20, save_freq=n_steps_per_env*200, backup_env_file=__file__ )
        except KeyboardInterrupt:
            sleep(1) # wait for child processes
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return
    
        env.close()
        eval_env.close()
        servers.kill()
        

    def test(self, args):

        # Uses different server and monitor ports
        server = Server( self.server_p-1, self.monitor_p, 1 )
        env = Primitive_Kick( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True )
        model = PPO.load( args["model_file"], env=env )

        try:
            self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
            self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
