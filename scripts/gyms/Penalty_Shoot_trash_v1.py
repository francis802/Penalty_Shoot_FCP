from agent.Base_Agent import Base_Agent as Agent
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os
import gymnasium as gym
import numpy as np
from math_ops.Math_Ops import Math_Ops as M





'''
Objective:
Learn how to fall (simplest example)
----------
- class Fall: implements an OpenAI custom gym
- class Train:  implements algorithms to train a new model or test an existing model
'''

class Penalty_Shoot(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:

        self.robot_type = r_type

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0 # to limit episode size

        # State space
        self.no_of_joints = self.player.world.robot.no_of_joints
        obs_dim = self.no_of_joints + 1 + 3 + 1 + 2  # joints + torso_z + robot_pos + dist + ball_vel
        self.obs = np.zeros(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space
        MAX = np.finfo(np.float32).max
        no_of_actions = self.no_of_joints
        self.action_space = gym.spaces.Box(low=np.full(no_of_actions,-MAX,np.float32), high=np.full(no_of_actions,MAX,np.float32), dtype=np.float32)

        self.player.scom.unofficial_move_ball((9, 0, 0.042))

        # Check if cheats are enabled
        assert np.any(self.player.world.robot.cheat_abs_pos), "Cheats are not enabled! Run_Utils.py -> Server -> Cheats"
        

    def observe(self):
        r = self.player.world.robot
        w = self.player.world

        # Joint positions (scaled)
        joint_positions = r.joints_position / 100.0

        # Torso height (z)
        torso_height = r.cheat_abs_pos[2]

        # Robot absolute position (x, y, z)
        robot_pos = r.cheat_abs_pos[:3]

        # Ball is always at fixed position (e.g., (9, 0))
        ball_pos = np.array([9.0, 0.0])

        # Distance to ball in 2D
        dist_to_ball = np.linalg.norm(ball_pos - robot_pos[:2])

        # Ball velocity (x, y)
        ball_vel = w.get_ball_abs_vel(6)[:2]

        # Final observation vector
        obs = np.concatenate([
            joint_positions,
            [torso_height],
            robot_pos,
            [dist_to_ball],
            ball_vel
        ]).astype(np.float32)

        self.obs = obs
        return self.obs


    def sync(self):
        ''' Run a single simulation step '''
        r = self.player.world.robot
        self.player.scom.commit_and_send( r.get_command() )
        self.player.scom.receive()


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # if you're inheriting from gym.Env, this sets the seed properly

        self.step_counter = 0
        r = self.player.world.robot

        for _ in range(25):
            self.player.scom.unofficial_beam((7, 0, 0.50), 0)
            self.player.behavior.execute("Zero")
            self.sync()

        self.player.scom.unofficial_beam((7, 0, r.beam_height), 0)
        
        r.joints_target_speed[0] = 0.01
        self.sync()

        for _ in range(7):
            self.player.behavior.execute("Zero")
            self.sync()

        obs = self.observe()
        return obs, {}  # <-- Now returns a tuple: (observation, info)


    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

    def step(self, action):
        r = self.player.world.robot
        w = self.player.world

        r.set_joints_target_position_direct(
            slice(self.no_of_joints),
            action * 10,
            harmonize=True
        )

        self.sync()
        self.step_counter += 1
        self.observe()

        reward = self.reward_function()
        done = self.is_terminal_state()

        return self.obs, reward, done, False, {}

        
    def reward_function(self):
        world = self.player.world
        robot = world.robot
        ball = world.ball_cheat_abs_pos

        # --- Blending factor ---
        ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        alpha = np.clip(ball_speed / 1.0, 0, 1)

        # --- Walking reward ---
        kick_target_pos, _, dist_to_kick_pos = self.player.path_manager.get_path_to_ball(
            x_ori=120, x_dev=-0.2, y_dev=-0.1, torso_ori=120
        )
        r_walk_dist = np.clip(1 - dist_to_kick_pos / 1.0, 0, 1)
        ori_diff = abs(M.normalize_deg(120 - robot.loc_torso_orientation))
        r_facing = np.clip(1 - ori_diff / 180, 0, 1)
        ball_rel = world.ball_rel_torso_cart_pos
        in_kick_x = 0.19 < ball_rel[0] < 0.22
        in_kick_y = -0.12 < ball_rel[1] < -0.1
        r_alignment = 1.0 if in_kick_x and in_kick_y else 0.0
        robot_speed = np.linalg.norm(robot.get_head_abs_vel(2)[:2])
        r_movement = np.clip(robot_speed / 1.0, 0, 1)

        walking_reward = (
            1.5 * r_walk_dist +
            1.0 * r_facing +
            1.0 * r_alignment +
            0.5 * r_movement
        )

        # --- Kick reward ---
        goal_center = np.array([15, 0])
        v = world.get_ball_abs_vel(6)[:2]
        goal_vec = goal_center - ball[:2]
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)
        speed_toward_goal = np.dot(v, goal_dir)
        r1 = np.clip(ball_speed / 3.0, 0, 1)
        r2 = np.clip(speed_toward_goal / 4.0, 0, 1)

        goal = ball[0] >= 15 and -1.1 < ball[1] < 1.1 and ball[2] < 0.8
        out_of_bounds = ball[0] >= 15
        r3 = 5.0 if goal else -1.0 if out_of_bounds else 0

        r4 = 0
        if goal:
            left_corner = np.array([15, -1.05])
            right_corner = np.array([15, 1.05])
            corner_reward = max(0, 1 - min(np.linalg.norm(ball[:2] - left_corner),
                                        np.linalg.norm(ball[:2] - right_corner)) / 1.5)
            r4 = 1.0 * corner_reward

        kick_reward = r1 + r2 + r3 + r4

        # --- Blended reward ---
        return (1 - alpha) * walking_reward + alpha * kick_reward

    

    
    def is_terminal_state(self):
        w = self.player.world
        torso_height = self.obs[-1]
        ball_pos = w.ball_cheat_abs_pos
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])

        # Goal detection
        goal = ball_pos[0] >= 15 and -1.1 < ball_pos[1] < 1.1 and ball_pos[2] < 0.8
        out_of_bounds = ball_pos[0] >= 15
        robot_fallen = torso_height < 0.15
        low_motion = ball_speed < 0.05

        # Ending conditions
        if goal or out_of_bounds:
            return True
        elif robot_fallen and low_motion:
            return True
        elif self.step_counter > 1000: # 20s passed and robot has not fallen (may be stuck)
            return True
        else:
            return False



class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)


    def train(self, args):

        #--------------------------------------- Learning parameters
        n_envs = min(4, os.cpu_count())
        n_steps_per_env = 2048   # RolloutBuffer is of size (n_steps_per_env * n_envs) (*RV: >=2048)
        minibatch_size = 256     # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 10000000     # (*RV: >=10M)
        learning_rate = 3e-4   # (*RV: 3e-4)
        # *RV -> Recommended value for more complex environments
        folder_name = f'Penalty_Shoot_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        #--------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return Penalty_Shoot( self.ip , self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False )
            return thunk

        servers = Server( self.server_p, self.monitor_p_1000, n_envs+1 ) #include 1 extra server for testing

        env = SubprocVecEnv( [init_env(i) for i in range(n_envs)] )
        eval_env = SubprocVecEnv( [init_env(n_envs)] )

        try:
            if "model_file" in args: # retrain
                model = PPO.load( args["model_file"], env=env, n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate )
            else: # train new model
                model = PPO( "MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate )

            model_path = self.learn_model( model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env*10, save_freq=n_steps_per_env*20, backup_env_file=__file__ )
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
        env = Penalty_Shoot( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True )
        model = PPO.load( args["model_file"], env=env )

        try:
            self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
            self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()


'''
The learning process takes about 5 minutes.
A video with the results can be seen at:
https://imgur.com/a/KvpXS41

State space:
- Composed of all joint positions + torso height
- The number of joint positions is different for robot type 4, so the models are not interchangeable
- For this example, this problem can be avoided by using only the first 22 joints and actuators

Reward:
- The reward for falling is 1, which means that after a while every episode will have a r=1.
- What is the incetive for the robot to fall faster? Discounted return.
  In every state, the algorithm will seek short-term rewards.
- During training, the best model is saved according to the average return, which is almost always 1.
  Therefore, the last model will typically be superior for this example.

Expected evolution of episode length:
    3s|o
      |o
      | o
      |  o
      |   oo
      |     ooooo
  0.4s|          oooooooooooooooo
      |------------------------------> time


This example scales poorly with the number of CPUs because:
- It uses a small rollout buffer (n_steps_per_env * n_envs)
- The simulation workload is light
- For these reasons, the IPC overhead is significant
'''
