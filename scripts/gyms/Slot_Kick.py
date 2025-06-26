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

class Slot_Kick(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw, test_mode=False) -> None:
        self.robot_type = r_type
        self.test_mode = test_mode
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym_Striker", True, enable_draw)
        self.goal_keeper = Agent(ip, server_p, monitor_p, 1, 1, "Gym_Keeper", True, enable_draw)
        self.step_counter = 0
        self.max_episode_steps = 500  # ~10 seconds
        
        # State space (Basic_Run obs + ball info + kick state)
        obs_size = 73  # 65 base + 6 ball info + + 2 goalkeeper position
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_size, -np.inf, np.float32),
            high=np.full(obs_size, np.inf, np.float32),
            dtype=np.float32
        )

        # Action space (20 joints (22 joints - 2 head joints)) + direction influence
        self.no_of_actions = act_size = 21
        MAX = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(
            low=np.full(act_size, -MAX, np.float32),
            high=np.full(act_size, MAX, np.float32),
            dtype=np.float32
        )

        # Initial ball position
        self.ball_pos = np.zeros(3, np.float32)

        self.behavior = self.player.behavior
        self.keeper_behavior = self.goal_keeper.behavior

    def observe(self, init=False):
        r = self.player.world.robot
        w = self.player.world

        # Basic_Run observations (first 65 elements)
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
  

        # Ball observations (65-71)
        ball_rel = w.ball_cheat_abs_pos - r.cheat_abs_pos
        self.obs[65:68] = ball_rel  # relative position
        self.obs[68:71] = w.ball_cheat_abs_vel  # velocity

        # Side to kick (71)
        #self.obs[71] = self.side_to_kick

        # Goalkeeper position
        if self.test_mode:
            self.obs[71] = self.fake_keeper_y
        else:
            self.obs[71] = self.goal_keeper.world.robot.cheat_abs_pos[1] # Keeper y position [-1, 1]

        self.obs[72] = self.obs[66] - self.obs[71]  # ball_y - goalkeeper_y


        return self.obs

    def sync(self):
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())

        r2 = self.goal_keeper.world.robot
        self.goal_keeper.scom.commit_and_send(r2.get_command())
        
        self.goal_keeper.scom.receive()
        self.player.scom.receive()

    def reset(self, seed=None):
        self.step_counter = 0
        r = self.player.world.robot
        
        # Reset robot and ball positions
        #Official ditance for player is 4.8. For training, we use 8.0
        self.player.scom.unofficial_beam((8, 0, r.beam_height), 0)

        if self.test_mode:
            # For testing, set the goalkeeper at a fixed position
            self.fake_keeper_y = np.random.uniform(-1.0, 1.0)
        
        self.goal_keeper.scom.unofficial_beam((-14, 0, r.beam_height), 0)

        self.ball_pos = np.array([9, 0, 0.042])
        self.player.scom.unofficial_move_ball(self.ball_pos)
        
        # Stabilize
        for _ in range(25):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.goal_keeper.behavior.execute("Zero_Bent_Knees_Auto_Head")
            self.sync()

        # Initialize tracking variables
        self.last_ball_dist = np.linalg.norm(self.ball_pos[:2] - r.cheat_abs_pos[:2])
        self.act = np.zeros(self.no_of_actions, np.float32)
        self.in_kick_range = False
        self.initial_velocity = True
        self.keeper_penalty = False
        self.side_to_kick = np.random.choice([-1, 1])  # Randomly choose left or right side to kick

        self.state = 0

        return self.observe(True), {}

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()
        self.goal_keeper.terminate()

    def step(self, action):
        r = self.player.world.robot
        w = self.player.world
        b_pos = w.ball_cheat_abs_pos
        ball_rel = b_pos - r.cheat_abs_pos


        # Exponential moving average for smooth actions
        self.act = 0.4 * self.act + 0.6 * action

        # Calculate ball distance and determine if in kicking range
        ball_dist = np.linalg.norm(ball_rel)

        self.in_kick_range = ball_dist < 0.3  # 30cm threshold for kicking

        # Concatenate [0.0, 0.0] (head influence) with the action array for the kick motion
        kick_action = np.concatenate(([0.0, 0.0], self.act[:20]*2))

        # Choose direction based on last action output (the last element of action array)
        direction = np.tanh(self.act[-1]) * 7.5
        self.behavior.execute("Basic_Kick", direction, kick_action)
        self.goalkeeper_state()  # Update goalkeeper state based on ball position and actions

        self.sync()
        self.step_counter += 1

        # Reward calculation
        reward = self.reward_function()

        # Terminal conditions
        terminal = self.terminal_condition()

        return self.observe(), reward, terminal, False, {}
    
    def goalkeeper_state(self):
        ball_2d = self.goal_keeper.world.ball_cheat_abs_pos
        if self.state == 2: # dive left
            self.state = 4 if self.keeper_behavior.execute("Dive_Left") else 2  # change state to wait after skill has finished
        elif self.state == 3: # dive right
            self.state = 4 if self.keeper_behavior.execute("Dive_Right") else 3 # change state to wait after skill has finished
        elif self.state == 4: # wait (after diving or during opposing kick)
            pass
        elif self.state == 1 or self.keeper_behavior.is_ready("Get_Up"): # if getting up or fallen
            self.state = 0 if self.keeper_behavior.execute("Get_Up") else 1 # return to normal state if get up behavior has finished
        else: # goalkeeper
            y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)
            self.keeper_behavior.execute("Walk", (-14,y_coordinate), True, 0, True, None) # Args: target, is_target_abs, ori, is_ori_abs, distance
            if ball_2d[0] < -10: 
                self.state = 2 if ball_2d[1] > 0 else 3 # dive to defend

    def terminal_condition(self):
        goal_x = 15.0
        r = self.player.world.robot
        w = self.player.world

        # Ball state
        ball_pos = w.ball_cheat_abs_pos
        ball_speed = np.linalg.norm(w.ball_cheat_abs_vel)  # Magnitude

        # Improved terminal conditions
        terminal = (
            ball_pos[0] >= (goal_x + 0.1) or  # CRITICAL: ball crossed goal line
            self.step_counter >= self.max_episode_steps or
            (ball_speed < 0.01 and r.cheat_abs_pos[2] < 0.3) or
            (ball_speed < 0.01 and not self.initial_velocity ) or
            (self.distance_from_keeper() < 0.15)
        )
        return terminal

    def reward_function(self):
        w = self.player.world
        goal_x = 15.0
        goal_y_min, goal_y_max = -1.1, 1.1
        goal_z_max = 0.8

        # Ball state
        ball_pos = w.ball_cheat_abs_pos
        ball_vel = w.ball_cheat_abs_vel

        reward = 0.0

        # 1. Base reward for ball movement
        speed = np.linalg.norm(ball_vel)
        if speed > 0.01 and self.initial_velocity:
            reward += min(max(ball_vel[0], -8), 8.0) # Forward movement
            reward += min(abs(ball_vel[1]) , 0.5) # Lateral movement
            reward += min(abs(ball_vel[2]) / 2, 4.0) # Vertical movement
            self.initial_velocity = False

        # 2. Goal scoring rewards
        in_goal = (
            ball_pos[0] >= (goal_x + 0.1) and
            goal_y_min < ball_pos[1] < goal_y_max and
            ball_pos[2] < goal_z_max
        )
        
        missed = ball_pos[0] >= (goal_x + 0.1) and not in_goal

        keeper_y = self.obs[-1]

        if in_goal:
            ''' Note: Deprecated side to kick logic. Define a continuous y coordinate for the goalkeeper for better results
            if keeper_y == 0.0:
                # Check if the ball is being kicked in the right direction
                if self.side_to_kick * ball_pos[1] > 0:
                    reward_goal = self.exp_field_value(ball_pos[1], ball_pos[2])
                else:
                    reward_goal = self.exp_field_value(ball_pos[1], ball_pos[2], k=-1.0, a=2.0, b=1.0)
                if reward_goal is None:
                    reward_goal = 0.0
                reward += reward_goal
            else:
            '''
            reward_goal = self.exp_field_value(ball_pos[1], ball_pos[2], y0=keeper_y, k=2.0, b=-1.0)
            if reward_goal is None:
                reward_goal = 0.0
            reward += reward_goal

        elif missed:
            reward = -10.0  # Penalty for missing the goal
        elif speed < 0.01 and not self.initial_velocity:
            reward -= 2.0  # Penalty for not scoring
        
        if self.distance_from_keeper() < 0.15 and not self.keeper_penalty:
            print("Keeper penalty triggered")
            reward -= 5.0
            self.keeper_penalty = True

        return reward
    
    def exp_field_value(self, y, z, y0=0.0, z0=0.0, k=3.0, a=1.0, b=0.0, is_euclidean=True):
        """
        Query the exponential field value at a given (y, z) coordinate.

        Parameters:
            k: Exponential growth rate
            a: Scaling factor
            b: Offset value
        
        Returns:
            float: The field value if (x, y) is within the rectangle.
            None: If the point is outside the rectangle.
        """
        y_min, y_max = -1.1, 1.1
        z_min, z_max = 0.0, 0.8

        if y_min <= y <= y_max and z_min <= z <= z_max:
            distance = 0
            if is_euclidean:
                distance = np.sqrt((y-y0)**2 + (z-z0)**2)
            else:
                distance = abs(y-y0) + abs(z-z0)
            value = a * (np.exp(k * distance) - 1 + b)
            return value
        else:
            return None
    
    def distance_from_keeper(self):
        """
        Calculate the distance from the goalkeeper to the ball.
        """
        keeper_pos = np.array(self.goal_keeper.world.robot.cheat_abs_pos)
        keeper_pos[0] = -keeper_pos[0]  # Flip x coordinate for goalkeeper
        keeper_pos[1] = -keeper_pos[1]  # Flip y coordinate for goalkeeper
        ball_pos = self.player.world.ball_cheat_abs_pos
        return np.linalg.norm(ball_pos[:2] - keeper_pos[:2])

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
        folder_name = f'Slot_Kick_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        #--------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return Slot_Kick( self.ip , self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False )
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
        env = Slot_Kick( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True, test_mode=True )
        model = PPO.load( args["model_file"], env=env )

        try:
            self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
            self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
