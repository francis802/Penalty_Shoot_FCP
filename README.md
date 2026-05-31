# Penalty Shootout with Reinforcement Learning

A RoboCup 3D Soccer Simulation penalty-shootout scenario built on top of the FC Portugal Codebase. A reinforcement-learning-trained striker walks up to the ball and shoots to score, while a goalkeeper dives left or right to defend. Developed as an MSc project in the context of reinforcement learning and autonomous agents.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?logo=cplusplus&logoColor=white)
![License](https://img.shields.io/badge/License-GPLv3-blue)
![RoboCup](https://img.shields.io/badge/RoboCup-3D%20Soccer%20Sim%20League-green)

![](https://s5.gifyu.com/images/Siov6.gif)

## Project: Penalty Shootout

This project implements a one-on-one penalty scenario for the RoboCup 3D simulated NAO robots:

- The **striker** is a reinforcement-learning agent trained with PPO. It learns to walk up to a fixed ball position, align its body, and kick toward the goal. The reward function blends a *walking/approach* phase (rewarding distance reduction to the kick spot, facing the goal, body alignment, and forward movement) with a *kick* phase (rewarding ball speed, velocity directed at the goal, scoring, and aiming for the corners). The two phases are blended by a factor based on the ball's speed, so the agent transitions from approaching to kicking.
- The **goalkeeper** uses a simple reactive policy: it tracks the ball along the goal line and, once the ball crosses a threshold, commits to a left or right dive depending on the ball's lateral position. The dives are hand-authored slot behaviors.

### Approach

- **RL training** is done through OpenAI Gym / Gymnasium environments and Stable Baselines3 (PPO), using the codebase's training utilities and parallel simulation servers.
- **Reward shaping** went through several iterations (for example, comparing distance metrics and experimenting with exponential reward terms) to encourage a clean approach-then-shoot behavior.
- **Goalkeeper dives** are implemented as slot behaviors (sequences of timed joint poses) rather than learned policies.

### Key files

| File | Role |
|---|---|
| `agent/Agent_Penalty.py` | Match agent for the penalty scenario: goalkeeper tracks and dives, kicker beams in and shoots. |
| `behaviors/slot/common/Dive_Left.xml`, `Dive_Right.xml` | Hand-authored goalkeeper dive slot behaviors. |
| `scripts/gyms/penalty_shoot_v1.py` | RL training gym (v1): direct joint-position control, blended approach + kick reward. |
| `scripts/gyms/penalty_shoot_v2.py` | RL training gym (v2): residual control on top of the Step walk primitive. |
| `start_penalty.sh`, `start_penalty_debug.sh` | Launch the penalty scenario (kicker vs. goalkeeper). |

### Running it

Prerequisites (standard RoboCup 3D toolchain):

- [`rcssserver3d`](https://gitlab.com/robocup-sim/SimSpark) — the simulation server.
- [RoboViz](https://github.com/magmaOffenburg/RoboViz) — the monitor/visualizer (optional, for watching the match).
- Python 3 with the codebase dependencies (see the documentation linked below), including Stable Baselines3 and Gymnasium for training.

Start the simulation server (`rcssserver3d`) and, optionally, RoboViz. Then launch the penalty scenario:

```bash
# kicker (uniform 11) vs. goalkeeper (uniform 1)
./start_penalty.sh
# or, with debug drawings enabled:
./start_penalty_debug.sh
```

To train or test the RL striker, run the gym launcher and select one of the `penalty_shoot_*` gyms:

```bash
python3 Run_Utils.py
```

Note: training uses the simulator's "cheat" (ground-truth) data, which must be enabled in the server settings.

## Built on the FC Portugal Codebase

This project is built on top of the **FC Portugal Codebase for the RoboCup 3D Soccer Simulation League**, an open-source research framework by Miguel Abreu, Luis Paulo Reis, and Nuno Lau. The base framework — the agent architecture, skills (walk, dribble, get-up, kick), localization, C++ modules, training infrastructure, and tooling — is **their work, not the author's**. The penalty-shootout scenario described above is the contribution layered on top.

The remainder of this section is preserved from the upstream framework's README.

### About

The FC Portugal Codebase was mainly written in Python, with some C++ modules. It was created to simplify and speed up the development of a team for participating in the RoboCup 3D Soccer Simulation League. We hope this release helps existing teams transition to Python more easily, and provides new teams with a robust and modern foundation upon which they can build new features.

### Documentation

The documentation is available [here](https://docs.google.com/document/d/1aJhwK2iJtU-ri_2JOB8iYvxzbPskJ8kbk_4rb3IK3yc/edit)

### Features

- The team is ready to play!
    - Sample Agent - the active agent attempts to score with a kick, while the others maintain a basic formation
        - Launch team with: **start.sh**
    - Sample Agent supports [Fat Proxy](https://github.com/magmaOffenburg/magmaFatProxy) 
        - Launch team with: **start_fat_proxy.sh**
    - Sample Agent Penalty - a striker performs a basic kick and a goalkeeper dives to defend
        - Launch team with: **start_penalty.sh**
- Skills
    - Get Ups (latest version)
    - Walk (latest version)
    - Dribble v1 (version used in RoboCup 2022)
    - Step (skill-set-primitive used by Walk and Dribble)
    - Basic kick
    - Basic goalkeeper dive
- Features
    - Accurate localization based on probabilistic 6D pose estimation [algorithm](https://doi.org/10.1007/s10846-021-01385-3) and IMU
    - Automatic head orientation
    - Automatic communication with teammates to share location of all visible players and ball
    - Basics: common math ops, server communication, RoboViz drawings (with arrows and preset colors)
    - Behavior manager that internally resets skills when necessary
    - Bundle script to generate a binary and the corresponding start/kill scripts
    - C++ modules are automatically built into shared libraries when changes are detected
    - Central arguments specification for all scripts
    - Custom A* pathfinding implementation in C++, optimized for the soccer environment
    - Easy integration of neural-network-based behaviors
    - Integration with Open AI Gym to train models with reinforcement learning
        - User interface to train, retrain, test & export trained models
        - Common features from Stable Baselines were automated, added evaluation graphs in the terminal
        - Interactive FPS control during model testing, along with logging of statistics
    - Interactive demonstrations, tests and utilities showcasing key features of the team/agents
    - Inverse Kinematics
    - Multiple agents can be launched on a single thread, or one agent per thread
    - Predictor for rolling ball position and velocity
    - Relative/absolute position & orientation of every body part & joint through forward kinematics and vision
    - Sample train environments
    - User-friendly interface to check active arguments and launch utilities & gyms

### Citing the Project

```
@article{abreu2023designing,
  title={Designing a Skilled Soccer Team for RoboCup: Exploring Skill-Set-Primitives through Reinforcement Learning},
  author={Abreu, Miguel and Reis, Luis Paulo and Lau, Nuno},
  journal={arXiv preprint arXiv:2312.14360},
  year={2023}
}
```

## License

This project, like the underlying FC Portugal Codebase, is released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
