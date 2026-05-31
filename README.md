# Penalty Shootout with Reinforcement Learning

A RoboCup 3D Soccer Simulation penalty-shootout scenario built on top of the FC Portugal Codebase. A reinforcement-learning-trained striker walks up to the ball and shoots to score, while a goalkeeper dives left or right to defend. Developed as an MSc project in the context of reinforcement learning and autonomous agents.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?logo=cplusplus&logoColor=white)
![License](https://img.shields.io/badge/License-GPLv3-blue)
![RoboCup](https://img.shields.io/badge/RoboCup-3D%20Soccer%20Sim%20League-green)

![](https://s5.gifyu.com/images/Siov6.gif)

## Project: Penalty Shootout

This project implements a one-on-one penalty scenario for the RoboCup 3D simulated NAO robots: a **striker** that learns to walk up to the ball and kick it into the goal, and a **goalkeeper** that reacts to the incoming ball and dives to defend.

### The striker (reinforcement learning)

The striker is trained with **PPO** (Stable Baselines3) inside a custom **Gymnasium** environment. It learns to walk from its starting position to a fixed ball, align its body for a kick, and shoot toward the goal.

The reward is a **two-phase, blended** signal. A blending factor `alpha` is derived from the ball's speed, so the agent smoothly transitions from the approach phase (ball still) to the kick phase (ball moving):

- **Approach / walk phase** rewards:
  - reducing the distance to the kick spot (computed via the codebase path manager),
  - facing the goal (torso orientation toward the goal direction),
  - body-to-ball alignment (the ball sitting in the correct position relative to the torso for a kick),
  - forward motion (head/body speed).
- **Kick phase** rewards:
  - ball speed,
  - velocity directed at the goal centre,
  - a bonus for scoring,
  - a bonus for aiming at the corners,
  - a penalty for sending the ball out of bounds.

An episode ends on a **goal**, the ball going **out of bounds**, the robot **falling** (and the ball stopping), or a **timeout** (step limit). Training relies on the simulator's ground-truth ("cheat") data and runs across **parallel servers** for throughput.

Two gym variants were explored:

- **v1 — direct joint control.** The policy outputs target positions for every joint directly. Observation: joint positions, torso height, robot position, distance to ball, and ball velocity. This is the version that carries the full two-phase approach-and-kick reward.
- **v2 — residual control on the Step primitive.** The policy outputs **residuals** that are added on top of the targets produced by the codebase's `Step` walk primitive, with the underlying step parameters (swing height, leg extension) also nudged by the policy. Its richer observation captures the walk state (step duration, progress, active leg) alongside IMU, foot force, and joint data. In its current form v2 optimises a stable, fast walk-up (forward displacement); the kick-phase reward terms are present in the file but disabled.

Reward shaping went through several iterations (for example, comparing distance metrics and experimenting with exponential reward terms) to encourage a clean approach-then-shoot behaviour.

### The goalkeeper (reactive state machine)

The goalkeeper is **not learned**. It is a small reactive state machine (in `agent/Agent_Penalty.py`):

1. It walks along the goal line, tracking the ball's lateral (`y`) position (clamped to the goal width).
2. Once the ball crosses a threshold along the field (`x < -10`), it **commits to a dive**: left if the ball is on the positive-`y` side, right otherwise.
3. The dives themselves are **hand-authored slot behaviors** — `Dive_Left.xml` and `Dive_Right.xml` — sequences of timed joint poses rather than a learned policy. If the keeper falls, it runs the framework's `Get_Up` behavior.

> Note: the scenario launched by `start_penalty.sh` pairs the reactive goalkeeper with a striker that uses the framework's scripted `Basic_Kick` (aimed at a random corner) for a self-contained, ready-to-watch match. The **PPO-trained striker** described above is trained, tested, and exported through the gym environments.

## Repository structure

This repository is the **FC Portugal Codebase** (an existing research framework, see attribution below) with a penalty-shootout scenario layered on top. The files authored for **this project** are:

| Path | Role |
|---|---|
| `agent/Agent_Penalty.py` | Penalty match agent: reactive goalkeeper (track + dive) and scripted kicker that beam in and play the shootout. |
| `scripts/gyms/penalty_shoot_v1.py` | RL training gym (v1): direct joint-position control with the blended approach + kick reward. |
| `scripts/gyms/penalty_shoot_v2.py` | RL training gym (v2): residual control on top of the `Step` walk primitive. |
| `behaviors/slot/common/Dive_Left.xml`, `behaviors/slot/common/Dive_Right.xml` | Hand-authored goalkeeper dive slot behaviors. |
| `start_penalty.sh`, `start_penalty_debug.sh` | Launch the penalty scenario (kicker vs. goalkeeper; the `_debug` variant enables RoboViz drawings). |

Everything else is the **inherited FC Portugal framework** and is **not the author's work**, including (non-exhaustively):

- `agent/Base_Agent.py`, `agent/Agent.py` — base agent architecture and the default sample agent.
- `behaviors/` — the skill set (Walk, Dribble, Get-Up, Step, Basic_Kick) and the slot/custom behavior machinery.
- `world/`, `communication/`, `math_ops/` — world model and localization, server/monitor communication, and math utilities.
- `cpp/` — C++ modules (localization, A* pathfinding, etc.) built automatically into shared libraries.
- `scripts/` (commons, utils, and the other sample gyms such as `Basic_Run.py`, `Fall.py`, `Get_Up.py`), and `Run_*.py` — the training/utility launchers and runtime entry points.
- The remaining `start_*.sh` / `kill.sh` scripts and the `bundle/` tooling.

## How to run

### Prerequisites

Standard RoboCup 3D toolchain:

- [`rcssserver3d`](https://gitlab.com/robocup-sim/SimSpark) — the simulation server.
- [RoboViz](https://github.com/magmaOffenburg/RoboViz) — the monitor/visualizer (optional, for watching the match or training).
- Python 3 with the codebase dependencies (see the [documentation](https://docs.google.com/document/d/1aJhwK2iJtU-ri_2JOB8iYvxzbPskJ8kbk_4rb3IK3yc/edit) linked below), including **Stable Baselines3** and **Gymnasium** for training.

### Watch the penalty scenario

Start the simulation server (`rcssserver3d`) and, optionally, RoboViz. Then launch the scenario (kicker on uniform 11 vs. goalkeeper on uniform 1):

```bash
./start_penalty.sh
# or, with RoboViz debug drawings enabled:
./start_penalty_debug.sh
```

### Train or test the RL striker

The RL striker is driven through the codebase's gym launcher:

```bash
python3 Run_Utils.py
```

From the interactive menu, pick the **Gyms** option and select one of the `penalty_shoot` gyms (`penalty_shoot_v1` or `penalty_shoot_v2`), then choose to **train**, **retrain**, or **test** a model. Testing also exports the trained policy to a `.pkl` so it can be wired in as a custom behavior.

> Training requires the simulator's **"cheat" (ground-truth) data** to be enabled — in `Run_Utils.py`, under `Server -> Cheats`. The gyms assert this and will fail fast if it is off. Training spins up several parallel servers automatically.

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
