# 起步总览

从仿真环境与接口稳定性入手，先打通“URDF → PyBullet 环境（Gym 接口）→ PPO 训练”的最小可跑闭环；同时抽象仿真后端，保留 MuJoCo 接入可能。

## 阶段 1：仿真环境与后端抽象

* 目标：实现可 reset/step 的环境，统一观察/动作/奖励与终止逻辑

* 实现要点：

  * 定义 `ISimBackend` 接口：`load_model(config) / reset(seed) / step(action) / get_obs()`

  * 实现 `PyBulletBackend`：加载 `hexapod.urdf`，建立地面与基本障碍，维护关节与接触信息

  * 将 `hexapod_walk_env.py` 改为 Gym 风格：`reset()` 返回 obs，`step(action)` 返回 `(obs, reward, done, info)`

  * 配置加载：`sim_params.yaml`（重力/步长/摩擦），`hexapod_joint_limits.yaml`（关节限位）

  * 域随机化钩子：摩擦/质量/传感噪声（先最简，预留接口）

## 阶段 2：URDF 与运动学基础

* 目标：稳定的关节命名与坐标系，基础 FK/IK 支撑观测与动作映射

* 实现要点：

  * 统一关节命名：`leg_{0..5}/{coxa,femur,tibia}`，对齐 `hexapod_model_notes.md`

  * 在 `common/kinematics` 完成 `leg_fk.py/leg_ik.py` 的函数签名与基本实现（先 FK，IK 可返回占位或数值法）

  * 在环境中将动作从归一化映射到关节目标（限位/速度限幅）

  * 观测拼接：IMU（仿真获取）、关节角/速、足端相对位置与接触标志

## 阶段 3：PPO 训练最小闭环

* 目标：能在平地上获得前进奖励并稳定行走

* 实现要点：

  * `rl/policies/actor_critic.py`：MLP Actor-Critic，观测归一化

  * `rl/algorithms/ppo_agent.py`：GAE、剪切损失、熵与价值损失

  * `rl/training/train_walk_ppo.py`：并行环境采样（先单环境），日志/评估/保存 checkpoint

  * 奖励：速度跟踪 + 姿态稳定 + 能耗与平滑惩罚（按文档）

## 阶段 4：脚本与日志

* 目标：可一键运行与基本可视化

* 实现要点：

  * `scripts/run_sim_demo.sh`：加载环境，键盘/脚本控制演示

  * `rl/logs/` 与 TensorBoard：训练曲线记录

  * `tools/`：简单可视化（IMU/足端）占位

## 阶段 5（可并行）：Jetson/RPi 推理骨架

* 目标：运行端最小主循环占位，加载模型、限幅与心跳

* 实现要点：

  * `jetson/apps/hexapod_brain/main.py`：读取 `runtime_params.yaml`，循环订阅状态（仿真/ROS2），前向推理，发送命令（UART/ROS2 可占位）

  * `policy_inference.py`：PyTorch/ONNX 加载，前向接口

  * `command_sender.py`：安全限幅与帧封装（按通信协议文档）

## 验证与交付

* 单元测试：`common/kinematics` 与奖励函数

* 仿真 Smoke Test：reset → 随机动作 → step 运行稳定，观测维度与范围检查

* 训练 Smoke Test：跑通 1\~2 小时，奖励曲线增长与策略评估通过

## 代码入口与文件

* `sim/pybullet_env/envs/hexapod_walk_env.py`：Gym 接口环境主类

* `sim/pybullet_env/config/*.yaml`：仿真参数与限位

* `common/kinematics/*.py`：FK/IK 与姿态变换

* `rl/algorithms/ppo_agent.py`、`rl/policies/actor_critic.py`、`rl/training/train_walk_ppo.py`

* `jetson/apps/hexapod_brain/*.py`：运行端主循环与推理/发送

## 里程碑与输出

* M1：环境 + 后端抽象 + 最小奖励/终止（可跑）

* M2：PPO 训练闭环 + 日志/checkpoint + 简单评估

* M3：Jetson/RPi 推理骨架 + 通信帧封装占位

## 选择说明

* 先用 PyBullet 保证开发效率与可视化；接口抽象保证未来切换到 MuJoCo 不影响 RL 与控制代码。

