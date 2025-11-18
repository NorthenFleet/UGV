# 总体说明
选择 PyBullet 作为首选仿真引擎，原因：
- 原生支持 URDF，Python API 完整，易于快速搭建与调试
- 在 macOS 上安装与可视化友好，学习成本低
- 接口稳定，易与 ROS2、RL 框架集成
同时保留仿真后端抽象，未来可平滑接入 MuJoCo（速度与接触精度更高）。

# 顶层目录
hexapod_robot/
- docs/: 文档
- sim/: 仿真（PyBullet/MuJoCo）
- rl/: 强化学习
- ros2_ws/: ROS2 工作空间
- jetson/: Jetson/Raspberry Pi 运行端
- control/: STM32 底层控制
- common/: 公共模块（协议、配置、工具）
- scripts/: 一键脚本
- tools/: 调试工具

# 系统架构
- 仿真端（Mac）：PyBullet 环境→状态生成→RL 采样训练→策略导出
- 运行端（Jetson/RPi）：订阅状态→策略推理→下发控制（UART/ROS2）
- 控制端（STM32）：PWM/舵机驱动→状态反馈→通信回传
- ROS2（可选）：仿真/实机状态话题桥接、Bringup 与工具链
- 公共模块：消息帧、编解码、运动学、配置与日志

# 数据流
- 仿真：Env.step(action)→observations/reward/done/info
- 训练：采样→优势计算→策略更新→评估→checkpoint
- 运行：传感器/状态→估计→策略前向→指令封装→UART/ROS2 下发
- 回环：控制反馈→安全检测→异常上报→日志保存

# 仿真设计（PyBullet 首选）
## 模型与资源
- URDF：`hexapod.urdf`，关节命名统一（如 leg_i/{coxa,femur,tibia}）
- 网格：`meshes/`（STL/OBJ），与 URDF 绑定
- 地面与障碍：`ground_plane.urdf`、台阶/随机地形生成脚本

## 物理与时序
- 时间步长：`Δt=2ms`（仿真）→ 控制周期 `50~100Hz`
- 重力/摩擦/接触参数：`sim_params.yaml`
- 控制模式：位置 PD（训练稳定），后期支持力矩控制

## 环境接口
- 观察空间（示例）：
  - IMU：姿态四元数/欧拉角（3~4）
  - 速度：线速度/角速度（6）
  - 关节：角度/速度（18/18）
  - 足端：相对足端位置（18），接触标志（6）
  - 障碍信息（可选）：高度图/距离（N）
- 动作空间：
  - 关节目标位置（18）或力矩（18），归一化到 [-1,1]
- 重置逻辑：姿态倾覆、身体高度异常、位置回到起点、随机初始相位
- 域随机化：摩擦、质量、关节阻尼、传感噪声、地形扰动

## 奖励设计（示例）
- 前进奖励：`w_v * 目标速度跟踪`
- 稳定性：`w_ori * 姿态偏差惩罚`
- 能耗：`w_tau * 力矩/速度惩罚`
- 平滑性：`w_smooth * 动作变化惩罚`
- 足端接触：`w_contact * 非法滑移/跳跃惩罚`
- 安全终止：倾覆/过载/离地异常 → 终止惩罚

# RL 训练设计
## 算法
- 首选 PPO（稳定鲁棒、实现成熟），可选 SAC（连续控制、探索友好）
- 采样并行：多环境并行 `N_env=8~32`

## 网络结构
- Actor-Critic（MLP）基础，选配 LSTM（步态相位记忆）或 Transformer（长时依赖）
- 归一化：观测标准化、优势标准化

## 训练流程
- 采样 rollout → GAE 优势计算 → 更新多轮 → 评估/日志 → 保存 checkpoint
- 课程学习：平地行走→轻微坡度→多障碍→不规则地形

## 监控与复现实验
- TensorBoard、CSV 日志，固定随机种子，保存 `best` 与阶段性 `ckpt`

# Jetson/Raspberry Pi 运行端
## 节点职责
- `main.py`：主循环，订阅状态→策略推理→下发控制
- `state_estimator.py`：融合 IMU/编码器，估计姿态与速度
- `policy_inference.py`：加载 PyTorch 模型，做前向（支持 ONNX/TensorRT）
- `command_sender.py`：UART/ROS2 指令封装与发送，含安全限幅
- `perception_node.py`（可选）：相机/LiDAR 简化处理

## 运行配置
- 频率：`control_frequency_hz=100`
- 安全参数：最大倾角、最大关节速度、指令变化限幅
- 模型路径：不同任务模型切换（走直、转弯、导航）

# STM32 底层控制
- 模块：PWM/舵机驱动、UART 通信、任务调度、步态插值（备选）
- 协议：固定帧头/长度/类型/CRC16；命令帧（关节目标/模式）、状态帧（关节反馈/IMU）
- 本地安全：超限保护、通信丢包保形、急停通道

# ROS2 工作空间（可选但推荐）
- 包：`hexapod_description`（URDF/Xacro）、`hexapod_bringup`（launch）、`hexapod_control`（控制节点）、`hexapod_rl_bridge`（仿真桥接）
- 话题（示例）：
  - `/hexapod/state`（IMU/关节/接触）
  - `/hexapod/cmd`（关节目标/期望足端）
- QoS：传感器数据 `best_effort`，控制命令 `reliable`

# 公共模块
- `messaging/`：帧定义、编解码、CRC
- `kinematics/`：单腿 IK/FK、身体位姿变换
- `config/`：连杆长度、坐标系、仿真/运行参数
- `utils/`：日志、数值工具、可视化

# 一键脚本与工具
- `setup_env.sh`：创建 Conda/venv、安装依赖（PyBullet/ROS2/PyTorch）
- `run_sim_demo.sh`：启动仿真演示，键盘/脚本控制
- `train_walk_ppo.sh`：一键训练流水线
- `sync_model_to_jetson.sh`：模型打包/同步到运行端
- `flash_stm32.sh`：固件烧录
- 工具：关节零位标定 GUI、IMU 可视化、日志回放

# 仿真后端抽象（便于切换到 MuJoCo）
- 定义接口 `ISimBackend`：`load_model() / reset() / step() / get_obs() / apply_action()`
- 实现 `PyBulletBackend`（默认）与 `MuJoCoBackend`（可选）
- Env 只依赖接口，切换后端无需改训练逻辑

# 配置与参数管理
- YAML：仿真参数、关节极限、课程学习阶段、运行端安全参数
- 统一 `Config` 读取与校验，支持命令行覆盖

# 测试与验证
- 单元测试：运动学、消息编解码、奖励函数
- 仿真测试：行走直线、转弯、坡地稳态
- HIL（硬件在环）：Jetson 运行端连接仿真，验证闭环
- 实机 Bringup：安全检查清单与回退策略

# 里程碑
- 里程碑 1：PyBullet 环境与 PPO 基线可跑，平地行走
- 里程碑 2：ROS2 桥接与工具链完善（仿真→ROS2）
- 里程碑 3：Jetson/RPi 运行端策略推理与安全控制
- 里程碑 4：STM32 联调（通信协议打通、基础步态）
- 里程碑 5：不规则地形任务与奖励优化
- 里程碑 6：MuJoCo 后端接入与加速训练（可选）

# 交付物
- 完整目录与文档骨架
- PyBullet 环境类与 Gym 接口适配
- PPO 训练脚本与配置示例
- Jetson/RPi 运行端主循环与配置
- 通信协议文档与编码解码模块
- ROS2 包与基础 launch
- 工具与脚本可运行占位
