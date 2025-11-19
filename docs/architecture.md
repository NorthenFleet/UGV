# 系统架构设计

## 总体架构
- 控制与仿真（Python）：支持模式1 PyBullet/MuJoCo 仿真；模式2 Jetson 读取编码器/IMU。
- 显示与交互（UE）：使用 FBX 六足骨骼模型显示与交互。
- 通信层：UDP（优先，低延迟）或 TCP；建议先用 UDP+JSON 调通，再切换二进制协议。

数据流：`[PyBullet 或 Jetson] -> joint_state -> [UE 骨骼显示]`；后续可加 UE→Python 的反向控制通道。

## 统一状态数据结构
```
{
  "src": "sim" | "real",
  "timestamp": 1718000000.123,
  "base": { "pos": [x,y,z], "ori": [qx,qy,qz,qw] },
  "joints": {
    "order": [
      "LF_HIP","LF_THIGH","LF_KNEE",
      "LM_HIP","LM_THIGH","LM_KNEE",
      "LR_HIP","LR_THIGH","LR_KNEE",
      "RF_HIP","RF_THIGH","RF_KNEE",
      "RM_HIP","RM_THIGH","RM_KNEE",
      "RR_HIP","RR_THIGH","RR_KNEE"
    ],
    "angles": [18 个关节角，rad]
  },
  "extra": { "contact": [6 布尔], "battery": 12.3, "mode": "walk" }
}
```
- 关节命名需与 URDF/FBX 骨骼一致，UE 按 `order` 依次绑定。

## 通信协议
- 简版 UDP+JSON：端口如 `50051`，仿真 30Hz，实机 20~50Hz。
- Python 端每帧发送 JSON 文本；UE 端用 Socket 插件或 C++ UObject 非阻塞接收并解析。

## UE 端接收与驱动（思路）
- C++/蓝图：非阻塞 `recvfrom` → JSON 解析 → 对 SkeletalMesh 逐骨骼设置旋转 → 更新 Actor 位姿。
- 蓝图可以用“Set Bone by Name/Transform (Modify) Bone”等节点驱动。

## 三端协同模式
- 模式1 仿真驱动：Python 仿真计算并通过 UDP 推送状态；UE 显示与交互。
- 模式2 实机驱动：Jetson 读取编码器/IMU，以同一 JSON 格式推送；可同步到仿真形成数字孪生。
- 模式3 UE 控制：新增反向通道，UE 发送控制 JSON（如 `set_gait`、`set_speed`、`move`），Python 侧更新参数。

## 项目结构（重构后）
- `src/hexapod_robot/`
  - `sim/pybullet_env/`：后端、环境、资产与脚本
  - `sim/ue_bridge/`：通信桥接（`udp_sender.py`）
  - `sim/ue_backend.py`：UE 后端骨架
  - `rl/`：算法、策略、配置
  - 其余模块：`common/`、`jetson/`、`control/`、`ros2_ws/`、`tools/`
- `src/training/`：训练与评估脚本、实验日志与模型在 `training/` 下
- `docs/`：设计文档与说明

## 快速上手
- 安装依赖：`pip install -r requirements.txt`
- 运行仿真演示：
  - `PYTHONPATH="$(pwd)/src:$(pwd)" HEXAPOD_GUI=1 python -m hexapod_robot.sim.pybullet_env.scripts.visualize_hexapod`
- 推送到 UE（UDP）：
  - `UE_UDP_IP=192.168.1.100 UE_UDP_PORT=50051 PYTHONPATH="$(pwd)/src:$(pwd)" python -m hexapod_robot.sim.pybullet_env.scripts.visualize_hexapod`