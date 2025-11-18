# 数据流与模块边界

训练侧：
- Env.reset → 生成初始状态（随机相位/姿态/地形）
- Actor.act(obs) → 动作（关节目标/力矩）
- Env.step(action) → 返回 obs, reward, done, info
- Rollout 收集 → 优势计算 → PPO/SAC 更新 → 评估 → checkpoint

运行侧：
- 传感器（IMU/编码器）→ StateEstimator 融合
- PolicyInference 前向 → 产生动作
- CommandSender 封装帧（CRC）→ UART/ROS2 下发
- STM32 执行 → 反馈状态帧 → Jetson 记录与监控

ROS2 桥接（可选）：
- /hexapod/state：发布 IMU/关节/接触
- /hexapod/cmd：订阅控制命令（目标关节位置/足端期望）