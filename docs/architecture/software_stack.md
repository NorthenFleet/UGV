# 软件栈

仿真与训练端：
- Python 3.10+：主开发语言
- PyBullet：默认仿真后端（URDF、接触、可视化）
- MuJoCo（可选）：高性能仿真后端（通过抽象接口接入）
- PyTorch：策略网络训练与推理（可导出 ONNX/TensorRT）
- Gym 接口：统一 `reset/step` 交互，便于算法复用

运行端（Jetson/RPi）：
- Python：主循环与策略推理
- ROS2（可选）：话题通信与工具链
- UART：与 STM32 可靠通信（CRC）
- Docker（可选）：部署封装与依赖管理

控制端（STM32）：
- STM32 HAL：外设驱动（UART、定时器、PWM）
- FreeRTOS（可选）：任务调度与优先级管理