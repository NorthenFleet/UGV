# Jetson/Raspberry Pi 环境配置

系统与依赖：
- Python3、pip/venv 或 Conda
- 可选 Docker：封装运行环境
- ROS2（可选）：desktop 版
- PyTorch（按设备架构选择）、numpy 等

配置：
- 串口：`/dev/ttyUSB0`、波特率 `115200`
- 控制频率：`100Hz`
- 模型路径与任务切换配置

部署步骤：
- 安装依赖→拉取代码→配置 `configs/*.yaml`→运行 `apps/hexapod_brain/main.py`