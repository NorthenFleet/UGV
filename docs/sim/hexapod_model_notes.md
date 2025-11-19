# 六足模型说明

关节命名：`leg_{0..5}/{coxa,femur,tibia}`，统一顺序与旋转轴定义。
坐标系：
- 机体 base_link：前 x、左 y、上 z
- 每条腿局部坐标：以髋关节为原点，依次沿关节轴定义

URDF 约定：
- 惯性与质量：合理设置，避免数值不稳定
- 限位：来自 `hexapod_joint_limits.yaml`
- 网格缩放/姿态：与 URDF link 对齐