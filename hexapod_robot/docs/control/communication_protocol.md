# Jetson ↔ STM32 通信协议

帧格式（小端）：
- Header：`0xAA 0x55`
- Length：payload 字节数（含 MsgID，不含 Header/Length/CRC）
- MsgID：消息类型（命令/状态/心跳）
- Payload：数据内容
- CRC16：多项式 `0x1021`，初值 `0xFFFF`

消息类型：
- 0x01 控制命令：关节目标位置（float32 ×18）、控制模式、时间戳
- 0x02 状态回传：关节角度/速度（float32 ×18）、IMU 姿态、错误码
- 0x03 心跳：计数器与时间戳

健壮性：
- 序列号与时间戳比对，丢包重发策略
- 超时/校验失败计数，进入保形/急停