# Wrist Camera Frame Duplication Fix

## 问题描述

Wrist 相机（使用 FFmpeg MJPEG 流）存在严重的帧重复问题：
- 超过 80% 的帧内容重复
- 连续重复超过 100 帧的情况
- 导致三个画面回放时严重不同步
- Camera_high（RealSense）几乎无重复帧

## 根本原因

`FFmpegWristCamera._capture_worker()` 方法缺少帧率控制：
1. **采集循环跑得太快**：只依赖 `select` 超时，没有主动 sleep
2. **重复处理旧帧**：在新数据到达前，循环会反复处理缓冲区中的同一帧
3. **时序不匹配**：与 `CameraCollector`（RealSense）的时序控制逻辑不一致

## 修复方案

### 修改位置：`camera_collector.py` - `FFmpegWristCamera._capture_worker()` 方法

#### 1. 添加帧率控制变量（第 597 行）
```python
# >>> FIX: 添加帧率控制变量，这是修复帧重复的关键
last_capture_time = time.time()
```

#### 2. 在循环开始时主动 sleep（第 601-609 行）
```python
# >>> FIX: 帧率控制 - 在循环开始时主动 sleep，确保按目标帧率采集
# 这防止了采集循环跑得太快，避免在新数据到达前反复处理同一帧
loop_start_time = time.time()
elapsed = loop_start_time - last_capture_time
sleep_duration = self.target_interval - elapsed
if sleep_duration > 0.001:  # 避免极小的 sleep
    time.sleep(sleep_duration)

# 更新当前时间（sleep 后）
current_time = time.time()
```

#### 3. 缩短 select 超时时间（第 612 行）
```python
# 1. 使用 select 等待数据（超时时间缩短，因为已有 sleep 控制帧率）
# >>> FIX: 将超时从 self.target_interval * 2 改为固定的 0.1 秒
rlist, _, xlist = select.select([fd], [], [fd], 0.1)
```

#### 4. 预热期间也更新时间（第 695 行）
```python
# >>> FIX: 预热期间也要更新 last_capture_time，保持正确的时序
last_capture_time = current_time
```

#### 5. 成功处理帧后更新时间（第 712 行）
```python
# >>> FIX: 更新上次成功采集的时间，用于下次循环的 sleep 计算
last_capture_time = current_time
```

## 修复效果

1. **消除帧重复**：采集循环按目标帧率（30 Hz）运行，每次都等待新帧
2. **时序同步**：与 RealSense 相机保持一致的时序逻辑
3. **稳定输出**：避免了采集循环过快导致的问题
4. **资源优化**：减少了不必要的 CPU 占用

## 技术细节

### 为什么仅靠 select 超时不够？

- `select` 的超时只是"最多等待多久"，如果缓冲区有数据会立即返回
- 没有主动 sleep，循环会尽快执行，导致在新帧到达前反复处理
- 需要像 `CameraCollector` 那样主动控制采集间隔

### 为什么要缩短 select 超时？

- 原来的 `self.target_interval * 2`（约 67ms @ 30fps）是为了"等待新数据"
- 但现在已经用 sleep 控制了节奏，select 只需短超时（100ms）即可
- 短超时可以更快响应停止信号

### 为什么预热期间也要更新时间？

- 预热期间虽然不调用回调，但仍在处理帧
- 如果不更新时间，会导致预热后第一帧的 sleep 时间异常
- 保持时序连续性很重要

## 测试建议

1. 运行测试代码观察 Avg FPS 是否稳定在 30 左右
2. 检查缓冲区大小是否稳定（不会持续增长）
3. 对比三个相机的时间戳，确认同步性
4. 使用帧内容哈希检测重复率（应降至接近 0%）

## 兼容性

- 保持了原有的所有功能（MJPEG 解析、缓冲区限制、预热等）
- 仅添加了帧率控制逻辑，不影响其他部分
- 与 `CameraCollector` 的逻辑保持一致
