#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机数据采集模块
负责一个Realsense相机(camera_high)和两个手腕相机(wrist cameras)的图像采集
修复了wrist相机帧重复问题：使用FFmpeg MJPEG流 + 帧边界搜索 + 主动清空缓冲区 + 预热 + 缓冲区限制策略 + 帧率控制
"""

import cv2
import numpy as np
import threading
import time
import subprocess
import os
import select
import fcntl
import sys
from typing import Optional, Callable, Dict, Any, Tuple
from time_sync import get_timestamp


# ================================================================
#               CameraCollector (RealSense - 无需改动)
# ================================================================
class CameraCollector:
    """Realsense相机数据采集器"""
    
    def __init__(self, camera_id: str, device_index: int = 0, 
                 resolution: tuple = (640, 480), fps: int = 30, serial_number: str = None):
        """
        初始化相机采集器
        
        Args:
            camera_id: 相机标识符 (如 "camera_high")
            device_index: 设备索引 (当不使用serial_number时)
            resolution: 分辨率 (width, height)
            fps: 帧率
            serial_number: RealSense设备序列号 (优先使用)
        """
        self.camera_id = camera_id
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.serial_number = serial_number
        self.target_interval = 1.0 / fps # RealSense 采集器仍然使用这个
        
        self._stop_flag = False
        self._capture_thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._use_realsense = False
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self._color_cap: Optional[cv2.VideoCapture] = None
        self._depth_cap: Optional[cv2.VideoCapture] = None # OpenCV 无法获取真实深度
        
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0.0

        # 数据缓存
        self.last_color_img: Optional[np.ndarray] = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.last_depth_img: Optional[np.ndarray] = np.zeros((resolution[1], resolution[0]), dtype=np.uint16) # 深度图类型
        self.last_timestamp: float = 0.0
        self.data_lock = threading.Lock()
        
        if serial_number:
            print(f"相机采集器初始化: {camera_id} (RealSense序列号: {serial_number})")
        else:
            print(f"相机采集器初始化: {camera_id} (设备{device_index})")
    
    def set_data_callback(self, callback: Callable[[str, np.ndarray, np.ndarray, float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def get_latest_data(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """获取最新的数据"""
        with self.data_lock:
            # 返回数据的副本以保证线程安全
            return self.last_timestamp, self.last_color_img.copy(), self.last_depth_img.copy()

    def connect(self) -> bool:
        """尝试连接 RealSense，如果失败则尝试 OpenCV"""
        try:
            import pyrealsense2 as rs # 尝试导入 RealSense SDK
            print(f"INFO: {self.camera_id}: 找到 RealSense SDK，尝试连接...")
            if self._connect_realsense():
                return True
            else:
                 print(f"WARN: {self.camera_id}: RealSense 连接失败，尝试 OpenCV 回退...")
                 return self._connect_opencv()
        except ImportError:
            print(f"WARN: {self.camera_id}: 未找到 RealSense SDK (pyrealsense2)，使用 OpenCV 连接")
            return self._connect_opencv()
        except Exception as e:
            print(f"ERROR: {self.camera_id}: 连接 RealSense 时发生意外错误 ({e})，尝试 OpenCV 回退...")
            return self._connect_opencv()
    
    def _connect_realsense(self) -> bool:
        """使用 RealSense SDK 连接相机"""
        try:
            import pyrealsense2 as rs
            
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 如果指定了序列号，则只连接该设备
            if self.serial_number:
                try:
                    config.enable_device(self.serial_number)
                    print(f"INFO: {self.camera_id}: 请求连接 RealSense 设备 {self.serial_number}")
                except RuntimeError as e:
                     print(f"ERROR: {self.camera_id}: 无法找到或启用序列号为 {self.serial_number} 的设备: {e}")
                     return False
            else:
                 # 如果未指定序列号，尝试连接第一个找到的设备（可能不稳定）
                 print(f"WARN: {self.camera_id}: 未指定 RealSense 序列号，将尝试连接第一个找到的设备。")
            
            # 配置颜色和深度流
            config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
            
            # 启动管道
            profile = self.pipeline.start(config)
            
            # 优化：增加 RealSense 内部缓冲区，给 Python 更多处理时间
            try:
                sensor = profile.get_device().first_color_sensor()
                if sensor:
                     # 设置为2，保留最新的两帧，防止因 Python 处理稍慢而丢帧
                     sensor.set_option(rs.option.frames_queue_size, 2)
                     print(f"INFO: {self.camera_id}: RealSense 颜色传感器帧队列大小设为 2")
            except Exception as e_opt:
                 print(f"WARN: {self.camera_id}: 设置 RealSense 帧队列大小时出错 (忽略): {e_opt}")

            self._is_connected = True
            self._use_realsense = True
            
            print(f"✅ {self.camera_id}: Realsense 连接成功")
            return True
        except Exception as e:
            print(f"❌ {self.camera_id}: Realsense 连接失败: {e}")
            # 尝试清理 pipeline
            if hasattr(self, 'pipeline'):
                 try: self.pipeline.stop()
                 except: pass
            return False
    
    def _connect_opencv(self) -> bool:
        """使用 OpenCV 连接相机（作为备选方案）"""
        try:
            self._color_cap = cv2.VideoCapture(self.device_index)
            if not self._color_cap.isOpened():
                print(f"❌ {self.camera_id}: OpenCV 无法打开设备 {self.device_index}")
                return False
            
            # 设置分辨率和帧率 (注意：OpenCV 对这些设置的支持可能有限)
            self._color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._color_cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 尝试设置 OpenCV 缓冲区大小为1，减少延迟（效果取决于后端）
            self._color_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 读取实际设置值
            actual_width = int(self._color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._color_cap.get(cv2.CAP_PROP_FPS)
            buffer_size = self._color_cap.get(cv2.CAP_PROP_BUFFERSIZE)

            print(f"✅ {self.camera_id}: OpenCV 连接成功 "
                  f"({actual_width}x{actual_height} @ {actual_fps:.1f} FPS, Buffer: {buffer_size})")
            
            self._is_connected = True
            self._use_realsense = False # 明确标记未使用 RealSense
            return True
        except Exception as e:
            print(f"❌ {self.camera_id}: OpenCV 连接失败: {e}")
            if self._color_cap:
                 try: self._color_cap.release()
                 except: pass
            return False
    
    def start_capture(self) -> bool:
        """启动采集线程"""
        if not self._is_connected:
            print(f"INFO: {self.camera_id}: 未连接，尝试连接...")
            if not self.connect():
                print(f"ERROR: {self.camera_id}: 连接失败，无法启动采集。")
                return False
        
        if self._capture_thread and self._capture_thread.is_alive():
            print(f"INFO: {self.camera_id}: 采集线程已在运行")
            return True
        
        self._stop_flag = False
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()
        
        print(f"🎬 {self.camera_id}: 开始图像采集")
        return True
    
    def stop_capture(self):
        """停止采集线程"""
        if not self._stop_flag:
            print(f"INFO: {self.camera_id}: 请求停止采集...")
            self._stop_flag = True
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                print(f"⚠️  {self.camera_id}: 警告 - 采集线程未能及时停止")
            else:
                 print(f"INFO: {self.camera_id}: 采集线程已停止")
        
        print(f"🛑 {self.camera_id}: 图像采集已停止")
    
    def _capture_worker(self):
        """采集工作线程"""
        print(f"INFO: {self.camera_id}: 采集线程已启动")
        
        last_capture_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 10 * self.fps # 允许连续10秒的错误

        while not self._stop_flag:
            try:
                loop_start_time = time.time()
                
                # --- RealSense 采集逻辑 ---
                # 使用 sleep + 阻塞 read (wait_for_frames)
                # 这种组合是可靠的，因为 wait_for_frames(timeout) 内部会处理掉旧帧
                
                # 1. 计算需要 sleep 的时间，保持目标帧率
                elapsed = loop_start_time - last_capture_time
                sleep_duration = self.target_interval - elapsed
                if sleep_duration > 0.001: # 避免极小的 sleep
                    time.sleep(sleep_duration)
                
                # 更新循环开始时间（如果在 sleep 后）
                current_time = time.time()
                
                # 2. 获取帧数据
                timestamp = get_timestamp() # 获取同步时间戳
                color_img, depth_img = self._get_frames()
                
                # 3. 处理获取到的帧
                if color_img is not None and depth_img is not None:
                    consecutive_errors = 0 # 成功获取，重置错误计数
                    with self.data_lock:
                        self.last_timestamp = timestamp
                        # 直接赋值，避免不必要的拷贝
                        self.last_color_img = color_img
                        self.last_depth_img = depth_img

                    # 调用回调函数（如果已设置）
                    if self.data_callback:
                        try:
                            # 传递的是当前读取到的帧，回调函数内部需要注意不要长时间持有
                            self.data_callback(self.camera_id, color_img, depth_img, timestamp)
                        except Exception as cb_e:
                             print(f"ERROR: {self.camera_id}: 回调函数出错: {cb_e}")
                    
                    self.frame_count += 1
                    self.last_frame_time = timestamp
                else:
                    # 获取帧失败
                    self.error_count += 1
                    consecutive_errors += 1
                    # 短暂休眠避免空转
                    time.sleep(0.01)
                    if consecutive_errors % self.fps == 0: # 每秒报告一次
                         print(f"WARN: {self.camera_id}: 获取图像失败 ({consecutive_errors} 次连续错误)")
                    if self.error_callback and consecutive_errors == 1: # 首次错误时报告
                         self.error_callback(self.camera_id, "获取图像失败")
                    
                    # 如果连续错误次数过多，可能硬件出问题了，停止线程
                    if consecutive_errors > max_consecutive_errors:
                         print(f"ERROR: {self.camera_id}: 连续 {consecutive_errors} 次获取图像失败，停止采集线程！")
                         if self.error_callback:
                              self.error_callback(self.camera_id, f"连续 {consecutive_errors} 次获取图像失败，线程停止")
                         self._stop_flag = True # 设置停止标志
                         break # 退出循环

                # 更新上次成功采集的时间戳（用于计算 sleep）
                # 注意：即使获取失败也更新，防止因连续失败导致 sleep 时间过长
                last_capture_time = current_time
                
            except Exception as e:
                # 捕获线程中的其他异常
                if self._stop_flag: # 如果是停止过程中发生的异常，忽略
                    break
                self.error_count += 1
                consecutive_errors += 1
                error_msg = f"采集线程异常: {e}"
                print(f"ERROR: {self.camera_id}: {error_msg}")
                if self.error_callback:
                    self.error_callback(self.camera_id, error_msg)
                
                # 发生异常后等待一段时间再重试
                time.sleep(0.1)
                
                # 检查连续错误
                if consecutive_errors > max_consecutive_errors:
                     print(f"ERROR: {self.camera_id}: 连续 {consecutive_errors} 次异常，停止采集线程！")
                     if self.error_callback:
                          self.error_callback(self.camera_id, f"连续 {consecutive_errors} 次异常，线程停止")
                     self._stop_flag = True
                     break

        print(f"INFO: {self.camera_id}: 采集线程已退出")
    
    def _get_frames(self):
        """根据连接类型获取帧"""
        try:
            if self._use_realsense:
                return self._get_realsense_frames()
            else:
                return self._get_opencv_frames()
        except Exception as e:
            # 简化错误报告，避免重复打印
            if "Frame didn't arrive in time" not in str(e): # RealSense 超时是正常现象
                 print(f"ERROR: {self.camera_id}: 获取帧时出错: {e}")
            return None, None
    
    def _get_realsense_frames(self):
        """使用 RealSense SDK 获取颜色和深度帧"""
        try:
            # wait_for_frames 会阻塞并返回最新的帧集（内部处理丢帧）
            # 设置超时时间，避免永久阻塞
            frames = self.pipeline.wait_for_frames(timeout_ms=int(self.target_interval * 1000 * 2)) # 等待最多2帧的时间
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            # 必须同时获取到颜色和深度才认为是有效帧
            if not color_frame or not depth_frame:
                return None, None
            
            # 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except RuntimeError as e:
            # RealSense 超时错误是 RuntimeError，特殊处理
            if "Frame didn't arrive in time" in str(e):
                # 超时是常见情况，尤其是在启动或高负载时，不算作严重错误
                pass
            else:
                 # 其他 RuntimeError 可能是更严重的问题
                 print(f"ERROR: {self.camera_id}: Realsense 运行时错误: {e}")
            return None, None
        except Exception as e:
            # 捕获其他可能的异常
            print(f"ERROR: {self.camera_id}: 获取 Realsense 帧时发生未知错误: {e}")
            return None, None
    
    def _get_opencv_frames(self):
        """使用 OpenCV 获取颜色帧，并生成伪深度帧"""
        try:
            # --- OpenCV 缓冲区处理 ---
            # 尝试主动读取并丢弃旧帧，获取最新帧
            # 注意：grab() 的效果依赖于摄像头驱动和 OpenCV 后端
            grabbed = self._color_cap.grab()
            if not grabbed:
                 # 如果 grab 失败，尝试传统的 read
                 ret, color_image = self._color_cap.read()
                 if not ret: return None, None
            else:
                 # 如果 grab 成功，retrieve 获取最新帧
                 ret, color_image = self._color_cap.retrieve()
                 if not ret: return None, None

            # --- 生成伪深度图 (保持不变) ---
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # 确保伪深度值不超出 uint16 范围
            depth_values = (255 - gray.astype(np.float32)) * 4.0
            depth_image = np.clip(depth_values, 0, 65535).astype(np.uint16)
            
            return color_image, depth_image
        except Exception as e:
            print(f"ERROR: {self.camera_id}: OpenCV 获取帧失败: {e}")
            return None, None
    
    def disconnect(self):
        """停止采集并释放资源"""
        self.stop_capture() # 确保线程已停止
        
        try:
            # 释放 RealSense 资源
            if hasattr(self, '_use_realsense') and self._use_realsense and hasattr(self, 'pipeline'):
                try:
                    self.pipeline.stop()
                    print(f"INFO: {self.camera_id}: RealSense pipeline 已停止")
                except Exception as e_rs_stop:
                     print(f"WARN: {self.camera_id}: 停止 RealSense pipeline 时出错 (忽略): {e_rs_stop}")
            
            # 释放 OpenCV 资源
            if hasattr(self, '_color_cap') and self._color_cap:
                try:
                    self._color_cap.release()
                    print(f"INFO: {self.camera_id}: OpenCV capture 已释放")
                except Exception as e_cv_release:
                     print(f"WARN: {self.camera_id}: 释放 OpenCV capture 时出错 (忽略): {e_cv_release}")
            
            # 清理引用
            self._color_cap = None
            self._depth_cap = None # 虽然 OpenCV 部分没有真的用 depth_cap
            if hasattr(self, 'pipeline'): self.pipeline = None

            self._is_connected = False
            print(f"INFO: {self.camera_id}: 相机已断开连接")
        except Exception as e:
            print(f"ERROR: {self.camera_id}: 断开连接时发生意外错误: {e}")
    
    def is_connected(self) -> bool:
        """检查相机是否连接"""
        # 可以增加更主动的检查，例如尝试读取一帧
        return self._is_connected
    
    def get_stats(self) -> Dict[str, Any]:
        """获取采集统计信息"""
        return {
            'camera_id': self.camera_id,
            'is_connected': self._is_connected,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'target_fps': self.fps,
            'last_frame_time': self.last_frame_time
        }
    
    def cleanup(self):
        """清理资源"""
        self.disconnect()

# ================================================================
#        >>> FIX: FFmpegWristCamera 修复帧重复问题 <<<
# ================================================================

class FFmpegWristCamera:
    """基于FFmpeg的手腕相机采集器 (MJPEG流 + 帧同步 + 主动清空 + 预热 + 缓冲限制 + 帧率控制)"""
    
    MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB 缓冲区限制
    WARMUP_FRAMES = 3  # 预热帧数

    def __init__(self, camera_id: str, device_index: int = 0, 
                 resolution: tuple = (640, 480), fps: int = 30):
        
        self.camera_id = camera_id
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.target_interval = 1.0 / fps 
        
        self._stop_flag = False
        self._capture_thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0.0
        
        self.last_color_img: Optional[np.ndarray] = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.last_timestamp: float = 0.0
        self.data_lock = threading.Lock()

        self._byte_buffer = bytearray()
        self._soi_marker = b'\xff\xd8'
        self._eoi_marker = b'\xff\xd9'

        print(f"手腕相机采集器初始化: {camera_id} (设备/dev/video{device_index}) [MJPEG增强模式]")
    
    def set_data_callback(self, callback: Callable[[str, np.ndarray, float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def get_latest_data(self) -> Tuple[float, np.ndarray]:
        with self.data_lock:
            return self.last_timestamp, self.last_color_img.copy()
    
    def connect(self) -> bool:
        try:
            command = [
                'ffmpeg',
                '-hide_banner', '-loglevel', 'error', 
                '-f', 'v4l2',                      
                '-input_format', 'mjpeg',          
                '-framerate', str(self.fps),       
                '-video_size', f'{self.resolution[0]}x{self.resolution[1]}', 
                '-fflags', 'nobuffer',             
                '-flags', 'low_delay',            
                '-probesize', '32',               
                '-analyzeduration', '0',          
                '-i', f'/dev/video{self.device_index}', 
                '-vsync', '0',
                '-f', 'mjpeg',                    
                '-q:v', '3',                      
                'pipe:1'                          
            ]
            
            print(f"INFO: {self.camera_id}: 启动 FFmpeg: {' '.join(command)}")
            
            self._ffmpeg_process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                bufsize=0 
            )

            fd = self._ffmpeg_process.stdout.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            print(f"INFO: {self.camera_id}: FFmpeg stdout 设置为非阻塞模式")

            time.sleep(0.5) 
            if self._ffmpeg_process.poll() is not None:
                 stderr_output = self._ffmpeg_process.stderr.read().decode(errors='ignore')
                 raise RuntimeError(f"FFmpeg 进程启动失败或立即退出。Stderr: {stderr_output}")

            print(f"INFO: {self.camera_id}: 初始清空 FFmpeg 输出缓冲区...")
            initial_discard_count = 0
            while True:
                rlist, _, _ = select.select([fd], [], [], 0) 
                if not rlist: break 
                try:
                    chunk = os.read(fd, 65536)
                    if not chunk: break 
                    initial_discard_count += len(chunk)
                except BlockingIOError: break 
                except Exception as e_discard:
                     print(f"WARN: {self.camera_id}: 初始清空缓冲区时发生错误 (忽略): {e_discard}")
                     break
            self._byte_buffer = bytearray() 
            print(f"INFO: {self.camera_id}: 初始丢弃 {initial_discard_count} 字节")

            self._is_connected = True
            print(f"✅ {self.camera_id}: FFmpeg 连接成功（MJPEG 流 + 非阻塞 + vsync passthrough）")
            return True
            
        except Exception as e:
            print(f"❌ {self.camera_id}: FFmpeg 连接失败: {e}")
            if hasattr(self, '_ffmpeg_process') and self._ffmpeg_process and self._ffmpeg_process.stderr:
                 try:
                      stderr_output = self._ffmpeg_process.stderr.read().decode(errors='ignore')
                      if stderr_output: print(f"ERROR: {self.camera_id}: FFmpeg Stderr: {stderr_output}")
                 except: pass 
            if hasattr(self, '_ffmpeg_process') and self._ffmpeg_process:
                 try: self._ffmpeg_process.kill()
                 except: pass
            self._ffmpeg_process = None
            self._is_connected = False
            return False
    
    def start_capture(self) -> bool:
        if not self._is_connected:
            print(f"INFO: {self.camera_id}: 未连接，尝试连接...")
            if not self.connect():
                print(f"ERROR: {self.camera_id}: 连接失败，无法启动采集。")
                return False
        
        if self._capture_thread and self._capture_thread.is_alive():
            print(f"INFO: {self.camera_id}: 采集线程已在运行")
            return True
        
        self._stop_flag = False
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()
        
        print(f"🎬 {self.camera_id}: 开始图像采集 (MJPEG 增强模式)")
        return True
    
    def stop_capture(self):
        if not self._stop_flag:
            print(f"INFO: {self.camera_id}: 请求停止采集...")
            self._stop_flag = True
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                print(f"⚠️  {self.camera_id}: 警告 - 采集线程未能及时停止")
            else:
                 print(f"INFO: {self.camera_id}: 采集线程已停止")

        print(f"🛑 {self.camera_id}: 图像采集已停止")
    
    def _capture_worker(self):
        """
        >>> FIX: 修复后的采集工作线程 - 添加帧率控制，解决帧重复问题
        
        核心修改：
        1. 添加 last_capture_time 变量跟踪上次处理帧的时间
        2. 在每次循环开始时计算并主动 sleep，确保按目标帧率采集
        3. 将 select 超时改为固定短时间（100ms），因为帧率已由 sleep 控制
        4. 确保即使在预热期间也更新 last_capture_time，保持正确的时序
        """
        print(f"INFO: {self.camera_id}: 采集线程已启动 (MJPEG 解析 + 清空 + 预热 + 缓冲限制 + 帧率控制)")
        
        if not self._ffmpeg_process or not self._ffmpeg_process.stdout:
             print(f"ERROR: {self.camera_id}: FFmpeg 进程未运行或 stdout 不可用，线程退出。")
             if self.error_callback:
                  self.error_callback(self.camera_id, "FFmpeg 进程无效")
             return

        fd = self._ffmpeg_process.stdout.fileno()
        consecutive_decode_errors = 0
        max_consecutive_decode_errors = 10
        
        warmup_frames_decoded = 0
        is_warmed_up = False
        
        # >>> FIX: 添加帧率控制变量，这是修复帧重复的关键
        last_capture_time = time.time()

        while not self._stop_flag:
            try:
                # >>> FIX: 帧率控制 - 在循环开始时主动 sleep，确保按目标帧率采集
                # 这防止了采集循环跑得太快，避免在新数据到达前反复处理同一帧
                loop_start_time = time.time()
                elapsed = loop_start_time - last_capture_time
                sleep_duration = self.target_interval - elapsed
                if sleep_duration > 0.001:  # 避免极小的 sleep
                    time.sleep(sleep_duration)
                
                # 更新当前时间（sleep 后）
                current_time = time.time()
                
                # 1. 使用 select 等待数据（超时时间缩短，因为已有 sleep 控制帧率）
                # >>> FIX: 将超时从 self.target_interval * 2 改为固定的 0.1 秒
                rlist, _, xlist = select.select([fd], [], [fd], 0.1)
                
                if xlist:
                     print(f"ERROR: {self.camera_id}: select 报告文件描述符错误，停止线程。")
                     if self.error_callback: self.error_callback(self.camera_id, "select 错误")
                     break
                if not rlist:
                    # >>> FIX: select 超时不再是问题，因为我们已经通过 sleep 控制了节奏
                    # 即使超时，也会在下次循环时等待正确的时间
                    continue

                # 2. 读取所有可用数据
                while True:
                    try:
                        chunk = os.read(fd, 65536)
                        if not chunk: 
                            if self._ffmpeg_process.poll() is not None:
                                print(f"ERROR: {self.camera_id}: FFmpeg 进程已退出。")
                                if self.error_callback: self.error_callback(self.camera_id, "FFmpeg 进程已退出")
                                self._stop_flag = True 
                            break 
                        self._byte_buffer.extend(chunk)
                    except BlockingIOError:
                        break 
                    except Exception as read_e:
                         print(f"ERROR: {self.camera_id}: 从 FFmpeg 管道读取时出错: {read_e}")
                         self.error_count += 1
                         if self.error_callback: self.error_callback(self.camera_id, f"读取管道错误: {read_e}")
                         break 
                
                if self._stop_flag: break

                # 检查并限制缓冲区大小
                if len(self._byte_buffer) > self.MAX_BUFFER_SIZE:
                    discard_size = len(self._byte_buffer) - self.MAX_BUFFER_SIZE
                    print(f"WARN: {self.camera_id}: 缓冲区超过 {self.MAX_BUFFER_SIZE} 字节，丢弃最旧的 {discard_size} 字节")
                    del self._byte_buffer[:discard_size]
                    processed_offset = 0 
                else:
                    processed_offset = 0

                # 3. 查找并处理帧
                last_valid_frame_bytes = None
                bytes_consumed_this_round = 0

                while True:
                    soi_index = self._byte_buffer.find(self._soi_marker, processed_offset)
                    
                    if soi_index == -1:
                        # 缓冲区剩余部分没有 SOI 了
                        if processed_offset > 0:
                             del self._byte_buffer[:processed_offset]
                        # 剩余部分可能包含半个 SOI，保留最后几个字节
                        if len(self._byte_buffer) > len(self._soi_marker):
                             del self._byte_buffer[:-len(self._soi_marker)]
                        break

                    eoi_index = self._byte_buffer.find(self._eoi_marker, soi_index + len(self._soi_marker))
                    
                    if eoi_index == -1:
                        # 找到了 SOI 但没找到 EOI，数据不完整
                        del self._byte_buffer[:soi_index]
                        processed_offset = 0 
                        break
                        
                    # 找到了一个完整的帧
                    frame_data = self._byte_buffer[soi_index : eoi_index + len(self._eoi_marker)]
                    last_valid_frame_bytes = frame_data 
                    
                    # 更新处理位置到这个完整帧之后
                    processed_offset = eoi_index + len(self._eoi_marker)
                    bytes_consumed_this_round = processed_offset 
                
                # 处理结束后，清理已消耗的字节
                if bytes_consumed_this_round > 0:
                    del self._byte_buffer[:bytes_consumed_this_round]

                # 4. 处理找到的最新有效帧
                if last_valid_frame_bytes:
                    timestamp = get_timestamp() 
                    
                    try:
                        frame_np = np.frombuffer(last_valid_frame_bytes, dtype=np.uint8)
                        color_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        
                        if color_img is None:
                            raise ValueError("cv2.imdecode 返回 None")
                        
                        consecutive_decode_errors = 0 
                        
                        # 预热逻辑
                        if not is_warmed_up:
                            warmup_frames_decoded += 1
                            if warmup_frames_decoded >= self.WARMUP_FRAMES:
                                is_warmed_up = True
                                print(f"✅ {self.camera_id}: 预热完成 ({self.WARMUP_FRAMES} 帧)，开始回调")
                            else:
                                # 预热期间不回调，但更新缓存
                                with self.data_lock:
                                    self.last_timestamp = timestamp
                                    self.last_color_img = color_img
                                # >>> FIX: 预热期间也要更新 last_capture_time，保持正确的时序
                                last_capture_time = current_time
                                continue

                        # 预热完成后，正常处理
                        with self.data_lock:
                            self.last_timestamp = timestamp
                            self.last_color_img = color_img 
                        
                        if self.data_callback:
                            try:
                                self.data_callback(self.camera_id, color_img, timestamp)
                            except Exception as cb_e:
                                print(f"ERROR: {self.camera_id}: 回调函数出错: {cb_e}")

                        self.frame_count += 1
                        self.last_frame_time = timestamp
                        
                        # >>> FIX: 更新上次成功采集的时间，用于下次循环的 sleep 计算
                        last_capture_time = current_time

                    except Exception as decode_e:
                        self.error_count += 1
                        consecutive_decode_errors += 1
                        error_msg = f"MJPEG 解码失败: {decode_e}"
                        # 减少打印频率
                        if consecutive_decode_errors == 1 or consecutive_decode_errors % 10 == 0:
                            print(f"ERROR: {self.camera_id}: {error_msg} (第 {consecutive_decode_errors} 次连续错误)")
                        if self.error_callback and consecutive_decode_errors == 1:
                            self.error_callback(self.camera_id, error_msg)
                        if consecutive_decode_errors > max_consecutive_decode_errors:
                            print(f"ERROR: {self.camera_id}: 连续 {consecutive_decode_errors} 次解码失败，停止线程！")
                            if self.error_callback: self.error_callback(self.camera_id, "连续解码失败，线程停止")
                            self._stop_flag = True 
                # >>> FIX: 即使没有找到有效帧，也不更新 last_capture_time
                # 这样下次循环会继续等待，直到有新帧到达
                
            except Exception as e:
                if self._stop_flag: break 
                self.error_count += 1
                error_msg = f"采集线程异常: {e}"
                print(f"ERROR: {self.camera_id}: {error_msg}")
                if self.error_callback:
                    self.error_callback(self.camera_id, error_msg)
                time.sleep(0.1)
        
        print(f"INFO: {self.camera_id}: 采集线程已退出")
    
    def disconnect(self):
        self.stop_capture() 
        
        try:
            if self._ffmpeg_process:
                print(f"INFO: {self.camera_id}: 正在终止 FFmpeg 进程...")
                self._ffmpeg_process.terminate()
                try:
                    self._ffmpeg_process.wait(timeout=1.0)
                    print(f"INFO: {self.camera_id}: FFmpeg 进程已终止")
                except subprocess.TimeoutExpired:
                    print(f"WARN: {self.camera_id}: FFmpeg 进程未能及时终止，强制 kill...")
                    self._ffmpeg_process.kill()
                    self._ffmpeg_process.wait(timeout=1.0) 
                    print(f"INFO: {self.camera_id}: FFmpeg 进程已被 kill")
                
                if self._ffmpeg_process.stdout: self._ffmpeg_process.stdout.close()
                if self._ffmpeg_process.stderr: self._ffmpeg_process.stderr.close()

                self._ffmpeg_process = None
            
            self._is_connected = False
            print(f"INFO: {self.camera_id}: 相机已断开连接")
        except Exception as e:
            print(f"ERROR: {self.camera_id}: 断开连接时发生意外错误: {e}")
            if hasattr(self, '_ffmpeg_process') and self._ffmpeg_process:
                 try: self._ffmpeg_process.kill()
                 except: pass
            self._ffmpeg_process = None
            self._is_connected = False 
    
    def is_connected(self) -> bool:
        if not self._is_connected: return False
        if self._ffmpeg_process and self._ffmpeg_process.poll() is not None:
            print(f"WARN: {self.camera_id}: 检测到 FFmpeg 进程已意外退出。")
            self._is_connected = False 
            return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'camera_id': self.camera_id,
            'is_connected': self._is_connected,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'target_fps': self.fps,
            'last_frame_time': self.last_frame_time,
            'buffer_size': len(self._byte_buffer)
        }
    
    def cleanup(self):
        self.disconnect()


# ================================================================
#               TripleCameraCollector (无需改动)
# ================================================================
class TripleCameraCollector:
    """三相机采集管理器：1个RealSense(camera_high) + 2个手腕相机(wrist cameras)"""
    
    def __init__(self, resolution: tuple = (640, 480), fps: int = 30):
        # RealSense相机
        self.camera_high = CameraCollector("camera_high", device_index=0, 
                                           resolution=resolution, fps=fps,
                                           serial_number="031522071209")
        
        # 两个手腕相机（使用修复后的 FFmpegWristCamera）
        self.camera_left_wrist = FFmpegWristCamera("camera_left_wrist", device_index=8,
                                                     resolution=resolution, fps=fps)
        self.camera_right_wrist = FFmpegWristCamera("camera_right_wrist", device_index=6,
                                                      resolution=resolution, fps=fps)
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        # 设置回调，将子采集器的事件传递给外部
        self.camera_high.set_data_callback(self._on_realsense_data)
        self.camera_high.set_error_callback(self._on_camera_error)
        self.camera_left_wrist.set_data_callback(self._on_wrist_data)
        self.camera_left_wrist.set_error_callback(self._on_camera_error)
        self.camera_right_wrist.set_data_callback(self._on_wrist_data)
        self.camera_right_wrist.set_error_callback(self._on_camera_error)
    
    def set_data_callback(self, callback: Callable):
        """设置统一的数据回调函数"""
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        """设置统一的错误回调函数"""
        self.error_callback = callback
    
    # --- 内部回调，用于将子采集器的事件转发给外部 ---
    def _on_realsense_data(self, camera_id: str, color_img: np.ndarray, 
                          depth_img: np.ndarray, timestamp: float):
        if self.data_callback:
            # RealSense 有 color 和 depth
            self.data_callback(camera_id, color_img, depth_img, timestamp)
    
    def _on_wrist_data(self, camera_id: str, color_img: np.ndarray, timestamp: float):
        if self.data_callback:
            # 手腕相机只有 color，depth 传递 None
            self.data_callback(camera_id, color_img, None, timestamp)
    
    def _on_camera_error(self, camera_id: str, error_message: str):
        if self.error_callback:
            self.error_callback(camera_id, error_message)
    # --- ------------------------------------------ ---

    def connect_all(self) -> bool:
        """连接所有三个相机"""
        print("INFO: 正在连接所有相机...")
        # 串行连接，方便调试
        results = {}
        results['high'] = self.camera_high.connect()
        results['left'] = self.camera_left_wrist.connect()
        results['right'] = self.camera_right_wrist.connect()
        
        success = all(results.values())
        print(f"相机连接状态: camera_high={'✅' if results['high'] else '❌'}, "
              f"camera_left_wrist={'✅' if results['left'] else '❌'}, "
              f"camera_right_wrist={'✅' if results['right'] else '❌'}")
        if not success:
             print("WARN: 部分相机连接失败。")
        return success
    
    def start_all(self) -> bool:
        """启动所有三个相机的采集线程"""
        print("INFO: 正在启动所有相机采集...")
        results = {}
        results['high'] = self.camera_high.start_capture()
        results['left'] = self.camera_left_wrist.start_capture()
        results['right'] = self.camera_right_wrist.start_capture()

        success = all(results.values())
        print(f"相机采集状态: camera_high={'✅' if results['high'] else '❌'}, "
              f"camera_left_wrist={'✅' if results['left'] else '❌'}, "
              f"camera_right_wrist={'✅' if results['right'] else '❌'}")
        if not success:
             print("WARN: 部分相机采集启动失败。")
        return success
    
    def stop_all(self):
        """停止所有三个相机的采集线程"""
        print("INFO: 正在停止所有相机采集...")
        # 并行停止可能更快，但串行停止更易于调试
        self.camera_high.stop_capture()
        self.camera_left_wrist.stop_capture()
        self.camera_right_wrist.stop_capture()
        print("🛑 所有相机采集已停止")
    
    def disconnect_all(self):
        """断开连接并清理所有三个相机"""
        print("INFO: 正在断开并清理所有相机...")
        self.camera_high.cleanup()
        self.camera_left_wrist.cleanup()
        self.camera_right_wrist.cleanup()
        print("🔌 所有相机已断开连接并清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有相机的统计信息"""
        return {
            'camera_high': self.camera_high.get_stats(),
            'camera_left_wrist': self.camera_left_wrist.get_stats(),
            'camera_right_wrist': self.camera_right_wrist.get_stats()
        }
    
    def cleanup(self):
        """清理所有相机资源"""
        self.disconnect_all()


# 兼容性别名
DualCameraCollector = TripleCameraCollector


# ================================================================
#                      测试代码
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 测试三相机数据采集系统（MJPEG 增强方案 + 帧率控制修复）")
    print("=" * 60)
    
    collector = TripleCameraCollector(resolution=(640, 480), fps=30)
    
    stats = {
        'camera_high': {'count': 0, 'last_ts': 0.0, 'interval_sum': 0.0},
        'camera_left_wrist': {'count': 0, 'last_ts': 0.0, 'interval_sum': 0.0},
        'camera_right_wrist': {'count': 0, 'last_ts': 0.0, 'interval_sum': 0.0}
    }
    lock = threading.Lock()

    def data_callback(camera_id, color_img, depth_img, timestamp):
        with lock:
            s = stats[camera_id]
            s['count'] += 1
            if s['last_ts'] > 0:
                 interval = timestamp - s['last_ts']
                 # 避免过大的间隔影响平均值（例如暂停或启动初期）
                 if interval < 1.0: # 只统计 1 秒内的间隔
                      s['interval_sum'] += interval
            s['last_ts'] = timestamp
            
            # 每隔一段时间打印一次，避免刷屏
            if s['count'] % 60 == 0: # 大约每 2 秒打印一次 @ 30fps
                avg_interval = s['interval_sum'] / (s['count'] -1) if s['count'] > 1 else 0
                avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                # 获取当前缓冲区大小（仅 wrist 相机）
                buffer_info = ""
                if "wrist" in camera_id:
                     cam = getattr(collector, camera_id, None)
                     if cam: buffer_info = f", Buf: {len(cam._byte_buffer)}B"

                print(f"[{camera_id}] 第{s['count']}帧 @ TS: {timestamp:.3f} (Avg FPS≈{avg_fps:.2f}{buffer_info})")
    
    def error_callback(camera_id, error_msg):
        print(f"❌ [{camera_id}] 错误: {error_msg}")
    
    collector.set_data_callback(data_callback)
    collector.set_error_callback(error_callback)
    
    try:
        print("\n🔌 连接相机...")
        if not collector.connect_all():
             print("❌ 连接失败，退出测试。")
             sys.exit(1)
        
        print("\n🎬 开始采集...")
        if not collector.start_all():
             print("❌ 启动采集失败，退出测试。")
             sys.exit(1)

        print("\n📹 采集中（10秒）...")
        print("💡 观察 wrist 相机是否稳定输出，以及 Avg FPS 是否接近 30\n")
        start_time = time.time()
        while time.time() - start_time < 10:
            time.sleep(1)
        
        print("\n🛑 停止采集...")
        collector.stop_all()
        
        print("\n📊 统计结果:")
        all_stats = collector.get_stats()
        for cam_name, cam_stat in all_stats.items():
            print(f"  {cam_name}: {cam_stat['frame_count']}帧, {cam_stat['error_count']}错误")
            # 基于回调计算平均 FPS
            s = stats[cam_name]
            if s['count'] > 1:
                 # 使用 interval_sum 计算更准确的平均值
                 valid_intervals_count = s['count'] - 1 # 减去第一帧
                 avg_interval = s['interval_sum'] / valid_intervals_count if valid_intervals_count > 0 else 0
                 avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                 print(f"     Callback Avg FPS ≈ {avg_fps:.2f} (基于 {valid_intervals_count} 个有效间隔)")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
         print(f"\n❌ 测试过程中发生意外错误: {e}")
         import traceback
         traceback.print_exc()
    finally:
        print("\n🧹 清理资源...")
        collector.cleanup()
        print("✅ 测试完成!")
