#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间同步模块
提供统一的时间戳获取函数
"""

import time


def get_timestamp() -> float:
    """
    获取当前时间戳（秒）
    
    Returns:
        float: 当前时间戳（单调时钟）
    """
    return time.time()
