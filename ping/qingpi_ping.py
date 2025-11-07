#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发Ping脚本，记录ping结果到本地日志
功能：
1. 每1秒并发ping一批IP地址
2. 将ping结果和每10秒成功率记录到本地日志并在控制台输出
"""

import asyncio
import subprocess
import time
from datetime import datetime
import json
from collections import defaultdict, deque
import signal
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ping_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class ConcurrentPinger:
    def __init__(self, ip_list):
        """
        初始化并发Pinger
        
        Args:
            ip_list: 需要ping的IP地址列表
        """
        self.ip_list = ip_list

        # 限制并发度的信号量
        self.semaphore = asyncio.Semaphore(50)  # 最多50个并发ping任务
        
        # 存储统计数据
        self.ping_results = defaultdict(deque)  # 存储每个IP最近的ping结果
        self.stats_window = 10  # 统计窗口大小(秒)
        self.running = True

    async def ping_ip(self, ip):
        """
        异步ping单个IP地址
        
        Args:
            ip: 目标IP地址
            
        Returns:
            tuple: (ip, success, response_time)
        """
        async with self.semaphore:
            try:
                # 使用系统ping命令，设置超时为1秒
                start_time = time.time()
                process = await asyncio.create_subprocess_exec(
                    'ping', '-c', '1', '-W', '1', ip,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await process.communicate()
                end_time = time.time()
                
                success = process.returncode == 0
                response_time = (end_time - start_time) * 1000 if success else None
                
                return ip, success, response_time
            except Exception as e:
                return ip, False, None
    
    async def ping_batch(self):
        """
        并发ping一批IP地址
        
        Returns:
            list: ping结果列表
        """
        tasks = [self.ping_ip(ip) for ip in self.ip_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error occurred during ping: {result}")
                continue
            processed_results.append(result)
            
        return processed_results
    
    def record_results(self, results):
        """
        记录ping结果到本地缓存
        
        Args:
            results: ping结果列表
        """
        timestamp = int(time.time())
        for ip, success, response_time in results:
            self.ping_results[ip].append({
                'timestamp': timestamp,
                'success': success,
                'response_time': response_time
            })
            
            # 限制每个IP保留的结果数量
            if len(self.ping_results[ip]) > 100:
                self.ping_results[ip].popleft()
    
    def calculate_success_rate(self, ip):
        """
        计算指定IP在统计窗口内的成功率
        
        Args:
            ip: 目标IP地址
            
        Returns:
            float: 成功率 (0-1)
        """
        current_time = int(time.time())
        window_start = current_time - self.stats_window
        
        results = self.ping_results[ip]
        if not results:
            return 0.0
            
        # 过滤统计窗口内的结果
        window_results = [r for r in results if r['timestamp'] >= window_start]
        if not window_results:
            return 0.0
            
        success_count = sum(1 for r in window_results if r['success'])
        return success_count / len(window_results)
    
    def log_ping_results(self, results):
        """
        记录ping结果到日志
        
        Args:
            results: ping结果列表
        """
        for ip, success, response_time in results:
            if success:
                logging.info(f"Ping {ip} - Success - Response time: {response_time:.2f}ms")
            else:
                logging.warning(f"Ping {ip} - Failed")
    
    def log_success_rate(self):
        """
        记录成功率统计到日志
        """
        current_time = int(time.time())
        if current_time % 10 == 0:  # 每10秒记录一次
            for ip in self.ip_list:
                success_rate = self.calculate_success_rate(ip)
                logging.info(f"Success rate for {ip}: {success_rate:.2%} (window: {self.stats_window}s)")

    async def run(self):
        """
        主运行循环
        """
        logging.info("Starting concurrent pinger...")
        last_stats_time = 0
        
        while self.running:
            try:
                # 并发ping所有IP
                results = await self.ping_batch()
                
                # 记录结果到本地缓存
                self.record_results(results)
                
                # 记录ping结果到日志
                self.log_ping_results(results)
                
                # 记录成功率统计到日志
                self.log_success_rate()
                
                # 每10秒输出一次本地统计信息
                current_time = int(time.time())
                if current_time - last_stats_time >= 10:
                    self.print_local_stats()
                    last_stats_time = current_time
                
                # 等待1秒后继续下一轮
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logging.info("Received interrupt signal, shutting down...")
                self.running = False
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)  # 出错时等待1秒再继续
    
    def print_local_stats(self):
        """
        打印本地统计信息到控制台
        """
        print(f"\n--- Statistics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        for ip in self.ip_list:
            success_rate = self.calculate_success_rate(ip)
            print(f"IP: {ip}, Success Rate: {success_rate:.2%}")
        print("-" * 50)
    
    def stop(self):
        """
        停止运行
        """
        self.running = False

def signal_handler(signum, frame):
    """
    信号处理函数
    """
    print("\nReceived signal, shutting down gracefully...")
    sys.exit(0)

def main():
    """
    主函数
    """
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 配置需要ping的IP地址列表
    ip_list = [
        '172.17.0.75',
        '172.17.4.75',
        '172.17.5.227',
        '172.17.80.139',
        '172.17.64.250',
        '172.17.80.136',
        '172.17.81.196'
    ]
    
    # 创建并运行并发Pinger
    pinger = ConcurrentPinger(ip_list)
    
    try:
        # 兼容旧版本Python的运行方式
        if sys.version_info >= (3, 7):
            # Python 3.7+
            asyncio.run(pinger.run())
        else:
            # Python 3.6及以下版本
            loop = asyncio.get_event_loop()
            loop.run_until_complete(pinger.run())
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pinger.stop()

if __name__ == '__main__':
    main()