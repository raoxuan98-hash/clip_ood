#!/usr/bin/env python
"""
配置管理模块

功能:
    1. 加载YAML配置文件
    2. 支持配置继承（base + experiment）
    3. 配置验证
    4. 命令行参数覆盖
    5. 自动生成实验目录结构

Usage:
    from src.utils.config_manager import ConfigManager
    
    # 加载配置
    config = ConfigManager.load("configs/experiments/my_exp.yaml")
    
    # 访问配置
    lr = config.training.lr
    datasets = config.data.id_datasets
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime


class ConfigError(Exception):
    """配置错误"""
    pass


class ConfigManager:
    """
    配置管理器
    
    支持：
    - 从YAML文件加载配置
    - 配置继承和合并
    - 配置验证
    - 转换为脚本参数
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        从字典初始化配置
        
        Args:
            config_dict: 配置字典
        """
        self._config = config_dict
        self._validate()
    
    @classmethod
    def load(cls, config_path: str, base_dir: str = "configs/base") -> "ConfigManager":
        """
        从YAML文件加载配置
        
        如果配置中包含`inherits`字段，会先加载基础配置并合并
        
        Args:
            config_path: 配置文件路径
            base_dir: 基础配置目录
        
        Returns:
            ConfigManager实例
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        
        # 加载实验配置
        with open(config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)
        
        # 检查是否继承基础配置
        inherits = exp_config.pop('inherits', None)
        
        if inherits:
            # 加载基础配置
            base_path = Path(base_dir) / f"{inherits}.yaml"
            if not base_path.exists():
                raise ConfigError(f"Base config not found: {base_path}")
            
            with open(base_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)
            
            # 合并配置（实验配置覆盖基础配置）
            merged_config = cls._merge_configs(base_config, exp_config)
            print(f"✓ Loaded config from {config_path} (inherits: {inherits})")
        else:
            merged_config = exp_config
            print(f"✓ Loaded config from {config_path}")
        
        return cls(merged_config)
    
    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> Dict:
        """
        递归合并两个字典
        
        override中的值会覆盖base中的值
        
        Args:
            base: 基础配置
            override: 覆盖配置
        
        Returns:
            合并后的配置
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并子字典
                result[key] = ConfigManager._merge_configs(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    def _validate(self):
        """
        验证配置合法性
        
        检查必填字段和字段类型
        """
        required_fields = ['experiment', 'data']
        
        for field in required_fields:
            if field not in self._config:
                raise ConfigError(f"Missing required field: {field}")
        
        # 验证experiment字段
        exp = self._config.get('experiment', {})
        if 'name' not in exp:
            raise ConfigError("Missing experiment.name")
        
        # 验证data字段
        data = self._config.get('data', {})
        if 'id_datasets' not in data and 'task_sequence' not in data:
            raise ConfigError("Missing data.id_datasets or data.task_sequence")
        
        # 验证数值范围
        if 'training' in self._config:
            training = self._config['training']
            if training.get('enabled', True):
                lr = training.get('lr', 0)
                if not 0 < lr < 1:
                    raise ConfigError(f"Invalid learning rate: {lr}")
                
                iterations = training.get('iterations', 0)
                if iterations < 1:
                    raise ConfigError(f"Invalid iterations: {iterations}")
        
        print("✓ Config validation passed")
    
    def __getattr__(self, name: str) -> Any:
        """
        通过属性访问配置
        
        Args:
            name: 配置项名称
        
        Returns:
            配置值
        """
        if name in self._config:
            value = self._config[name]
            # 如果值是字典，包装为ConfigAccessor以便链式访问
            if isinstance(value, dict):
                return ConfigAccessor(value)
            return value
        raise AttributeError(f"Config has no attribute: {name}")
    
    def __getitem__(self, key: str) -> Any:
        """通过键访问配置"""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        安全获取配置值
        
        Args:
            key: 键，支持点号分隔（如"training.lr"）
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
    
    def to_args(self) -> argparse.Namespace:
        """
        转换为argparse.Namespace（用于脚本）
        
        将嵌套配置展平为argparse参数
        
        Returns:
            argparse.Namespace
        """
        flat_dict = self._flatten_dict(self._config)
        return argparse.Namespace(**flat_dict)
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        展平嵌套字典
        
        Args:
            d: 字典
            parent_key: 父键
            sep: 分隔符
        
        Returns:
            展平后的字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ConfigManager._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def save(self, path: str):
        """
        保存配置到YAML文件
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ Config saved to {path}")
    
    def create_experiment_dir(self, base_dir: str = None) -> Path:
        """
        创建实验目录结构
        
        Args:
            base_dir: 基础目录，默认为config中的output_dir
        
        Returns:
            实验目录路径
        """
        if base_dir is None:
            base_dir = self.get('paths.output_dir', './outputs')
        
        exp_name = self.experiment.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
        
        # 创建子目录
        subdirs = ['checkpoints', 'logs', 'results', 'cache', 'visualizations']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 保存配置副本
        self.save(exp_dir / 'config.yaml')
        
        print(f"✓ Experiment directory created: {exp_dir}")
        return exp_dir
    
    def __repr__(self) -> str:
        """字符串表示"""
        exp_name = self.get('experiment.name', 'unknown')
        return f"ConfigManager(experiment={exp_name})"


class ConfigAccessor:
    """
    配置访问器
    
    支持通过属性链式访问嵌套配置
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigAccessor(value)
            return value
        raise AttributeError(f"Config has no attribute: {name}")
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def __repr__(self) -> str:
        return f"ConfigAccessor({self._data})"


def load_config(config_path: str, **overrides) -> ConfigManager:
    """
    便捷函数：加载配置
    
    Args:
        config_path: 配置文件路径
        **overrides: 覆盖的配置项
    
    Returns:
        ConfigManager实例
    
    Example:
        config = load_config("configs/exp.yaml", training__lr=0.001)
    """
    config = ConfigManager.load(config_path)
    
    # 应用覆盖
    for key, value in overrides.items():
        keys = key.split('__')
        d = config._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    
    return config


# 命令行参数解析
def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加配置相关命令行参数
    
    Args:
        parser: ArgumentParser实例
    
    Returns:
        添加了参数的parser
    """
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--override', '-o',
        type=str,
        nargs='*',
        default=[],
        help='Override config values (format: key=value)'
    )
    parser.add_argument(
        '--exp-dir', '-d',
        type=str,
        default=None,
        help='Experiment directory (auto-generated if not specified)'
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ConfigManager:
    """
    从命令行参数创建配置
    
    Args:
        args: 解析后的命令行参数
    
    Returns:
        ConfigManager实例
    """
    config = ConfigManager.load(args.config)
    
    # 应用命令行覆盖
    for override in getattr(args, 'override', []):
        if '=' not in override:
            raise ConfigError(f"Invalid override format: {override}. Use key=value")
        key, value = override.split('=', 1)
        
        # 尝试解析值的类型
        try:
            value = eval(value)
        except:
            pass  # 保持为字符串
        
        # 设置值
        keys = key.split('.')
        d = config._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        
        print(f"  Override: {key} = {value}")
    
    return config


# 示例用法
if __name__ == "__main__":
    # 测试配置加载
    try:
        config = ConfigManager.load("configs/experiments/lora_nsp_caltech.yaml")
        print(f"\nExperiment: {config.experiment.name}")
        print(f"Description: {config.experiment.description}")
        print(f"ID datasets: {config.data.id_datasets}")
        print(f"Learning rate: {config.training.lr}")
        
        # 创建实验目录
        exp_dir = config.create_experiment_dir()
        print(f"\nCreated experiment at: {exp_dir}")
        
    except Exception as e:
        print(f"Error: {e}")