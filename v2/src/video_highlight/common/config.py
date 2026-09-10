"""配置读取与路径解析工具。"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


def load_mapping(path: str | Path) -> dict[str, Any]:
    """读取 JSON 或 YAML 配置；JSON 兼容 YAML 无需额外依赖。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"配置文件不存在: {config_path}")
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as import_error:
            raise ConfigurationError(
                f"{config_path} 不是 JSON 兼容 YAML，且当前环境未安装 PyYAML"
            ) from import_error
        try:
            value = yaml.safe_load(text)
        except Exception as yaml_error:
            raise ConfigurationError(f"无法解析配置 {config_path}: {yaml_error}") from json_error
    if not isinstance(value, dict):
        raise ConfigurationError(f"配置根节点必须是对象: {config_path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置，override 中的值优先。"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    """展开环境变量，并把相对路径解析到指定基准目录。"""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = Path(base_dir) / expanded
    return expanded.resolve()
