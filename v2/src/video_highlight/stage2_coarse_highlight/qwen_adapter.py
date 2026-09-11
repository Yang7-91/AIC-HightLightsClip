"""通过 OpenAI 兼容 API 调用已经部署好的 Qwen3.5-4B。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from video_highlight.common.exceptions import ExternalToolError


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    reasoning: str | None
    response_id: str | None
    model: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "response_id": self.response_id,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


class QwenBackend(Protocol):
    def healthcheck(self) -> None: ...

    def analyze(self, system_prompt: str, user_prompt: str, video_item: dict[str, Any],response_format: dict[str, Any]|None) -> ModelResponse: ...


class OpenAIQwenBackend(QwenBackend):
    """只负责 HTTP 请求，不包含模型加载、量化或 vLLM 部署逻辑。"""

    def __init__(self, api_config: dict[str, Any], generation_config: dict[str, Any]) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ExternalToolError("缺少 openai 包，请在 video-clip Conda 环境中安装") from error
        self.model = str(api_config["model"])
        self.generation_config = generation_config
        self.client = OpenAI(
            base_url=str(api_config["base_url"]).rstrip("/") + "/",
            api_key=str(api_config.get("api_key", "EMPTY")),
            timeout=float(api_config.get("timeout_sec", 300.0)),
            max_retries=int(api_config.get("max_retries", 2)),
        )

    def healthcheck(self) -> None:
        try:
            identifiers = {item.id for item in self.client.models.list().data}
        except Exception as error:
            raise ExternalToolError(f"无法访问 vLLM OpenAI API: {error}") from error
        if self.model not in identifiers:
            raise ExternalToolError(f"vLLM 未暴露配置模型 {self.model!r}，当前模型: {sorted(identifiers)}")

    def analyze(self, system_prompt: str, user_prompt: str, video_item: dict[str, Any],response_format: dict[str, Any]|None) -> ModelResponse:
        parameters: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [video_item, {"type": "text", "text": user_prompt}]},
            ],
            "temperature": float(self.generation_config.get("temperature", 0.0)),
            "top_p": float(self.generation_config.get("top_p", 0.9)),
            "max_tokens": int(self.generation_config.get("max_tokens", 4096)),
        }
        if response_format is not None:
            parameters["response_format"] = response_format
        elif self.generation_config.get("response_format"):
            parameters["response_format"] = self.generation_config["response_format"]
        if self.generation_config.get("extra_body"):
            parameters["extra_body"] = self.generation_config["extra_body"]
        try:
            response = self.client.chat.completions.create(**parameters)
        except Exception as error:
            raise ExternalToolError(f"Qwen API 调用失败: {error}") from error
        if not response.choices:
            raise ExternalToolError("Qwen API 返回空 choices")
        choice = response.choices[0]
        message = choice.message
        content = message.content
        if isinstance(content, list): # 兼容判断，目前实际上content只会返回要求格式的内容
            text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        else:
            text = str(content or "")
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        finish_reason = getattr(choice, "finish_reason", None)
        # 新版 vLLM/OpenAI SDK 使用 message.reasoning，部分兼容实现仍使用
        # reasoning_content。思考文本仅用于诊断，不能当作最终 JSON 解析。
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        completion_tokens = usage.get("completion_tokens") if usage else None
        completion_details = usage.get("completion_tokens_details") if usage else None
        reasoning_tokens = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        )
        thinking_enabled = bool(
            self.generation_config.get("extra_body", {})
            .get("chat_template_kwargs", {})
            .get("enable_thinking", False)
        )
        if not text.strip():
            hint = "；当前 enable_thinking=true，建议关闭思考模式" if thinking_enabled else ""
            raise ExternalToolError(
                "Qwen API 返回空 content"
                f"（finish_reason={finish_reason!r}, completion_tokens={completion_tokens}, "
                f"reasoning_tokens={reasoning_tokens}）{hint}"
            )
        if finish_reason == "length":
            hint = "，思考模式可能占用了输出预算" if thinking_enabled else ""
            raise ExternalToolError(
                "Qwen API 输出达到 max_tokens 而被截断"
                f"（completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}{hint}）"
            )
        return ModelResponse(
            text=text,
            reasoning=str(reasoning) if reasoning else None,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
            finish_reason=finish_reason,
            usage=usage,
        )


class MockQwenBackend(QwenBackend):
    """离线测试数据链路使用，不参与正式预测。"""

    def healthcheck(self) -> None:
        return None

    def analyze(self, system_prompt: str, user_prompt: str, video_item: dict[str, Any],response_format: dict[str, Any]|None) -> ModelResponse:
        del system_prompt, user_prompt, video_item
        result = {
            "has_highlight": True,
            "candidates": [
                {
                    "summary": "mock candidate",
                    "highlight_score": 0.75,
                    "completeness_score": 0.8,
                    "start_offset_sec": 1.0,
                    "end_offset_sec": 2.0,
                    "category": "other",
                    "subject": "mock subject",
                    "reason": "mock backend",
                    "evidence_sample_ids": [0],
                },
                {
                    "summary": "second mock candidate",
                    "highlight_score": 0.7,
                    "completeness_score": 0.75,
                    "start_offset_sec": 3.5,
                    "end_offset_sec": 4.5,
                    "category": "emotion_peak",
                    "subject": None,
                    "reason": "second mock event",
                    "evidence_sample_ids": [],
                },
            ],
        }
        return ModelResponse(json.dumps(result), None, "mock", "mock", "stop", None)


def build_backend(config: dict[str, Any]) -> QwenBackend:
    backend_name = str(config.get("runtime", {}).get("backend", "openai")).lower()
    if backend_name == "openai":
        return OpenAIQwenBackend(config["api"], config.get("generation", {}))
    if backend_name == "mock":
        return MockQwenBackend()
    raise ExternalToolError(f"未知 Stage 2 backend: {backend_name}")


# def build_backend(config: dict[str, Any]) -> QwenBackend:
#     backend = str(config.get("runtime", {}).get("backend", "openai")).lower()
#     if backend == "openai":
#         return OpenAIQwenBackend(config["api"], config.get("generation", {}))
#     if backend == "mock":
#         return MockQwenBackend()
#     raise ValueError(f"未知 Stage 2 backend: {backend}")
