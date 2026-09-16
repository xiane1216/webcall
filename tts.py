# tts.py — MOSS-TTS 语音合成模块（OpenAI 兼容 /v1/audio/speech）

import os
import re
import requests
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "moss-tts-1.5-flash"
DEFAULT_VOICE_ID = "187b0c9b-612c-4019-98fe-c7bb59eb8074"
DEFAULT_API_BASE = "https://api.mosi.cn/v1"


def _get_env(name, default=""):
    return os.environ.get(name, default).strip()


def synthesize(text, api_key=None, voice_id=None, group_id=None, model=None):
    """
    将文本合成为 MP3 语音。
    返回 mp3 bytes；输入为纯标点/空白时返回 None。
    group_id 参数为兼容旧接口保留，moss 不使用。
    """
    if not text or not isinstance(text, str):
        return None

    stripped = re.sub(r'[^\w]', '', text, flags=re.UNICODE)
    if not stripped:
        return None

    if api_key is None:
        api_key = _get_env("MOSS_API_KEY")
    if not api_key:
        raise ValueError("未提供 api_key，且环境变量 MOSS_API_KEY 未设置")

    if voice_id is None:
        voice_id = _get_env("MOSS_VOICE_ID", DEFAULT_VOICE_ID)
    if model is None:
        model = _get_env("MOSS_TTS_MODEL", DEFAULT_MODEL)
    api_base = _get_env("MOSS_API_BASE", DEFAULT_API_BASE)

    url = api_base.rstrip("/") + "/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice_id,
        "response_format": "mp3",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        raise Exception("MOSS TTS 请求超时")
    except requests.exceptions.ConnectionError as e:
        raise Exception(f"MOSS TTS 连接失败: {e}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"MOSS TTS 请求异常: {e}")

    if resp.status_code != 200:
        raise Exception(f"MOSS TTS HTTP 错误: {resp.status_code} — {resp.text[:500]}")

    ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype:
        try:
            data = resp.json()
        except Exception:
            raise Exception("MOSS TTS 返回了无法解析的 JSON")
        audio_hex = data.get("audio") or (data.get("data") or {}).get("audio")
        if audio_hex:
            try:
                audio_bytes = bytes.fromhex(audio_hex)
            except Exception:
                raise Exception("MOSS TTS 音频 hex 解码失败")
        else:
            raise Exception(f"MOSS TTS 返回异常: {str(data)[:300]}")
    else:
        audio_bytes = resp.content

    if not audio_bytes:
        raise Exception("MOSS TTS 返回的音频数据为空")

    logger.info("MOSS TTS 合成成功，音频大小=%d bytes", len(audio_bytes))
    return audio_bytes
