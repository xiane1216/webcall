# stt.py — 硅基流动 SenseVoiceSmall 语音识别模块

import os
import re
import time
import tempfile
import subprocess
import requests
import logging

logger = logging.getLogger(__name__)

_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"


def _clean_sensevoice_tags(text: str) -> str:
    """清理 SenseVoice 返回的特殊标签，如 <|zh|> <|NEUTRAL|> <|Speech|>"""
    text = re.sub(r'<\s*\|?/?[A-Za-z_0-9]+\|?\s*>', '', text)
    text = re.sub(r'<\s*\|?/?[A-Za-z_0-9]+\|?\s*>', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _convert_to_wav(src_path: str):
    """用 ffmpeg 将音频转为 16kHz mono WAV，失败返回 None。"""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=15,
        )
        if result.returncode != 0:
            os.remove(wav_path)
            return None
        return wav_path
    except FileNotFoundError:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return None
    except Exception:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return None


def recognize(audio_file):
    """
    语音识别入口，使用硅基流动 SenseVoiceSmall。
    audio_file: Flask request.files 中的文件对象
    返回 dict — 成功 {text, time, engine} / 失败 {error} / {text:'', message:'录音太短'}
    """
    tmp_path = None
    wav_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".webm")
        os.close(fd)
        audio_file.save(tmp_path)

        if os.path.getsize(tmp_path) < 1000:
            return {"text": "", "message": "录音太短"}

        wav_path = _convert_to_wav(tmp_path)
        audio_path = wav_path if wav_path else tmp_path
        audio_ext = ".wav" if wav_path else ".webm"
        audio_mime = "audio/wav" if wav_path else "audio/webm"

        api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            try:
                from llm import get_stt_defaults
                api_key = get_stt_defaults().get("siliconflow_api_key", "") or api_key
            except Exception:
                pass
        if not api_key:
            return {"error": "语音识别未配置（需设置 SILICONFLOW_API_KEY）"}

        start_time = time.time()
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            filename = "audio" + audio_ext
            with open(audio_path, "rb") as f:
                files = {"file": (filename, f, audio_mime)}
                data = {"model": "FunAudioLLM/SenseVoiceSmall"}
                resp = requests.post(
                    _SILICONFLOW_API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30,
                )
        except requests.exceptions.Timeout:
            return {"error": "语音识别超时，请重试"}
        except requests.exceptions.ConnectionError:
            return {"error": "无法连接语音识别服务"}
        except Exception as e:
            return {"error": f"语音识别异常: {str(e)}"}

        elapsed = round(time.time() - start_time, 2)

        if resp.status_code != 200:
            return {"error": f"硅基流动 API 错误 ({resp.status_code}): {resp.text[:200]}"}

        try:
            result = resp.json()
        except Exception:
            return {"error": "硅基流动返回了无效的 JSON"}

        text = (result.get("text") or "").strip()
        text = _clean_sensevoice_tags(text)

        return {"text": text, "time": elapsed, "engine": "siliconflow-sensevoice"}

    except Exception as e:
        return {"error": f"语音识别异常: {str(e)}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
