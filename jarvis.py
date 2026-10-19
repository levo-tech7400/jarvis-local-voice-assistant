#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jarvis - tamamen yerel calisan sesli asistan.

Akis:
    mikrofon --> VAD (enerji tabanli) --> faster-whisper --> LM Studio (/v1/chat/completions)
             --> edge-tts (veya pyttsx3) --> hoparlor

Hizli baslangic:
    python jarvis.py --check          # ortami kontrol et
    python jarvis.py                  # sesli mod
    python jarvis.py --text           # mikrofon yerine klavye
    python jarvis.py --once "merhaba" # tek komut
    python jarvis.py --say "test"     # sadece ses testi
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Opsiyonel bagimliliklar (--check bunlari raporlayabilmek icin yumusak import)
# ---------------------------------------------------------------------------
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore[assignment]

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


APP_DIR = Path(__file__).resolve().parent
LOG = logging.getLogger("jarvis")
_LOCAL = threading.local()
WHISPER_RATE = 16000
_DLL_HANDLES: List[Any] = []   # os.add_dll_directory tutamaclari (GC'ye kurban gitmesin)

DEFAULT_SYSTEM_PROMPT = (
    "Sen Jarvis'sin: kullanicinin bilgisayarinda tamamen yerel calisan, Turkce konusan bir sesli asistansin. "
    "Cevaplarin hoparlorden okunacagi icin kisa, dogal ve net konus (genelde 1-3 cumle). "
    "Madde isareti, emoji, tablo, kod blogu ve markdown bicimlendirmesi kullanma. "
    "Gerektiginde nazik bir sekilde 'efendim' diye hitap edebilirsin ama abartma. "
    "Bilmedigin bir seyi uydurma; bilmiyorsan acikca soyle. "
    "Hava durumu, sicaklik veya yagis sorulursa mutlaka get_weather aracini cagir; "
    "konum soylenmezse varsayilan konum Kirklareli kullanilir. "
    "Kullanici senden bir islem yapmani isterse, yapamayacagin islemler icin kisaca nedenini acikla."
)

EXIT_WORDS = (
    "cikis", "cik", "kapat", "kendini kapat", "kapan", "gorusuruz",
    "exit", "quit", "stop", "bye", "kapat jarvis", "yeter",
)

EPILOG = """\
ornekler:
  python jarvis.py --check                     ortam + LM Studio kontrolu
  python jarvis.py                             sesli sohbet (varsayilan)
  python jarvis.py --whisper-model base        daha hizli STT (daha az dogru)
  python jarvis.py --voice en-US-AvaMultilingualNeural   cok dilli ses
  python jarvis.py --tts pyttsx3               tamamen cevrimdisi seslendirme
  python jarvis.py --no-tts --text             sadece yazi modu
  python jarvis.py --wake-word jarvis          sadece 'jarvis' gecen komutlar
"""


# ---------------------------------------------------------------------------
# Kucuk yardimcilar
# ---------------------------------------------------------------------------
def setup_console() -> None:
    """Windows konsolunda Turkce karakterlerin bozulmasini engeller."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                # line_buffering: cikti bir dosyaya/pipe'a yonlendirildiginde de aninda yazilsin
                reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            except TypeError:
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            except Exception:
                pass


_COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    """ANSI renk (TTY degilse duz metin)."""
    return "\033[{}m{}\033[0m".format(code, text) if _COLOR else text


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "evet", "on", "acik")


@contextlib.contextmanager
def _silence_output() -> Iterator[None]:
    """Kutuphanelerin acilis gurultusunu (pygame banner'i gibi) bastirir."""
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as null:
            sys.stdout = null
            sys.stderr = null
            yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def to_whisper_pcm(audio: Any, src_rate: int) -> Any:
    """int16 kaydi Whisper'in beklediği 16 kHz float32 diziye cevirir."""
    if np is None:
        raise JarvisError("numpy kurulu degil. Kurulum: pip install numpy")
    data = np.asarray(audio)
    if data.size == 0:
        return np.zeros(0, dtype=np.float32)
    if data.dtype != np.int16:
        if data.dtype.kind == "f":
            # float32 [-1, 1] (orn. faster_whisper.decode_audio ciktisi) veya ham genlik
            peak = float(np.max(np.abs(data)))
            scale = 32768.0 if peak <= 1.5 else 1.0
            data = np.clip(data * scale, -32768, 32767).astype(np.int16)
        else:
            data = np.clip(data, -32768, 32767).astype(np.int16)
    if src_rate == WHISPER_RATE:
        return data.astype(np.float32) / 32768.0
    if src_rate > WHISPER_RATE and src_rate % WHISPER_RATE == 0:
        # Tam sayi orani: kutu filtre ile yumusak indirgeme (48k -> 16k)
        factor = src_rate // WHISPER_RATE
        usable = (data.size // factor) * factor
        if usable == 0:
            return data.astype(np.float32) / 32768.0
        blocks = data[:usable].astype(np.float32).reshape(-1, factor)
        return (blocks.mean(axis=1) / 32768.0).astype(np.float32)
    # Genel durum: dogrusal enterpolasyon
    source = data.astype(np.float32) / 32768.0
    target_len = max(1, int(round(source.size * float(WHISPER_RATE) / float(src_rate))))
    old_x = np.linspace(0.0, float(source.size), num=source.size, endpoint=False)
    new_x = np.linspace(0.0, float(source.size), num=target_len, endpoint=False)
    return np.interp(new_x, old_x, source).astype(np.float32)


def _prepare_cuda_dlls() -> List[str]:
    """nvidia-* pip paketlerindeki cuBLAS/cuDNN DLL klasorlerini gorunur yapar.

    CTranslate2 bu kutuphaneleri calisma zamaninda LoadLibrary ile arar; bu yuzden
    klasorlerin process PATH'inde olmasi gerekir. Ayni klasorler
    os.add_dll_directory ile de eklenir (tutamaclari modul seviyesinde saklanir).
    """
    if os.name != "nt":
        return []
    roots: List[Path] = []
    try:
        import site

        roots.extend(Path(p) for p in site.getsitepackages() if p)
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        pass

    added: List[str] = []
    seen: set = set()
    for root in roots:
        nvidia_dir = root / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in sorted(nvidia_dir.glob("*/bin")):
            if not bin_dir.is_dir():
                continue
            path_str = str(bin_dir)
            if path_str.lower() in seen:
                continue
            seen.add(path_str.lower())
            current = os.environ.get("PATH", "")
            if path_str.lower() not in current.lower():
                os.environ["PATH"] = path_str + os.pathsep + current
            try:
                _DLL_HANDLES.append(os.add_dll_directory(path_str))
            except Exception:
                pass
            added.append(path_str)
    if added:
        LOG.debug("CUDA DLL klasorleri eklendi: %s", ", ".join(added))
    return added


def parse_nvidia_smi(text: str) -> List[Tuple[int, str, int]]:
    """`nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader` ciktisini ayristirir.

    Ornek satir: ``0, NVIDIA GeForce RTX 3060, 12288`` → (0, 'NVIDIA GeForce RTX 3060', 12288)
    """
    devices: List[Tuple[int, str, int]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            memory = int(float(parts[-1].replace(" MiB", "").replace("MiB", "").strip()))
        except (TypeError, ValueError):
            continue
        name = ",".join(parts[1:-1]).strip() or "?"
        devices.append((index, name, memory))
    return devices


def list_cuda_devices() -> List[Tuple[int, str, int]]:
    """Sistemdeki NVIDIA GPU'lari (indeks, ad, VRAM MiB) olarak dondurur."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=creationflags,
        )
    except Exception as exc:
        LOG.debug("nvidia-smi calistirilamadi: %s", exc)
        return []
    if completed.returncode != 0:
        LOG.debug("nvidia-smi hatasi: %s", (completed.stderr or "").strip()[:200])
        return []
    return parse_nvidia_smi(completed.stdout)


def count_cuda_devices() -> int:
    """CTranslate2'nin gordugu CUDA aygit sayisi."""
    try:
        import ctranslate2  # type: ignore

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    # LM Studio
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""                     # bos -> /v1/models uzerinden otomatik
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.6
    max_tokens: int = 1024
    history_turns: int = 8
    request_timeout: float = 120.0
    disable_thinking: bool = True       # Qwen3 vb. reasoning modelleri kisa cevap icin
    # STT
    language: str = "tr"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_gpu: int = 0                # coklu GPU: Whisper hangi kartta calissin (0,1,...)
    # Mikrofon / VAD
    input_device: Optional[int] = None
    sample_rate: int = 16000
    frame_ms: int = 30
    silence_ms: int = 900
    max_seconds: float = 20.0
    vad_threshold: float = 0.0          # 0 -> otomatik kalibrasyon
    # TTS
    tts_engine: str = "edge"            # edge | pyttsx3 | none
    voice: str = "tr-TR-EmelNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    # Davranis
    wake_word: str = ""
    # Hava durumu araci (open-meteo, anahtarsiz)
    weather_enabled: bool = True
    weather_location: str = "Kirklareli,TR"
    weather_timeout: float = 10.0

    def normalize(self) -> "Settings":
        self.base_url = (self.base_url or "http://localhost:1234/v1").strip().rstrip("/")
        self.api_key = (self.api_key or "lm-studio").strip()
        self.model = (self.model or "").strip()
        if not str(self.system_prompt or "").strip():
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.temperature = _as_float(self.temperature, 0.6)
        self.max_tokens = max(64, _as_int(self.max_tokens, 1024))
        self.history_turns = max(1, _as_int(self.history_turns, 8))
        self.request_timeout = _as_float(self.request_timeout, 120.0)
        self.disable_thinking = _as_bool(self.disable_thinking, True)
        self.language = (self.language or "").strip()
        self.whisper_model = (self.whisper_model or "small").strip()
        self.whisper_device = (self.whisper_device or "auto").strip().lower()
        self.whisper_compute_type = (self.whisper_compute_type or "auto").strip().lower()
        self.whisper_gpu = max(0, _as_int(self.whisper_gpu, 0))
        self.sample_rate = max(8000, _as_int(self.sample_rate, WHISPER_RATE))
        self.frame_ms = max(10, _as_int(self.frame_ms, 30))
        self.silence_ms = max(200, _as_int(self.silence_ms, 900))
        self.max_seconds = max(2.0, _as_float(self.max_seconds, 20.0))
        self.vad_threshold = max(0.0, _as_float(self.vad_threshold, 0.0))
        self.tts_engine = (self.tts_engine or "edge").strip().lower()
        self.voice = (self.voice or "tr-TR-EmelNeural").strip()
        self.rate = (self.rate or "+0%").strip()
        self.pitch = (self.pitch or "+0Hz").strip()
        self.volume = (self.volume or "+0%").strip()
        self.wake_word = (self.wake_word or "").strip().lower()
        self.weather_enabled = _as_bool(self.weather_enabled, True)
        self.weather_location = (self.weather_location or "Kirklareli,TR").strip()
        self.weather_timeout = max(2.0, _as_float(self.weather_timeout, 10.0))
        if self.input_device is not None:
            device_index = _as_int(self.input_device, -1)
            self.input_device = None if device_index < 0 else device_index
        return self


def _read_config_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Ayar dosyasi okunamadi (%s): %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        LOG.warning("Ayar dosyasi bir JSON nesnesi olmali: %s", path)
        return {}
    return payload


ENV_MAP = {
    "JARVIS_BASE_URL": "base_url",
    "JARVIS_API_KEY": "api_key",
    "JARVIS_MODEL": "model",
    "JARVIS_SYSTEM_PROMPT": "system_prompt",
    "JARVIS_LANGUAGE": "language",
    "JARVIS_WHISPER_MODEL": "whisper_model",
    "JARVIS_WHISPER_GPU": "whisper_gpu",
    "JARVIS_TTS_ENGINE": "tts_engine",
    "JARVIS_VOICE": "voice",
    "JARVIS_WAKE_WORD": "wake_word",
}


def load_settings(args: argparse.Namespace) -> Settings:
    data: Dict[str, Any] = {}

    candidates: List[Path] = []
    if getattr(args, "config", None):
        candidates.append(Path(args.config))
    else:
        candidates.extend([APP_DIR / "config.json", Path.cwd() / "config.json"])
    for candidate in candidates:
        if candidate.is_file():
            data.update(_read_config_file(candidate))
            LOG.debug("Ayar dosyasi kullanildi: %s", candidate)
            break

    for env_name, key in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value:
            data[key] = value

    overrides = {
        "base_url": getattr(args, "base_url", None),
        "model": getattr(args, "model", None),
        "system_prompt": getattr(args, "system_prompt", None),
        "temperature": getattr(args, "temperature", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "language": getattr(args, "language", None),
        "whisper_model": getattr(args, "whisper_model", None),
        "whisper_device": getattr(args, "whisper_device", None),
        "whisper_compute_type": getattr(args, "whisper_compute_type", None),
        "whisper_gpu": getattr(args, "whisper_gpu", None),
        "input_device": getattr(args, "device", None),
        "silence_ms": getattr(args, "silence_ms", None),
        "max_seconds": getattr(args, "max_seconds", None),
        "tts_engine": getattr(args, "tts_engine", None),
        "voice": getattr(args, "voice", None),
        "wake_word": getattr(args, "wake_word", None),
        "weather_location": getattr(args, "weather_location", None),
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    allowed = {f.name for f in fields(Settings)}
    clean = {k: v for k, v in data.items() if k in allowed}
    settings = Settings(**clean).normalize()

    if getattr(args, "no_tts", False):
        settings.tts_engine = "none"
    if getattr(args, "think", False):
        settings.disable_thinking = False
    if getattr(args, "no_weather", False):
        settings.weather_enabled = False
    return settings


# ---------------------------------------------------------------------------
# Hatalar
# ---------------------------------------------------------------------------
class JarvisError(RuntimeError):
    """Kullaniciya gosterilebilir hata."""


class LLMError(JarvisError):
    pass


class TTSError(JarvisError):
    pass


# ---------------------------------------------------------------------------
# Metin isleme yardimcilari
# ---------------------------------------------------------------------------
_MARKDOWN_RULES = (
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), " "),
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), " "),
    (re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE), " "),
    (re.compile(r"[*_~>|#]+"), " "),
    (re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]"), " "),
)


def strip_for_speech(text: str) -> str:
    """Markdown/emoji temizligi: seslendirilecek duz metin."""
    out = text or ""
    for pattern, replacement in _MARKDOWN_RULES:
        out = pattern.sub(replacement, out)
    out = out.replace("&nbsp;", " ")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\"'”’)\]]?\s+")
MAX_CHUNK = 200


def split_into_chunks(text: str, max_len: int = MAX_CHUNK) -> List[str]:
    """Metni cumlelere, uzun cumleleri de solunum araliklarina boler."""
    cleaned = strip_for_speech(text)
    if not cleaned:
        return []
    chunks: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        sentence = sentence.strip()
        while len(sentence) > max_len:
            cut = sentence.rfind(",", 0, max_len)
            if cut < max_len // 3:
                cut = sentence.rfind(" ", 0, max_len)
            if cut < max_len // 3:
                cut = max_len
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].lstrip(", ").strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def _pop_sentences(buffer: str) -> Tuple[List[str], str]:
    """Stream'den gelen parcalardan tamamlanmis cumleleri cikarir."""
    chunks: List[str] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(buffer):
        piece = buffer[start:match.end()]
        if piece.strip():
            chunks.append(piece)
        start = match.end()
    rest = buffer[start:]
    if len(rest) > MAX_CHUNK:
        cut = rest.rfind(" ", 0, MAX_CHUNK)
        if cut <= 0:
            cut = MAX_CHUNK
        chunks.append(rest[:cut])
        rest = rest[cut:]
    return chunks, rest


def normalize_for_match(text: str) -> str:
    table = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    normalized = (text or "").lower().translate(table)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_EXIT_NORMALIZED = {normalize_for_match(word) for word in EXIT_WORDS}


def is_exit_command(text: str) -> bool:
    normalized = normalize_for_match(text)
    return bool(normalized) and normalized in _EXIT_NORMALIZED


# ---------------------------------------------------------------------------
# Mikrofon + VAD
# ---------------------------------------------------------------------------
class MicrophoneRecorder:
    """Enerji tabanli VAD ile konusma segmentlerini kaydeder."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sample_rate = int(settings.sample_rate)   # gercek kayit orani (acilista guncellenir)
        self.frame_ms = int(settings.frame_ms)
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._noise_floor = 120.0
        self._user_threshold = float(settings.vad_threshold or 0.0)
        self.on_state = lambda state: None  # type: ignore[assignment]

    # -- temel ------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            LOG.debug("ses akisi durumu: %s", status)
        self._queue.put(bytes(indata))

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _open_device(self, rate: int):
        return sd.RawInputStream(
            samplerate=rate,
            blocksize=max(1, int(rate * self.frame_ms / 1000)),
            device=self.settings.input_device,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )

    @contextlib.contextmanager
    def _open_stream(self) -> Iterator[None]:
        if sd is None:
            raise JarvisError(
                "sounddevice kurulu degil. Kurulum: pip install sounddevice\n"
                "Windows'ta bu paket PortAudio'yu da getirir."
            )
        if np is None:
            raise JarvisError("numpy kurulu degil. Kurulum: pip install numpy")

        self._drain_queue()
        candidates: List[int] = []
        for rate in (int(self.sample_rate), WHISPER_RATE, 48000, 44100):
            if rate > 0 and rate not in candidates:
                candidates.append(rate)

        stream = None
        last_error: Optional[Exception] = None
        for rate in candidates:
            try:
                candidate = self._open_device(rate)
                candidate.start()
            except Exception as exc:
                last_error = exc
                LOG.debug("Mikrofon %d Hz ile acilamadi: %s", rate, exc)
                continue
            stream = candidate
            if rate != self.sample_rate:
                LOG.warning("Mikrofon %d Hz desteklemiyor; %d Hz kullanilacak (16 kHz'e cevrilecek).",
                            self.sample_rate, rate)
            self.sample_rate = rate
            break

        if stream is None:
            raise JarvisError(
                "Mikrofon acilamadi: {}\n"
                "  - 'python jarvis.py --list-devices' ile cihaz indeksini bulup --device N verin.\n"
                "  - Windows > Ayarlar > Gizlilik > Mikrofon iznini kontrol edin.".format(last_error)
            )

        try:
            yield
        finally:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self._drain_queue()

    @staticmethod
    def _rms(data: bytes) -> float:
        if not data or np is None:
            return 0.0
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    @property
    def threshold(self) -> float:
        if self._user_threshold > 0:
            return self._user_threshold
        return max(90.0, self._noise_floor * 2.0 + 60.0)

    # -- kalibrasyon -------------------------------------------------
    def calibrate(self, seconds: float = 1.0) -> float:
        """Ortam gurultusunu olcup esigi ayarlar."""
        frames_needed = max(4, int(seconds * 1000 / max(1, self.frame_ms)))
        levels: List[float] = []
        with self._open_stream():
            for _ in range(frames_needed):
                try:
                    data = self._queue.get(timeout=2.0)
                except queue.Empty:
                    break
                levels.append(self._rms(data))
        if levels:
            levels.sort()
            self._noise_floor = levels[len(levels) // 2]
        LOG.info("Gurultu tabani: %.1f  |  konusma esigi: %.1f", self._noise_floor, self.threshold)
        return self.threshold

    # -- kayit -------------------------------------------------------
    def listen(self) -> Optional[Any]:
        """Konusma baslayana kadar bekler; int16 numpy dizisi dondurur (yoksa None)."""
        start_frames = max(1, int(250 / max(1, self.frame_ms)))
        silence_frames = max(1, int(self.settings.silence_ms / max(1, self.frame_ms)))
        max_frames = max(10, int(self.settings.max_seconds * 1000 / max(1, self.frame_ms)))
        preroll_len = max(1, int(300 / max(1, self.frame_ms)))

        preroll: List[bytes] = []
        blocks: List[bytes] = []
        in_speech = False
        speech_run = 0
        silence_run = 0

        self._notify("dinliyor")
        with self._open_stream():
            while True:
                try:
                    data = self._queue.get(timeout=5.0)
                except queue.Empty:
                    LOG.debug("mikrofondan veri gelmiyor (5 sn)")
                    continue
                level = self._rms(data)

                if not in_speech:
                    preroll.append(data)
                    if len(preroll) > preroll_len:
                        preroll.pop(0)
                    if level > self.threshold:
                        speech_run += 1
                        if speech_run >= start_frames:
                            in_speech = True
                            blocks = list(preroll)
                            silence_run = 0
                            self._notify("konusma")
                    else:
                        speech_run = 0
                        self._noise_floor = 0.9 * self._noise_floor + 0.1 * level
                    continue

                blocks.append(data)
                if level <= self.threshold:
                    silence_run += 1
                else:
                    silence_run = 0
                if silence_run >= silence_frames or len(blocks) >= max_frames:
                    break

        self._notify("bekliyor")
        if not blocks:
            return None

        audio = np.frombuffer(b"".join(blocks), dtype=np.int16).copy()
        if audio.size < int(self.sample_rate * 0.3):
            LOG.debug("Kayit cok kisa (%.2f sn), yoksayildi.", audio.size / float(self.sample_rate))
            return None
        LOG.debug("Kayit: %.2f sn @ %d Hz", audio.size / float(self.sample_rate), self.sample_rate)
        return audio

    def _notify(self, state: str) -> None:
        try:
            self.on_state(state)
        except Exception:
            pass


def list_audio_devices() -> int:
    if sd is None:
        print("sounddevice kurulu degil: pip install sounddevice")
        return 1
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in, _default_out = sd.default.device
    except Exception as exc:
        print("Cihazlar okunamadi: {}".format(exc))
        return 1

    print("Girdi (mikrofon) cihazlari:")
    found = False
    for index, device in enumerate(devices):
        try:
            channels = int(device.get("max_input_channels", 0))
        except Exception:
            channels = 0
        if channels <= 0:
            continue
        found = True
        api_index = int(device.get("hostapi", -1))
        api_name = hostapis[api_index]["name"] if 0 <= api_index < len(hostapis) else "?"
        marker = "  <-- varsayilan" if index == default_in else ""
        print("  {:2d}  {}  [{}]{}".format(index, device.get("name", "?"), api_name, marker))
    if not found:
        print("  (girdi cihazi bulunamadi)")
    return 0


# ---------------------------------------------------------------------------
# STT: faster-whisper
# ---------------------------------------------------------------------------
class Transcriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._device = ""
        self._device_index = 0
        self._lock = threading.Lock()

    @property
    def device_label(self) -> str:
        """Ornek: 'cuda:1', 'cpu'."""
        if (self._device or "").startswith("cuda"):
            return "{}:{}".format(self._device, self._device_index)
        return self._device or "?"

    def _resolve_gpu_index(self) -> int:
        """Ayarlanan GPU indeksini dogrular; kart yoksa GPU 0'a duser."""
        index = int(self.settings.whisper_gpu or 0)
        if index <= 0:
            return 0
        count = count_cuda_devices()
        if count == 0 or index >= count:
            LOG.warning(
                "whisper_gpu=%d ayarli ama sistemde %d CUDA aygiti gorunuyor; GPU 0 kullanilacak.",
                index,
                count,
            )
            return 0
        return index

    def _build(self, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel

        _prepare_cuda_dlls()  # cuBLAS/cuDNN (varsa) gorunur olsun
        device_index = self._resolve_gpu_index() if device.startswith("cuda") else 0
        LOG.info(
            "Whisper '%s' yukleniyor (%s:%d / %s) ...",
            self.settings.whisper_model,
            device,
            device_index,
            compute,
        )
        self._model = WhisperModel(
            self.settings.whisper_model,
            device=device,
            device_index=device_index,
            compute_type=compute,
        )
        self._device = device
        self._device_index = device_index

    def _self_test(self) -> None:
        """1 sn sessizlikle gercek bir cikarim yapar.

        CUDA kutuphaneleri (cublas/cudnn) eksikse hata *ilk transkripsiyonda*
        degil, model yuklendikten sonra ortaya cikar; bu yuzden burada dogrulariz.
        """
        if np is None or self._model is None:
            return
        probe = np.zeros(WHISPER_RATE, dtype=np.float32)
        segments, _info = self._model.transcribe(
            probe,
            language=self.settings.language or None,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        for _segment in segments:  # uretici tembellestirilmis: gercek is burada baslar
            break

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel  # noqa: F401
            except Exception as exc:  # pragma: no cover
                raise JarvisError(
                    "faster-whisper kurulu degil. Kurulum: pip install -r requirements.txt\n"
                    "Detay: {}".format(exc)
                ) from exc

            device = self.settings.whisper_device or "auto"
            compute = self.settings.whisper_compute_type or "auto"
            if device == "auto":
                attempts = [("cuda", "float16" if compute == "auto" else compute), ("cpu", "int8")]
            elif device == "cpu":
                attempts = [("cpu", "int8" if compute == "auto" else compute)]
            else:
                # Acikca baska bir aygit istendiyse bile CPU son care olarak denenir
                attempts = [(device, "int8" if compute == "auto" else compute), ("cpu", "int8")]

            last_error: Optional[Exception] = None
            for dev, comp in attempts:
                try:
                    self._build(dev, comp)
                    self._self_test()
                except Exception as exc:
                    last_error = exc
                    self._model = None
                    self._device = ""
                    LOG.debug("Whisper denemesi basarisiz (%s / %s): %s", dev, comp, exc)
                    if dev != "cpu":
                        LOG.warning(
                            "%s uzerinde calismadi (%s). CPU'ya geciliyor...",
                            dev.upper(),
                            str(exc).splitlines()[0][:160],
                        )
                    continue
                LOG.info("Whisper hazir (%s / %s).", self.device_label, comp)
                return

            raise JarvisError(
                "Whisper modeli yuklenemedi: {}\n"
                "  - Ilk calistirmada model internetten indirilir (tek seferlik).\n"
                "  - Daha kucuk model deneyin: --whisper-model base\n"
                "  - GPU sorunluysa: --whisper-device cpu".format(last_error)
            )

    def _transcribe_once(self, pcm: Any) -> str:
        segments, _info = self._model.transcribe(
            pcm,
            language=self.settings.language or None,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            temperature=0.0,
        )
        parts = []
        for segment in segments:
            text_piece = (segment.text or "").strip()
            if not text_piece:
                continue
            # Sessizlikte olusan halusinasyonlari ele
            if float(getattr(segment, "no_speech_prob", 0.0) or 0.0) > 0.75:
                continue
            if float(getattr(segment, "avg_logprob", 0.0) or 0.0) < -1.2:
                continue
            parts.append(text_piece)
        return " ".join(parts).strip()

    def transcribe(self, audio: Any, sample_rate: int = WHISPER_RATE) -> str:
        self.load()
        if self._model is None:  # pratikte olmaz, tip kontrolu icin
            raise JarvisError("Whisper modeli yuklu degil.")

        pcm = to_whisper_pcm(audio, sample_rate)
        try:
            return self._transcribe_once(pcm)
        except Exception as exc:
            if self._device != "cpu":
                # Ornegin VRAM yetmedi ya da cuBLAS/cuDNN kutuphanesi eksik
                LOG.warning(
                    "%s ile yaziya cevirme basarisiz (%s); CPU'ya geciliyor.",
                    self.device_label.upper(),
                    str(exc).splitlines()[0][:160],
                )
                with self._lock:
                    self._model = None
                    self._device = ""
                try:
                    self._build("cpu", "int8")
                    self._self_test()
                except Exception as rebuild_exc:
                    raise JarvisError(
                        "Yaziya cevirme basarisiz: {}".format(rebuild_exc)
                    ) from rebuild_exc
                try:
                    return self._transcribe_once(pcm)
                except Exception as cpu_exc:
                    raise JarvisError(
                        "Yaziya cevirme basarisiz (CPU): {}".format(cpu_exc)
                    ) from cpu_exc
            raise JarvisError("Yaziya cevirme basarisiz: {}".format(exc)) from exc


# ---------------------------------------------------------------------------
# Hava durumu araci (open-meteo.com: anahtarsiz, ucretsiz)
# ---------------------------------------------------------------------------
WMO_DESCRIPTIONS = {
    0: "açık",
    1: "az bulutlu",
    2: "parçalı bulutlu",
    3: "kapalı",
    45: "sisli",
    48: "kırağılı sis",
    51: "hafif çisenti",
    53: "çisenti",
    55: "yoğun çisenti",
    56: "dondurucu çisenti",
    57: "yoğun dondurucu çisenti",
    61: "hafif yağmurlu",
    63: "yağmurlu",
    65: "şiddetli yağmurlu",
    66: "dondurucu yağmurlu",
    67: "şiddetli dondurucu yağmurlu",
    71: "hafif kar yağışlı",
    73: "kar yağışlı",
    75: "yoğun kar yağışlı",
    77: "kar taneleri",
    80: "hafif sağanak yağışlı",
    81: "sağanak yağışlı",
    82: "şiddetli sağanak yağışlı",
    85: "hafif kar sağanağı",
    86: "yoğun kar sağanağı",
    95: "gök gürültülü fırtına",
    96: "dolulu fırtına",
    99: "şiddetli dolulu fırtına",
}

# Yerel niyet tespiti (LLM arac cagiramazsa yedek olarak da kullanilir)
WEATHER_KEYWORDS = (
    "hava", "sicaklik", "sıcaklık", "derece", "yagmur", "yağmur", "kar yag", "kar yağ",
    "ruzgar", "rüzgar", "nem ", "tahmin", "bulut", "gunes", "güneş", "don ", "sis ",
    "weather", "forecast",
)


def describe_weather_code(code: Any) -> str:
    """WMO hava durumu kodunu Turkce aciklamaya cevirir."""
    try:
        return WMO_DESCRIPTIONS.get(int(code), "bilinmiyor")
    except (TypeError, ValueError):
        return "bilinmiyor"


def is_weather_question(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in WEATHER_KEYWORDS)


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    try:
        return None if value is None or value == "" else int(round(float(value)))
    except (TypeError, ValueError):
        return None


@dataclass
class WeatherReport:
    location: str
    day: str
    description: str
    temperature: float
    apparent: Optional[float] = None
    humidity: Optional[int] = None
    wind_kmh: Optional[float] = None
    precipitation: Optional[float] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    rain_probability: Optional[int] = None
    observed_at: str = ""

    def as_tool_result(self) -> Dict[str, Any]:
        """LLM'e (tool mesaji olarak) verilecek kisa JSON."""
        data: Dict[str, Any] = {
            "konum": self.location,
            "gun": self.day,
            "durum": self.description,
            "sicaklik_c": self.temperature,
        }
        optional = {
            "hissedilen_c": self.apparent,
            "nem_yuzde": self.humidity,
            "ruzgar_kmh": self.wind_kmh,
            "yagis_mm": self.precipitation,
            "en_dusuk_c": self.t_min,
            "en_yuksek_c": self.t_max,
            "yagis_olasiligi_yuzde": self.rain_probability,
            "gozlem": self.observed_at,
        }
        for key, value in optional.items():
            if value is not None and value != "":
                data[key] = value
        return data

    def summary(self) -> str:
        """Konsol/--weather ciktisi icin okunabilir ozet."""
        daily = (
            self.t_min is not None
            and self.t_max is not None
            and abs(float(self.t_max) - float(self.temperature)) < 0.01
        )
        if daily:
            head = "{}: en yüksek {}°C, en düşük {}°C, {}".format(
                self.location, self.t_max, self.t_min, self.description
            )
        else:
            head = "{}: {}°C, {}".format(self.location, self.temperature, self.description)

        details: List[str] = []
        if not daily and self.t_min is not None and self.t_max is not None:
            details.append("en düşük {}°C / en yüksek {}°C".format(self.t_min, self.t_max))
        if self.apparent is not None:
            details.append("hissedilen {}°C".format(self.apparent))
        if self.humidity is not None:
            details.append("nem %{}".format(self.humidity))
        if self.wind_kmh is not None:
            details.append("rüzgar {} km/s".format(self.wind_kmh))
        if self.rain_probability is not None:
            details.append("yağış olasılığı %{}".format(self.rain_probability))
        return head + (" (" + " | ".join(details) + ")" if details else "")


class WeatherService:
    """open-meteo uzerinden hava durumu; API anahtari gerekmez.

    Konum: sehir adi ("Istanbul,TR"), koordinat ("41.73,27.22") veya bos
    (varsayilan: settings.weather_location -> Kirklareli/TR).
    """

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    DEFAULT_COORDS = (41.73508, 27.22521)   # Kirklareli merkez
    DEFAULT_LABEL = "Kırklareli, Türkiye"
    TIMEZONE = "Europe/Istanbul"
    COORD_RE = re.compile(r"^\(?\s*(-?\d{1,3}(?:[.,]\d+)?)\s*[,;]\s*(-?\d{1,3}(?:[.,]\d+)?)\s*\)?$")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: Dict[str, Tuple[str, float, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _require_requests() -> None:
        if requests is None:
            raise JarvisError("requests kurulu degil. Kurulum: pip install requests")

    def resolve(self, location: Optional[str] = None) -> Tuple[str, float, float]:
        """(etiket, enlem, boylam) dondurur; sonuclar oturum boyunca onbelleklenir."""
        raw = (location or "").strip() or self.settings.weather_location or "Kirklareli,TR"
        key = raw.lower()
        with self._lock:
            cached = self._cache.get(key)
        if cached:
            return cached

        match = self.COORD_RE.match(raw)
        if match:
            # "41.73,27.22" gibi koordinatlar icin ag gerekmez
            lat = float(match.group(1).replace(",", "."))
            lon = float(match.group(2).replace(",", "."))
            resolved = ("{:.4f}, {:.4f}".format(lat, lon), lat, lon)
        else:
            self._require_requests()
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            params: Dict[str, Any] = {
                "name": parts[0],
                "count": 1,
                "language": "tr",
                "format": "json",
            }
            if len(parts) > 1 and len(parts[1]) == 2:
                params["countryCode"] = parts[1].upper()
            try:
                response = requests.get(
                    self.GEO_URL, params=params, timeout=float(self.settings.weather_timeout)
                )
                response.raise_for_status()
                payload = json.loads(response.content.decode("utf-8", "replace"))
            except Exception as exc:
                raise JarvisError(
                    "Konum aranamadi ({}): {}".format(raw, exc)
                ) from exc
            results = payload.get("results") if isinstance(payload, dict) else None
            if not results:
                raise JarvisError(
                    "'{}' konumu bulunamadi. Ornek: 'Istanbul,TR' veya '41.73,27.22'".format(raw)
                )
            first = results[0]
            lat = float(first["latitude"])
            lon = float(first["longitude"])
            unique_parts: List[str] = []
            for part in (
                first.get("name"),
                first.get("admin1"),
                first.get("country_code") or first.get("country"),
            ):
                if not part:
                    continue
                text_part = str(part).strip()
                if not text_part or text_part.lower() in [item.lower() for item in unique_parts]:
                    continue
                unique_parts.append(text_part)
            resolved = (", ".join(unique_parts) or raw, lat, lon)

        with self._lock:
            self._cache[key] = resolved
        LOG.debug("Konum cozuldu: %s -> %.4f, %.4f", resolved[0], resolved[1], resolved[2])
        return resolved

    def get(self, location: Optional[str] = None, day: str = "bugun") -> WeatherReport:
        self._require_requests()
        day_key = (day or "bugun").strip().lower()
        tomorrow = day_key in ("yarin", "yarın", "tomorrow", "yarina", "yarına")
        label, lat, lon = self.resolve(location)

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "timezone": self.TIMEZONE,
            "forecast_days": 2,
        }
        try:
            response = requests.get(
                self.FORECAST_URL, params=params, timeout=float(self.settings.weather_timeout)
            )
            response.raise_for_status()
            payload = json.loads(response.content.decode("utf-8", "replace"))
        except Exception as exc:
            raise JarvisError("Hava durumu alinamadi: {}".format(exc)) from exc

        current = payload.get("current") or {}
        daily = payload.get("daily") or {}

        def daily_value(key: str, index: int) -> Any:
            values = daily.get(key) or []
            return values[index] if index < len(values) else None

        if tomorrow:
            t_max = _as_optional_float(daily_value("temperature_2m_max", 1))
            t_min = _as_optional_float(daily_value("temperature_2m_min", 1))
            return WeatherReport(
                location=label,
                day="yarin",
                description=describe_weather_code(daily_value("weather_code", 1)),
                temperature=float(t_max if t_max is not None else 0.0),
                t_min=t_min,
                t_max=t_max,
                rain_probability=_as_optional_int(daily_value("precipitation_probability_max", 1)),
                observed_at=str(daily_value("time", 1) or ""),
            )

        temperature = _as_optional_float(current.get("temperature_2m"))
        if temperature is None:
            temperature = _as_optional_float(daily_value("temperature_2m_max", 0))
        return WeatherReport(
            location=label,
            day="bugun",
            description=describe_weather_code(
                current.get("weather_code", daily_value("weather_code", 0))
            ),
            temperature=float(temperature if temperature is not None else 0.0),
            apparent=_as_optional_float(current.get("apparent_temperature")),
            humidity=_as_optional_int(current.get("relative_humidity_2m")),
            wind_kmh=_as_optional_float(current.get("wind_speed_10m")),
            precipitation=_as_optional_float(current.get("precipitation")),
            t_min=_as_optional_float(daily_value("temperature_2m_min", 0)),
            t_max=_as_optional_float(daily_value("temperature_2m_max", 0)),
            rain_probability=_as_optional_int(daily_value("precipitation_probability_max", 0)),
            observed_at=str(current.get("time") or ""),
        )


def weather_tool_schema() -> List[Dict[str, Any]]:
    """LM Studio'ya gonderilen OpenAI uyumlu tool tanimi."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": (
                    "Guncel hava durumunu ve gunluk en dusuk/en yuksek sicakligi dondurur. "
                    "Kullanici konum soylemezse varsayilan konum Kirklareli, Turkiye kullanilir."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "konum": {
                            "type": "string",
                            "description": "Sehir adi, orn. 'Istanbul,TR'. Bos birakilirsa varsayilan konum.",
                        },
                        "gun": {
                            "type": "string",
                            "enum": ["bugun", "yarin"],
                            "description": "Hangi gun icin tahmin isteniyor.",
                        },
                    },
                    "required": [],
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# LLM: LM Studio (OpenAI uyumlu)
# ---------------------------------------------------------------------------
@dataclass
class LLMEvent:
    """Akistan gelen tek bir olay: icerik, dusunce veya arac cagrisi."""

    kind: str                 # "content" | "reasoning" | "tool_call"
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    tool_arguments: str = ""


def _connection_hint(base_url: str, exc: Exception) -> str:
    return (
        "LM Studio sunucusuna ulasilamadi ({}).\n"
        "  1) LM Studio'yu acin.\n"
        "  2) 'Developer' (Local Server) sekmesinde 'Start Server' deyin.\n"
        "  3) Bir chat/instruct modeli yukleyin; port 1234 olmali.\n"
        "  4) Farkli adres icin: --base-url http://localhost:1234/v1\n"
        "  Teknik detay: {}".format(base_url, exc)
    )


def _decode_json_body(response) -> Any:
    try:
        return json.loads(response.content.decode("utf-8", "replace"))
    except Exception as exc:
        raise LLMError("LM Studio gecersiz JSON dondurdu: {}".format(exc)) from exc


class LMStudioClient:
    """LM Studio'nun yerel OpenAI API'si ile konusur."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")
        self._model: Optional[str] = None

    # -- altyapi -----------------------------------------------------
    @staticmethod
    def _ensure_requests() -> None:
        if requests is None:
            raise LLMError("requests kurulu degil. Kurulum: pip install requests")

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": "Bearer {}".format(self.settings.api_key or "lm-studio"),
        }

    def list_models(self) -> List[str]:
        self._ensure_requests()
        try:
            response = requests.get(self.base_url + "/models", headers=self._headers(), timeout=10)
            response.raise_for_status()
        except Exception as exc:
            raise LLMError(_connection_hint(self.base_url, exc)) from exc

        payload = _decode_json_body(response)
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return []
        return [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]

    # Embedding/rerank modelleri sohbet icin kullanilamaz
    NON_CHAT_HINTS = ("embed", "rerank", "bge-", "gte-", "e5-", "minilm")

    def chat_models(self) -> List[str]:
        return [m for m in self.list_models() if not any(h in m.lower() for h in self.NON_CHAT_HINTS)]

    @staticmethod
    def _preferred(models: List[str]) -> str:
        """Instruct/chat modellerini tercih eder; uncensored/RP turevlerini geri plana atar."""
        preferred_hints = ("instruct", "chat", "it-")
        demoted_hints = ("uncensored", "abliterated", "heretic", "roleplay", "rp-")

        def score(name: str) -> int:
            low = name.lower()
            value = 0
            if any(hint in low for hint in preferred_hints):
                value += 2
            if any(hint in low for hint in demoted_hints):
                value -= 1
            return value

        best = models[0]
        best_score = score(best)
        for candidate in models[1:]:
            candidate_score = score(candidate)
            if candidate_score > best_score:
                best, best_score = candidate, candidate_score
        return best

    def resolve_model(self) -> str:
        if self._model:
            return self._model
        requested = (self.settings.model or "").strip()
        if requested and requested.lower() not in ("auto", "default", "otomatik"):
            self._model = requested
            return self._model
        models = self.chat_models()
        if not models:
            raise LLMError(
                "LM Studio'da kullanilabilir sohbet modeli bulunamadi.\n"
                "  LM Studio icinde bir instruct/chat modeli yukleyip sunucuyu yeniden baslatin."
            )
        self._model = self._preferred(models)
        LOG.info("Model otomatik secildi: %s (adaylar: %s)", self._model, ", ".join(models))
        return self._model

    # -- sohbet ------------------------------------------------------
    @staticmethod
    def _with_hint(messages: List[Dict[str, str]], hint: str) -> List[Dict[str, str]]:
        if not hint:
            return messages
        prepared = [dict(message) for message in messages]
        for message in reversed(prepared):
            if message.get("role") == "user":
                message["content"] = "{}{}".format(message.get("content", ""), hint)
                break
        return prepared

    def _iter_events(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        disable_thinking: bool,
        hint: str = "",
        use_tools: bool = True,
    ) -> Iterator[LLMEvent]:
        """Akistan icerik/dusunce olaylari ve (varsa) arac cagrilarini uretir."""
        self._ensure_requests()
        payload: Dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": self._with_hint(messages, hint),
            "temperature": float(self.settings.temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if disable_thinking:
            # Qwen3 vb. modellerde dusunme blogunu kapatir (desteklemeyen modellerde yok sayilir)
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if use_tools and self.settings.weather_enabled:
            payload["tools"] = weather_tool_schema()
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(
                self.base_url + "/chat/completions",
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=(15, float(self.settings.request_timeout)),
            )
        except Exception as exc:
            raise LLMError(_connection_hint(self.base_url, exc)) from exc

        # Arac cagrisi argumanlari parca parca gelir; index'e gore birlestirilir
        pending: Dict[int, Dict[str, str]] = {}

        with response:
            if response.status_code >= 400:
                detail = response.content.decode("utf-8", "replace")[:400]
                raise LLMError(
                    "LM Studio {} dondurdu (model: {}).\n{}".format(
                        response.status_code, payload["model"], detail
                    )
                )
            for raw in response.iter_lines(chunk_size=4096):
                if not raw:
                    continue
                # UTF-8 elle cozulur: charset belirtilmeyen yanitlarda bozulmayi onler
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    LOG.debug("SSE satiri atlandi: %s", line[:120])
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    raise LLMError("LM Studio hatasi: {}".format(event.get("error")))
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    yield LLMEvent("reasoning", text=str(reasoning))

                piece = delta.get("content")
                if piece:
                    yield LLMEvent("content", text=str(piece))

                for tool_call in delta.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    index = _as_int(tool_call.get("index"), 0)
                    slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tool_call.get("id"):
                        slot["id"] = str(tool_call["id"])
                    function = tool_call.get("function") or {}
                    if function.get("name"):
                        slot["name"] = str(function["name"])
                    if function.get("arguments"):
                        slot["arguments"] += str(function["arguments"])

        for index in sorted(pending):
            slot = pending[index]
            if not slot["name"]:
                continue
            LOG.debug("Arac cagrisi toplandi: %s %s", slot["name"], slot["arguments"])
            yield LLMEvent(
                "tool_call",
                tool_id=slot["id"] or "call_{}".format(index),
                tool_name=slot["name"],
                tool_arguments=slot["arguments"] or "{}",
            )

    def _content_round(
        self,
        messages: List[Dict[str, str]],
        tool_calls: List[LLMEvent],
        use_tools: bool = True,
    ) -> Iterator[str]:
        """Tek tur icin icerik uretir (reasoning modellerine karsi 2 deneme)."""
        base_tokens = int(self.settings.max_tokens)
        retry_tokens = min(max(base_tokens * 3, 1024), 4096)
        attempts = [
            (base_tokens, bool(self.settings.disable_thinking), "", use_tools),
            (retry_tokens, True, " /no_think", use_tools),
        ]

        for index, (tokens, disable_thinking, hint, with_tools) in enumerate(attempts):
            content_chars = 0
            reasoning_chars = 0
            if index > 0:
                LOG.warning(
                    "Yanit bos kaldi, %d token butcesi ve dusunme-kapali modla tekrar deneniyor...", tokens
                )
            for event in self._iter_events(messages, tokens, disable_thinking, hint, with_tools):
                if event.kind == "tool_call":
                    tool_calls.append(event)
                    continue
                if event.kind == "reasoning":
                    reasoning_chars += len(event.text)
                    continue
                content_chars += len(event.text)
                yield event.text
            if content_chars or tool_calls:
                return
            if not reasoning_chars and index > 0:
                return

    def _tool_followup(
        self,
        conversation: List[Dict[str, str]],
        tool_calls: List[LLMEvent],
        tool_runner: Any,
    ) -> List[Dict[str, str]]:
        """Arac cagrilarini calistirip sonuclari konusmaya ekler."""
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.tool_id,
                    "type": "function",
                    "function": {"name": call.tool_name, "arguments": call.tool_arguments},
                }
                for call in tool_calls
            ],
        }
        tool_messages: List[Dict[str, Any]] = []
        for call in tool_calls:
            LOG.debug("Arac cagriliyor: %s(%s)", call.tool_name, call.tool_arguments)
            try:
                result = tool_runner(call.tool_name, call.tool_arguments)
                content = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as exc:
                LOG.warning("Arac hatasi (%s): %s", call.tool_name, exc)
                content = json.dumps({"hata": str(exc)}, ensure_ascii=False)
            if len(content) > 2000:
                content = content[:2000] + "..."
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_id,
                    "name": call.tool_name,
                    "content": content,
                }
            )
        return list(conversation) + [assistant_message] + tool_messages

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tool_runner: Optional[Callable[[str, str], Any]] = None,
    ) -> Iterator[str]:
        """Nihai cevabi parca parca uretir; model arac cagirirsa calistirip devam eder.

        Bazi modeller (Qwen3 vb.) yalnizca 'reasoning_content' dondurur; o durumda
        daha buyuk token butcesi ve /no_think ipucuyla bir kez daha denenir.
        """
        conversation = list(messages)
        produced = 0

        for round_index in range(3):
            tool_calls: List[LLMEvent] = []
            for piece in self._content_round(conversation, tool_calls, use_tools=(round_index == 0)):
                produced += len(piece)
                yield piece

            if produced:
                return
            if tool_calls and tool_runner:
                conversation = self._tool_followup(conversation, tool_calls, tool_runner)
                continue
            break

        raise LLMError(
            "Model nihai cevap uretmedi (yalnizca 'reasoning' dondurdu).\n"
            "  Cozum onerileri:\n"
            "  - LM Studio'da dusunmeyen bir instruct model yukleyin (orn. Qwen2.5-Instruct, Llama-3.1-Instruct, gemma-3-it)\n"
            "  - veya --model <baska-model> ile secin ('--list-models' adaylari gosterir)\n"
            "  - veya --think ile dusunmeyi acip --max-tokens degerini artirin"
        )

    def check(self) -> str:
        models = self.chat_models()
        model = self.resolve_model()
        extra = "" if len(models) <= 1 else " ({} sohbet modeli yuklu)".format(len(models))
        return "{}{}".format(model, extra)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Ayni is parcaciginda tekrar kullanilabilen bir event loop uzerinde calistirir."""
    loop = getattr(_LOCAL, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _LOCAL.loop = loop
    return loop.run_until_complete(coro)


class AudioPlayer:
    """mp3/wav calar: pygame -> ffplay."""

    def __init__(self) -> None:
        self._pygame = None
        try:
            with _silence_output():
                import pygame  # type: ignore
            self._pygame = pygame
        except Exception:
            self._pygame = None
        self._ffplay = shutil.which("ffplay")
        self._ready = False
        # SDL/mixer ana is parcaciginda baslatilir (thread sorunlarini onler)
        self._ensure_pygame()

    @property
    def available(self) -> bool:
        return self._ready or bool(self._ffplay)

    def describe(self) -> str:
        if self._ready:
            return "pygame"
        if self._ffplay:
            return "ffplay"
        return "yok"

    def _ensure_pygame(self) -> bool:
        if self._pygame is None:
            return False
        if self._ready:
            return True
        try:
            with _silence_output():
                self._pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._ready = True
            return True
        except Exception as exc:
            LOG.debug("pygame mixer baslatilamadi: %s", exc)
            self._pygame = None
            return False

    def play(self, path: str, stop_event: Optional[threading.Event] = None) -> None:
        if self._ensure_pygame():
            self._play_pygame(path, stop_event)
            return
        if self._ffplay:
            self._play_ffplay(path, stop_event)
            return
        raise TTSError(
            "Ses calma araci yok. Cozum: pip install pygame (veya ffmpeg kurup PATH'e ekleyin)."
        )

    def _play_pygame(self, path: str, stop_event: Optional[threading.Event]) -> None:
        assert self._pygame is not None
        mixer = self._pygame.mixer
        try:
            mixer.music.load(path)
            mixer.music.play()
        except Exception as exc:
            raise TTSError("Ses dosyasi calinamadi: {}".format(exc)) from exc
        try:
            while mixer.music.get_busy():
                if stop_event is not None and stop_event.is_set():
                    mixer.music.stop()
                    break
                self._pygame.time.wait(40)
        finally:
            try:
                mixer.music.unload()
            except Exception:
                pass

    def _play_ffplay(self, path: str, stop_event: Optional[threading.Event]) -> None:
        process = subprocess.Popen(
            [self._ffplay or "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            while process.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    process.terminate()
                    break
                time.sleep(0.04)
        finally:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=2)
            except Exception:
                pass

    def close(self) -> None:
        if self._pygame is not None and self._ready:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass
            self._ready = False


class EdgeTTSEngine:
    """Microsoft edge-tts: cok dogal Turkce sesler (internet gerekir).

    Not: Microsoft bazi sesleri ucretsiz uctan kaldirabiliyor (orn. tr-TR-AhmetNeural
    su an ses uretmiyor). Bu yuzden sentez basarisiz olursa sirasiyla yedek sesler
    denenir ve calisan ses oturum boyunca hatirlanir.
    """

    name = "edge-tts"
    FALLBACK_VOICES = (
        "tr-TR-EmelNeural",
        "en-US-AvaMultilingualNeural",
        "en-US-EmmaMultilingualNeural",
        "de-DE-SeraphinaMultilingualNeural",
    )

    def __init__(self, settings: Settings) -> None:
        try:
            import edge_tts  # noqa: F401
        except Exception as exc:
            raise TTSError("edge-tts kurulu degil. Kurulum: pip install edge-tts") from exc
        self.settings = settings
        self.player = AudioPlayer()
        if not self.player.available:
            raise TTSError(
                "edge-tts sesi uretir ama calmak icin pygame/ffplay gerekiyor. "
                "Kurulum: pip install pygame"
            )
        self._voice = settings.voice
        self._tmpdir: Optional[str] = None
        self._counter = 0
        self._lock = threading.Lock()

    def _tmp_dir(self) -> str:
        if self._tmpdir is None or not os.path.isdir(self._tmpdir):
            self._tmpdir = tempfile.mkdtemp(prefix="jarvis_tts_")
        return self._tmpdir

    def synthesize(self, text: str) -> str:
        import edge_tts

        candidates: List[str] = []
        for voice in (self._voice,) + self.FALLBACK_VOICES:
            if voice and voice not in candidates:
                candidates.append(voice)

        last_error: Optional[Exception] = None
        for voice in candidates:
            with self._lock:
                self._counter += 1
                index = self._counter
            path = os.path.join(self._tmp_dir(), "part_{:04d}.mp3".format(index))

            async def _save(voice_name: str = voice, target: str = path) -> None:
                communicate = edge_tts.Communicate(
                    text,
                    voice_name,
                    rate=self.settings.rate,
                    volume=self.settings.volume,
                    pitch=self.settings.pitch,
                )
                await communicate.save(target)

            try:
                _run_async(_save())
            except Exception as exc:
                last_error = exc
                LOG.debug("edge-tts sesi basarisiz (%s): %s", voice, exc)
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue

            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                last_error = TTSError("bos ses dosyasi")
                continue

            if voice != self._voice:
                LOG.warning("edge-tts '%s' ses uretmedi; '%s' sesine gecildi.", self._voice, voice)
                self._voice = voice
            return path

        raise TTSError(
            "edge-tts sentezleme hatasi: {}\n"
            "  Ipucu: internet baglantisini kontrol edin, farkli bir --voice deneyin "
            "(orn. tr-TR-EmelNeural) ya da cevrimdisi mod: --tts pyttsx3".format(last_error)
        )

    def speak(self, text: str, stop_event: Optional[threading.Event] = None) -> None:
        if stop_event is not None and stop_event.is_set():
            return
        path = self.synthesize(text)
        try:
            if stop_event is not None and stop_event.is_set():
                return
            self.player.play(path, stop_event)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def stop(self) -> None:
        # calma dongusu stop_event'i izler; ek islem gerekmez
        return None

    def close(self) -> None:
        self.player.close()
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None


class Pyttsx3Engine:
    """Windows SAPI5 / eSpeak: tamamen cevrimdisi, hafif seslendirme.

    Motor bilerek *kullanildigi* is parcaciginda (TTS worker) baslatilir:
    SAPI5/COM nesneleri onlari olusturan thread'e baglidir.
    """

    name = "pyttsx3"

    def __init__(self, settings: Settings) -> None:
        try:
            import pyttsx3
        except Exception as exc:
            raise TTSError("pyttsx3 kurulu degil. Kurulum: pip install pyttsx3") from exc
        self.settings = settings
        self._pyttsx3 = pyttsx3
        self._engine = None
        self._voice_name: Optional[str] = None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        try:  # COM'u bu is parcaciginda baslat (SAPI5 icin gerekli)
            import comtypes  # type: ignore

            comtypes.CoInitialize()
        except Exception:
            pass
        try:
            self._engine = self._pyttsx3.init()
        except Exception as exc:
            raise TTSError("pyttsx3 baslatilamadi: {}".format(exc)) from exc
        self._engine.setProperty("rate", 175)
        self._engine.setProperty("volume", 1.0)
        self._voice_name = self._pick_voice()
        LOG.info("pyttsx3 sesi: %s", self._voice_name)
        return self._engine

    def _pick_voice(self) -> str:
        wanted = normalize_for_match(self.settings.language or "tr") or "tr"
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:
            voices = []
        for voice in voices:
            blob = normalize_for_match("{} {}".format(getattr(voice, "id", ""), getattr(voice, "name", "")))
            languages = getattr(voice, "languages", None) or []
            lang_blob = " ".join(
                item.decode("utf-8", "ignore") if isinstance(item, bytes) else str(item) for item in languages
            ).lower()
            if wanted in blob or "turkish" in blob or wanted in lang_blob:
                try:
                    self._engine.setProperty("voice", voice.id)
                except Exception:
                    continue
                return str(getattr(voice, "name", voice.id))
        LOG.warning("pyttsx3: Turkce ses bulunamadi; sistem varsayilani kullanilacak.")
        return "varsayilan"

    def speak(self, text: str, stop_event: Optional[threading.Event] = None) -> None:
        if stop_event is not None and stop_event.is_set():
            return
        engine = self._ensure_engine()
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            raise TTSError("pyttsx3 hatasi: {}".format(exc)) from exc
        if stop_event is not None and stop_event.is_set():
            try:
                engine.stop()
            except Exception:
                pass

    def stop(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.stop()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        self._engine = None


class NullTTSEngine:
    name = "kapali"

    def speak(self, text: str, stop_event: Optional[threading.Event] = None) -> None:
        LOG.debug("TTS kapali, metin: %s", text)

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


class Speaker:
    """TTS motorunu ayri bir is parcaciginda calistirir.

    Cumleler kuyruga eklenir; motor sentezleyip calar. Boylece LLM hala
    yanit uretirken seslendirme coktan baslamis olur (dusuk gecikme).
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.engine_name = getattr(engine, "name", type(engine).__name__)
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop = threading.Event()
        self._closed = False
        self._fail_streak = 0
        self._thread = threading.Thread(target=self._worker, name="jarvis-tts", daemon=True)
        self._thread.start()

    # -- dis API -----------------------------------------------------
    def feed(self, text: str) -> None:
        if self._closed:
            return
        for chunk in split_into_chunks(text):
            if self._stop.is_set():
                return
            self._queue.put(chunk)

    def finish(self, timeout: Optional[float] = None) -> None:
        """Kuyruktaki tum cumleler calinana kadar bekler."""
        if self._closed or self._stop.is_set():
            return
        if timeout is None:
            self._queue.join()
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return
            time.sleep(0.02)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.engine.stop()
        except Exception:
            pass
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def resume(self) -> None:
        self._stop.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        try:
            self.engine.close()
        except Exception:
            pass

    # -- ic isleyis --------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._stop.is_set():
                    continue
                try:
                    self.engine.speak(item, self._stop)
                    self._fail_streak = 0
                except Exception as exc:
                    self._fail_streak += 1
                    if self._fail_streak == 1:
                        LOG.error("Seslendirme hatasi: %s", exc)
                        LOG.error(
                            "Ipucu: edge-tts icin internet gerekir; cevrimdisi: --tts pyttsx3 ; "
                            "sessiz calismak icin: --no-tts"
                        )
                    else:
                        LOG.debug("Seslendirme hatasi (tekrar %d): %s", self._fail_streak, exc)
                    if self._fail_streak >= 3 and not isinstance(self.engine, NullTTSEngine):
                        LOG.warning("Seslendirme devre disi birakildi (art arda hata).")
                        try:
                            self.engine.close()
                        except Exception:
                            pass
                        self.engine = NullTTSEngine()
                        self.engine_name = "kapali"
            finally:
                self._queue.task_done()


def build_speaker(settings: Settings) -> Speaker:
    engine_name = (settings.tts_engine or "edge").lower()
    if engine_name in ("none", "off", "yok", "kapali"):
        return Speaker(NullTTSEngine())

    engine: Any = None
    errors: List[str] = []

    if engine_name in ("edge", "edge-tts", "auto"):
        try:
            engine = EdgeTTSEngine(settings)
        except Exception as exc:
            errors.append("edge-tts: {}".format(exc))
            LOG.warning("edge-tts kullanilamadi, pyttsx3 denenecek. Sebep: %s", exc)

    if engine is None:
        try:
            engine = Pyttsx3Engine(settings)
        except Exception as exc:
            errors.append("pyttsx3: {}".format(exc))

    if engine is None:
        for message in errors:
            LOG.warning("%s", message)
        LOG.warning("Hicbir TTS motoru kullanilamadi; Jarvis sessiz calisacak.")
        engine = NullTTSEngine()

    return Speaker(engine)


def list_edge_voices() -> int:
    try:
        import edge_tts
    except Exception:
        print("edge-tts kurulu degil: pip install edge-tts")
        return 1
    try:
        voices = asyncio.run(edge_tts.list_voices())
    except Exception as exc:
        print("Sesler alinamadi (internet gerekli): {}".format(exc))
        return 1
    turkish = [voice for voice in voices if str(voice.get("Locale", "")).lower().startswith("tr")]
    if not turkish:
        print("Turkce ses bulunamadi.")
        return 0
    print("Turkce edge-tts sesleri:")
    for voice in turkish:
        print("  {:<24} {:<8} {}".format(
            voice.get("ShortName", "?"), voice.get("Gender", "?"), voice.get("Locale", "?")))
    return 0


def speak_text(speaker: Speaker, text: str) -> None:
    speaker.resume()
    speaker.feed(text)
    speaker.finish()


# ---------------------------------------------------------------------------
# Asistan (bellek + akisli yanit + seslendirme)
# ---------------------------------------------------------------------------
class Assistant:
    def __init__(
        self,
        settings: Settings,
        llm: LMStudioClient,
        speaker: Speaker,
        weather: Optional["WeatherService"] = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.speaker = speaker
        if weather is not None:
            self.weather: Optional[WeatherService] = weather
        elif settings.weather_enabled:
            self.weather = WeatherService(settings)
        else:
            self.weather = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": settings.system_prompt}
        ]

    def run_tool(self, name: str, arguments: str) -> Dict[str, Any]:
        """Modelin cagirdigi araclari calistirir (su an: get_weather)."""
        if name != "get_weather" or self.weather is None:
            raise JarvisError("Bilinmeyen veya kapali arac: {}".format(name))
        try:
            raw_args = json.loads(arguments or "{}")
            args = raw_args if isinstance(raw_args, dict) else {}
        except ValueError:
            LOG.debug("Arac argumani JSON degil: %r", arguments)
            args = {}

        location = args.get("konum") or args.get("location") or args.get("sehir")
        if isinstance(location, str):
            location = location.strip() or None
        else:
            location = None
        day = args.get("gun") or args.get("day") or "bugun"

        report = self.weather.get(location, str(day))
        LOG.debug("Hava durumu: %s", report.summary())
        return report.as_tool_result()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.settings.system_prompt}]

    def _trim(self) -> None:
        limit = max(2, int(self.settings.history_turns) * 2)
        while len(self.messages) - 1 > limit:
            del self.messages[1]

    def reply(self, user_text: str) -> str:
        question = (user_text or "").strip()
        if not question:
            return ""

        self.messages.append({"role": "user", "content": question})
        self._trim()
        self.speaker.resume()

        print(c("Jarvis:", "96") + " ", end="", flush=True)
        pieces: List[str] = []
        buffer = ""
        try:
            for piece in self.llm.stream_chat(self.messages, tool_runner=self.run_tool):
                if not piece:
                    continue
                pieces.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
                buffer += piece
                ready, buffer = _pop_sentences(buffer)
                for chunk in ready:
                    self.speaker.feed(chunk)
            if buffer.strip():
                self.speaker.feed(buffer)
        except LLMError as exc:
            print()
            print(c("Hata:", "91") + " " + str(exc))
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
            speak_text(self.speaker, "Baglanti hatasi. LM Studio sunucusunu kontrol edin.")
            return ""
        except KeyboardInterrupt:
            self.speaker.stop()
            print()
            return ""
        finally:
            print()

        answer = "".join(pieces).strip()
        if not answer:
            answer = "Bir yanit uretemedim efendim."
            self.speaker.feed(answer)
        self.messages.append({"role": "assistant", "content": answer})
        self._trim()
        self.speaker.finish()
        return answer


# ---------------------------------------------------------------------------
# Calisma modlari
# ---------------------------------------------------------------------------
def banner(settings: Settings) -> None:
    print(c("=" * 62, "90"))
    print(c("  J A R V I S", "96") + c("  -  yerel sesli asistan", "90"))
    print(c("=" * 62, "90"))
    print("  LM Studio : {}".format(settings.base_url))
    print("  STT       : faster-whisper ({})".format(settings.whisper_model))
    print("  TTS       : {} / {}".format(settings.tts_engine, settings.voice))
    print("  Dil       : {}".format(settings.language or "otomatik"))
    if settings.wake_word:
        print("  Uyandirma : '{}'".format(settings.wake_word))
    if settings.weather_enabled:
        print("  Hava      : {} (open-meteo)".format(settings.weather_location))
    print(c("-" * 62, "90"))


def run_text_mode(assistant: Assistant) -> None:
    print("Yazili mod. Cikis: 'cikis'  |  Bellek temizle: '/reset'")
    while True:
        try:
            line = input(c("Sen:", "93") + " ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if is_exit_command(line):
            say_goodbye(assistant.speaker)
            return
        if line.lower() in ("/reset", "reset", "sifirla"):
            assistant.reset()
            print("Hafiza temizlendi.")
            continue
        assistant.reply(line)


def run_voice_loop(assistant: Assistant, recorder: MicrophoneRecorder,
                   transcriber: Transcriber, settings: Settings) -> None:
    print("Sesli mod hazir. Konusun ('cikis' deyin veya Ctrl+C).")
    while True:
        audio = recorder.listen()
        if audio is None:
            continue

        started = time.perf_counter()
        try:
            text = transcriber.transcribe(audio, recorder.sample_rate)
        except JarvisError as exc:
            print(c("STT hatasi:", "91") + " " + str(exc))
            continue
        stt_seconds = time.perf_counter() - started
        text = (text or "").strip()
        if not text:
            LOG.debug("Bos transkript, atlandi.")
            continue

        print("{} {}  {}".format(c("Sen:", "93"), text, c("({:.2f}s)".format(stt_seconds), "90")))

        if is_exit_command(text):
            say_goodbye(assistant.speaker)
            return

        if settings.wake_word:
            needle = normalize_for_match(settings.wake_word)
            if needle and needle not in normalize_for_match(text):
                LOG.info("Uyandirma kelimesi yok, atlandi.")
                continue
            if needle:
                text = re.sub(re.escape(settings.wake_word), " ", text, flags=re.IGNORECASE)
                text = text.strip(" ,.!?;:")
                if not text:
                    speak_text(assistant.speaker, "Efendim?")
                    continue

        assistant.reply(text)


def say_goodbye(speaker: Speaker) -> None:
    try:
        speak_text(speaker, "Gorusuruz efendim.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ortam kontrolu
# ---------------------------------------------------------------------------
def _report(ok: bool, label: str, detail: str = "") -> bool:
    mark = c("[ OK ]", "92") if ok else c("[HATA]", "91")
    line = "  {} {:<30}".format(mark, label)
    if detail:
        line += " {}".format(detail)
    print(line)
    return ok


def list_lmstudio_models(settings: Settings) -> int:
    client = LMStudioClient(settings)
    try:
        all_models = client.list_models()
    except LLMError as exc:
        print(c("Hata:", "91") + " " + str(exc))
        return 1

    try:
        selected = client.resolve_model()
    except LLMError:
        selected = ""

    print("LM Studio modelleri ({}):".format(settings.base_url))
    for name in all_models:
        is_chat = not any(hint in name.lower() for hint in client.NON_CHAT_HINTS)
        kind = "sohbet" if is_chat else "embed/rerank"
        marker = "  <-- secili" if name == selected else ""
        print("  [{:12s}] {}{}".format(kind, name, marker))
    if not any(name == selected for name in all_models):
        print("  Uyari: secili model listede yok, --model ile belirtin.")
    return 0


def show_weather(settings: Settings, location: Optional[str] = None) -> int:
    """--weather: LLM'e girmeden ham hava durumu raporu."""
    service = WeatherService(settings)
    try:
        today = service.get(location, "bugun")
        tomorrow = service.get(location, "yarin")
    except JarvisError as exc:
        print(c("Hata:", "91") + " " + str(exc))
        return 1
    print("  Bugun : " + today.summary())
    print("  Yarin : " + tomorrow.summary())
    return 0


def run_check(settings: Settings, tts_test: bool = False, stt_test: bool = False) -> int:
    print(c("Jarvis ortam kontrolu", "96"))
    print(c("-" * 62, "90"))
    problems = 0

    version = "{}.{}.{}".format(*sys.version_info[:3])
    if not _report(sys.version_info >= (3, 9), "Python >= 3.9", version):
        problems += 1

    for module_name in ("numpy", "sounddevice", "requests"):
        try:
            __import__(module_name)
            _report(True, module_name)
        except Exception as exc:
            _report(False, module_name, "pip install {} ({})".format(module_name, exc))
            problems += 1

    try:
        import faster_whisper  # noqa: F401

        _report(True, "faster-whisper", getattr(faster_whisper, "__version__", ""))
    except Exception as exc:
        _report(False, "faster-whisper", "pip install faster-whisper ({})".format(exc))
        problems += 1

    try:
        import edge_tts  # noqa: F401

        _report(True, "edge-tts", getattr(edge_tts, "__version__", ""))
    except Exception as exc:
        _report(False, "edge-tts", "pip install edge-tts ({})".format(exc))
        problems += 1

    player = AudioPlayer()
    _report(player.available, "ses calma (pygame/ffplay)", player.describe())
    if not player.available:
        problems += 1
    player.close()

    try:
        import pyttsx3  # noqa: F401

        _report(True, "pyttsx3 (cevrimdisi yedek)")
    except Exception:
        _report(False, "pyttsx3 (opsiyonel)", "pip install pyttsx3")

    gpus = list_cuda_devices()
    cuda_count = count_cuda_devices()
    print()
    if gpus:
        _report(True, "CUDA aygitlari", "{} GPU".format(len(gpus)))
        for index, name, memory in gpus:
            marker = "  <-- Whisper" if index == settings.whisper_gpu else ""
            print("        GPU {}: {} ({} MiB){}".format(index, name, memory, marker))
        if settings.whisper_gpu >= len(gpus):
            print("        Uyari: whisper_gpu={} gecersiz, GPU 0 kullanilacak.".format(settings.whisper_gpu))
    elif cuda_count:
        _report(True, "CUDA aygitlari", "{} GPU (nvidia-smi yok)".format(cuda_count))
    else:
        print("  [BILGI] CUDA aygiti yok; Whisper CPU/int8 ile calisacak.")

    print()
    if sd is None:
        print(c("Mikrofon: sounddevice yok, atlandi.", "93"))
    else:
        try:
            default_in, _default_out = sd.default.device
            has_input = default_in is not None and int(default_in) >= 0
            device_index = int(default_in) if has_input else None
            info = sd.query_devices(device_index, "input") if device_index is not None else {}
            name = info.get("name", "?") if isinstance(info, dict) else "?"
            _report(True, "mikrofon", "{} (indeks {})".format(name, device_index))
        except Exception as exc:
            _report(False, "mikrofon", str(exc))
            problems += 1
        else:
            working_rate = None
            rate_error: Optional[Exception] = None
            for rate in (settings.sample_rate, WHISPER_RATE, 48000, 44100, 8000):
                try:
                    sd.check_input_settings(device=device_index, channels=1, samplerate=rate, dtype="int16")
                except Exception as exc:
                    rate_error = exc
                    continue
                working_rate = rate
                break
            if working_rate is None:
                _report(False, "mono kayit", str(rate_error))
                problems += 1
            else:
                _report(True, "{} Hz mono kayit".format(working_rate),
                        "(16 kHz'e otomatik cevrilir)" if working_rate != WHISPER_RATE else "")

    print()
    client = LMStudioClient(settings)
    try:
        model = client.check()
        _report(True, "LM Studio", "{} -> {}".format(settings.base_url, model))
        try:
            for name in client.chat_models():
                print("        - {}".format(name))
        except LLMError:
            pass
    except LLMError as exc:
        lines = str(exc).splitlines()
        _report(False, "LM Studio", lines[0] if lines else "")
        for line in lines[1:]:
            print("        " + line)
        problems += 1

    if stt_test:
        print()
        transcriber = Transcriber(settings)
        try:
            started = time.perf_counter()
            transcriber.load()
            elapsed = time.perf_counter() - started
            _report(True, "STT modeli", "{:.1f} sn | aygit: {}".format(elapsed, transcriber.device_label))
        except JarvisError as exc:
            lines = str(exc).splitlines()
            _report(False, "STT modeli", lines[0] if lines else "")
            problems += 1

    if tts_test and problems == 0:
        print()
        print("Ses testi: 'Merhaba efendim, sistem hazir.'")
        speaker = build_speaker(settings)
        try:
            speak_text(speaker, "Merhaba efendim, sistem hazir.")
        finally:
            speaker.close()

    print()
    if problems == 0:
        print(c("Her sey hazir. 'python jarvis.py' ile baslatin.", "92"))
    else:
        print(c("{} sorun bulundu. Yukaridaki ipuclarini izleyin.".format(problems), "93"))
    return 0 if problems == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Jarvis - tamamen yerel sesli asistan (faster-whisper + LM Studio + edge-tts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("--base-url", default=None, help="LM Studio adresi (varsayilan: http://localhost:1234/v1)")
    parser.add_argument("--model", default=None, help="LM Studio model kimligi (bos ise otomatik secilir)")
    parser.add_argument("--system-prompt", default=None, help="Sistem promptunu degistir")
    parser.add_argument("--temperature", type=float, default=None, help="LLM sicakligi (varsayilan 0.6)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Yanit uzunlugu siniri (varsayilan 512)")

    parser.add_argument("--language", default=None, help="STT dili (varsayilan: tr)")
    parser.add_argument("--whisper-model", default=None, help="tiny/base/small/medium/large-v3 (varsayilan: small)")
    parser.add_argument("--whisper-device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--whisper-compute-type", default=None, help="int8 / float16 / int8_float16")
    parser.add_argument("--whisper-gpu", type=int, default=None,
                        help="Coklu GPU: Whisper'in kullanacagi kart indeksi (varsayilan 0)")

    parser.add_argument("--device", type=int, default=None, help="Mikrofon cihaz indeksi (--list-devices)")
    parser.add_argument("--silence-ms", type=int, default=None, help="Sessizlik esigi ms (varsayilan 900)")
    parser.add_argument("--max-seconds", type=float, default=None, help="Tek kayit suresi siniri (varsayilan 20)")

    parser.add_argument("--tts", dest="tts_engine", default=None, choices=["edge", "pyttsx3", "none"],
                        help="Seslendirme motoru (varsayilan: edge)")
    parser.add_argument("--no-tts", action="store_true", help="Seslendirmeyi kapat")
    parser.add_argument("--voice", default=None, help="edge-tts sesi (varsayilan: tr-TR-EmelNeural)")
    parser.add_argument("--wake-word", default=None, help="Sadece bu kelimeyi iceren komutlari isle")
    parser.add_argument("--weather-location", default=None,
                        help="Hava durumu icin varsayilan konum (varsayilan: Kirklareli,TR)")
    parser.add_argument("--no-weather", action="store_true", help="Hava durumu aracini kapat")
    parser.add_argument("--weather", action="store_true",
                        help="Konumun hava durumunu yazdir ve cik (LLM/mikrofon kullanmaz)")

    parser.add_argument("--text", action="store_true", help="Mikrofon yerine klavyeden yaz")
    parser.add_argument("--once", default=None, metavar="METIN", help="Tek komutu isle ve cik")
    parser.add_argument("--say", default=None, metavar="METIN", help="Sadece TTS testi yap")
    parser.add_argument("--think", action="store_true",
                        help="Reasoning modellerinde dusunmeyi ac (varsayilan: kapali, daha hizli)")
    parser.add_argument("--list-models", action="store_true", help="LM Studio'daki modelleri listele")

    parser.add_argument("--check", action="store_true", help="Ortam ve baglanti kontrolu")
    parser.add_argument("--check-tts", action="store_true", help="--check ile ses testi de yap")
    parser.add_argument("--check-stt", action="store_true", help="--check ile Whisper modelini de yukle")
    parser.add_argument("--list-devices", action="store_true", help="Mikrofonlari listele")
    parser.add_argument("--list-voices", action="store_true", help="Turkce edge-tts seslerini listele")

    parser.add_argument("--config", default=None, help="config.json yolu")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--verbose", action="store_true", help="Ayrintili kayit (DEBUG)")
    return parser.parse_args(argv)


def configure_logging(args: argparse.Namespace) -> None:
    level_name = "DEBUG" if getattr(args, "verbose", False) else (args.log_level or "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    setup_console()
    _prepare_cuda_dlls()
    args = parse_args(argv)
    configure_logging(args)
    settings = load_settings(args)

    if args.list_devices:
        return list_audio_devices()
    if args.list_voices:
        return list_edge_voices()
    if args.list_models:
        return list_lmstudio_models(settings)
    if args.weather:
        return show_weather(settings, args.weather_location)
    if args.check:
        return run_check(settings, tts_test=args.check_tts, stt_test=args.check_stt)

    banner(settings)
    speaker = build_speaker(settings)
    print("  Ses motoru: {}".format(speaker.engine_name))

    try:
        if args.say:
            speak_text(speaker, args.say)
            return 0

        llm = LMStudioClient(settings)
        try:
            model = llm.resolve_model()
        except LLMError as exc:
            print()
            print(c("Hata:", "91") + " " + str(exc))
            print()
            print("Kontrol: python jarvis.py --check")
            return 1
        print("  Model     : {}".format(model))
        print(c("-" * 62, "90"))

        assistant = Assistant(settings, llm, speaker)

        if args.once:
            assistant.reply(args.once)
            return 0

        if args.text:
            run_text_mode(assistant)
            return 0

        transcriber = Transcriber(settings)
        transcriber.load()

        recorder = MicrophoneRecorder(settings)

        def on_state(state: str) -> None:
            if state == "dinliyor":
                LOG.debug("mikrofon dinliyor")

        recorder.on_state = on_state
        recorder.calibrate()
        print(c("-" * 62, "90"))
        speak_text(speaker, "Sistem hazir efendim.")
        run_voice_loop(assistant, recorder, transcriber, settings)
        return 0

    except JarvisError as exc:
        print()
        print(c("Hata:", "91") + " " + str(exc))
        return 1
    except KeyboardInterrupt:
        print()
        print("Gorusuruz.")
        return 0
    finally:
        speaker.close()


if __name__ == "__main__":
    sys.exit(main())
