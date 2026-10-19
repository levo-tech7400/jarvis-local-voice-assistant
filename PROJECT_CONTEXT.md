# PROJECT_CONTEXT — Jarvis (tamamen yerel sesli asistan)

> **Bu dosya, projeye sonradan gelen bir geliştirici/ajan içindir.**
> Kodun tamamı `jarvis.py` (tek dosya, katmanlı, ~2016 satır) içindedir.
> Aşağıdaki bilgiler **bu makinede çalıştırılarak doğrulanmıştır** (son doğrulama: tüm testler 36/36 OK).

Son güncelleme: 2026-10-20 — makine: Windows, Python 3.12.6, RTX 3060 (12 GB, compute 8.6)

---

## 0. 30 saniyede başla

```bat
cd C:\Jarvis
.venv\Scripts\activate                     :: sanal ortam
python jarvis.py --check --check-stt       :: ortam + mikrofon + LM Studio + STT aygıtı
python jarvis.py                            :: sesli mod (asıl kullanım)
```

Gereksinimlerin hepsi zaten kurulu; LM Studio'nun açık ve **Server** başlatılmış olması yeterli.

---

## 1. Sistem mimarisi

```
        ┌──────────────────────┐        ┌──────────────────────────────┐
 🎙 ───►│  MicrophoneRecorder  │───────►│  Transcriber                 │
        │  sounddevice 16 kHz  │        │  faster-whisper "small"      │
        │  enerji tabanlı VAD  │        │  cuda / float16  (CPU yedek) │
        └──────────────────────┘        └───────────────┬──────────────┘
                     ▲                                  │ metin
                     │                                  ▼
                     │                  ┌──────────────────────────────┐
                     │                  │  LMStudioClient              │
                     │                  │  POST /v1/chat/completions   │
                     │                  │  stream (SSE), qwen3.8-27b   │
                     │                  └───────────────┬──────────────┘
                     │                                  │ token akışı
                     │                                  ▼
                     │                  ┌──────────────────────────────┐
                     │                  │  Assistant                   │
                     │                  │  sohbet geçmişi + cümleleme  │
                     │                  └───────────────┬──────────────┘
                     │                                  │ cümle kuyruğu
                     │                  ┌───────────────▼──────────────┐
                     └──────────────────│  Speaker → EdgeTTS / pyttsx3 │──► 🔊
                        (konuşurken       │  pygame (veya ffplay) çalar │
                         mikrofon kapalı) └──────────────────────────────┘
```

### Bileşen sorumlulukları

| Katman | Sınıf | Sorumluluk |
|---|---|---|
| Kayıt + VAD | `MicrophoneRecorder` | 16 kHz mono akış, otomatik gürültü kalibrasyonu, konuşma başlangıç/bitiş tespiti, pre-roll, örnekleme oranı düşüşü |
| STT | `Transcriber` | Whisper modeli yükleme (CUDA→CPU otomatik), 1 sn'lik doğrulama çıkarımı, segment filtreleme |
| LLM | `LMStudioClient` | Model keşfi/seçimi, SSE ile akışlı yanıt, reasoning (düşünme) modellerine dayanıklılık |
| TTS | `EdgeTTSEngine`, `Pyttsx3Engine`, `NullTTSEngine` | Sentez + oynatma; ses yedekleme zinciri; çevrimdışı yedek |
| Kuyruk/iş parçacığı | `Speaker` | Cümle cümle sentez-oynat, `stop()` / `resume()` ile kesme |
| Orkestrasyon | `Assistant` | Sistem promptu, son N tur geçmiş, akışı cümlelere bölüp TTS'e besleme |
| Ayarlar | `Settings`, `load_settings` | Katmanlı yapılandırma (CLI > env > config.json > varsayılan) |
| CLI/doktor | `parse_args`, `run_check`, `main` | Modlar, tanılama, temiz kapanış |

### Tasarım kararları (neden böyle)

1. **Akışlı yanıt + erken TTS**: LLM yanıtı SSE ile okunur, ilk cümle tamamlanınca seslendirme başlar → algılanan gecikme düşer.
2. **Konuşurken mikrofon kapalı**: echo ile kendini yanıtlama problemi yok (barge-in bilinçli olarak yok, bkz. §11).
3. **Her katmanda yedek yol**: CUDA→CPU, edge-tts→yedek sesler→pyttsx3→sessiz, pygame→ffplay, 16k→48k/44.1k.
4. **Tek dosya**: dağıtımı ve okumayı kolaylaştırmak için modüller tek dosyada, bölüm başlıklarıyla ayrılmış.

---

## 2. Sanal ortam ve çalıştırma

| Ne | Yol / komut |
|---|---|
| Proje kökü | `C:\Jarvis` |
| Sanal ortam | `C:\Jarvis\.venv` (Python 3.12.6) |
| Aktifleştirme (cmd) | `.venv\Scripts\activate` |
| Aktifleştirme (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Aktifleştirmeden çalıştırma | `.venv\Scripts\python.exe jarvis.py ...` |
| Python sürümü | 3.9–3.12 (3.13'te `faster-whisper`/`av` sorunlu) |

### Çalıştırma komutları

```bat
run_jarvis.bat                       :: venv varsa onunla çalıştırır (çift tık için)
python jarvis.py                     :: sesli mod
python jarvis.py --text              :: mikrofon yerine klavye
python jarvis.py --once "..."        :: tek komut (LLM+TTS, mikrofon yok)
python jarvis.py --say "..."         :: sadece TTS
python jarvis.py --check             :: ortam doktoru
python jarvis.py --check --check-stt :: + Whisper'ı yükle, kullanılan aygıtı göster
python jarvis.py --list-models       :: LM Studio modelleri (sohbet / embed)
python jarvis.py --list-devices      :: mikrofon indeksleri
python jarvis.py --list-voices       :: edge-tts Türkçe sesleri
```

**Çıkış**: sesli/yazılı modda `"çıkış"`, `"kapat"`, `"görüşürüz"`, `"exit"`, `"quit"` … veya `Ctrl+C`.
Çıkış kelimeleri: `jarvis.EXIT_WORDS` (normalize edilerek karşılaştırılır, Türkçe karakter duyarsız).

---

## 3. Bağımlılıklar (kurulu sürümler)

`requirements.txt` (çekirdek) + `requirements-gpu.txt` (CUDA ekstraları).

| Paket | Kurulu | Rol / not |
|---|---|---|
| `faster-whisper` | 1.2.1 | STT (CTranslate2 tabanlı Whisper) |
| `ctranslate2` | 4.8.3 | faster-whisper'ın motoru (otomatik geldi) |
| `av` | **13.1.0** | Ses dosyası çözme. **Pinli**: 15+ sürümlerde `metadata_errors` parametresi kaldırıldığı için `decode_audio` patlıyor (`av>=13,<14`) |
| `sounddevice` | 0.5.6 | Mikrofon (PortAudio ikilisi ile) |
| `numpy` | 2.5.4 | VAD, resampling |
| `requests` | 2.34.2 | LM Studio HTTP + SSE |
| `edge-tts` | 7.3.1 | Doğal Türkçe ses (internet gerekir) |
| `pygame` | 2.6.1 | mp3 oynatma (yoksa `ffplay`) |
| `pyttsx3` | 2.99 | Çevrimdışı yedek TTS (Windows SAPI5) |
| `onnxruntime` | 1.31.0 | Whisper'ın `vad_filter` (Silero VAD) bağımlılığı |
| `nvidia-cublas-cu12` | 12.9.2.10 | **GPU için** `cublas64_12.dll` |
| `nvidia-cudnn-cu12` | 9.27.0.42 | **GPU için** `cudnn64_9.dll` |

> CUDA DLL'leri `site-packages/nvidia/*/bin` altındadır. CTranslate2 bunları `LoadLibrary` ile
> aradığı için `jarvis._prepare_cuda_dlls()` açılışta bu klasörleri process PATH'ine ve
> `os.add_dll_directory`'ye ekler. **Ekstra ortam değişkeni gerekmez.**

---

## 4. Çalışma parametreleri (`Settings`)

Öncelik: **CLI argümanı > ortam değişkeni > `config.json` > dataclass varsayılanı**

| Alan | Varsayılan | Açıklama |
|---|---|---|
| `base_url` | `http://localhost:1234/v1` | LM Studio OpenAI uyumlu uç |
| `api_key` | `lm-studio` | Bearer (LM Studio doğrulamaz) |
| `model` | `""` → otomatik | Boşsa `/v1/models` içinden seçilir |
| `system_prompt` | `DEFAULT_SYSTEM_PROMPT` | Türkçe, 1-3 cümle, markdown yok, "efendim" hitabı |
| `temperature` | `0.6` | |
| `max_tokens` | `1024` | Reasoning modelleri için yükseltildi (eski: 512) |
| `disable_thinking` | `true` | `chat_template_kwargs.enable_thinking=false` gönderilir |
| `history_turns` | `8` | Son 8 tur (16 mesaj) tutulur |
| `request_timeout` | `120` s | Okuma zaman aşımı (bağlantı: 15 s) |
| `language` | `tr` | Whisper dili |
| `whisper_model` | `small` | `tiny/base/small/medium/large-v3` |
| `whisper_device` | `auto` | `cuda` → `cpu` düşüşü |
| `whisper_compute_type` | `auto` | CUDA'da `float16`, CPU'da `int8` |
| `input_device` | `None` | `--device N` |
| `sample_rate` | `16000` | Desteklenmezse 48k/44.1k'ya düşer |
| `frame_ms` | `30` | VAD pencere boyu |
| `silence_ms` | `900` | Konuşma bittikten sonra beklenen sessizlik |
| `max_seconds` | `20` | Tek kayıt üst sınırı |
| `vad_threshold` | `0` (otomatik) | Sabit eşik istiyorsanız 200-400 arası verin |
| `tts_engine` | `edge` | `edge` / `pyttsx3` / `none` |
| `voice` | `tr-TR-EmelNeural` | Bkz. §7 (Ahmet sesi bozuk) |
| `rate` / `pitch` / `volume` | `+0%` / `+0Hz` / `+0%` | edge-tts prosody |
| `wake_word` | `""` | Doluysa sadece bu kelimeyi içeren komutlar işlenir |
| `weather_enabled` | `true` | `get_weather` aracını (tool calling) açar/kapatır |
| `weather_location` | `"Kirklareli,TR"` | Hava durumu varsayılan konumu; `"Istanbul,TR"` veya `"41.73,27.22"` biçimleri de geçerli |
| `weather_timeout` | `10` s | open-meteo istek zaman aşımı |

### Bu makinedeki `config.json`

```json
{
  "model": "qwen3.8-27b-heretic-abliterated-uncensored",
  "temperature": 0.6,
  "max_tokens": 1024,
  "disable_thinking": true,
  "language": "tr",
  "whisper_model": "small",
  "whisper_device": "auto",
  "whisper_compute_type": "auto",
  "silence_ms": 900,
  "tts_engine": "edge",
  "voice": "tr-TR-EmelNeural"
}
```

### Ortam değişkenleri

`JARVIS_BASE_URL`, `JARVIS_API_KEY`, `JARVIS_MODEL`, `JARVIS_SYSTEM_PROMPT`, `JARVIS_LANGUAGE`,
`JARVIS_WHISPER_MODEL`, `JARVIS_TTS_ENGINE`, `JARVIS_VOICE`, `JARVIS_WAKE_WORD`

---

## 5. VAD (ses etkinlik tespiti) — eşikler ve zamanlamalar

`MicrophoneRecorder` sınıfı, **enerji (RMS) tabanlı** bir VAD kullanır (harici model yok).

| Parametre | Değer | Nerede |
|---|---|---|
| Çerçeve boyu | `frame_ms = 30 ms` | `Settings` |
| Konuşma eşiği | `max(90.0, gürültü_tabanı * 2.0 + 60.0)` | `MicrophoneRecorder.threshold` |
| Manuel eşik | `vad_threshold > 0` ise o kullanılır | `Settings` |
| Gürültü tabanı | Açılışta 1 sn ölçümün **medyanı**; boştayken EMA (`0.9*eski + 0.1*yeni`) | `calibrate()` / `listen()` |
| Konuşma başlangıcı | Eşiğin üstünde **250 ms** kesintisiz | `start_frames` |
| Pre-roll | Konuşma öncesi **300 ms** tampon | `preroll_len` |
| Konuşma bitişi | **900 ms** kesintisiz sessizlik (`silence_ms`) | `silence_frames` |
| Zorunlu bitiş | **20 sn** (`max_seconds`) | `max_frames` |
| En kısa geçerli kayıt | **300 ms** (altı atılır) | `listen()` |
| Örnekleme | 16 000 Hz; cihaz desteklemezse 48 000 / 44 100 → 16 kHz'e indirgeme | `_open_stream` + `to_whisper_pcm` |

Whisper tarafındaki ikinci VAD ve filtreler:

| Filtre | Değer | Etki |
|---|---|---|
| `vad_filter` | `True` | Silero VAD ile sessizliği atar |
| `vad_parameters.min_silence_duration_ms` | `400` | Whisper içi sessizlik eşiği |
| `no_speech_prob` | `> 0.75` → segment atılır | Sessizlikte halüsinasyonu engeller |
| `avg_logprob` | `< -1.2` → segment atılır | Güven düşük çıktıyı atar |
| `beam_size` / `best_of` | `5` / `5` | |
| `temperature` | `0.0` | Deterministik |

> **Bu makinede ölçülen gürültü tabanı: RMS ≈ 0.5–3** → eşik 90'da kalır (çok sessiz oda).
> Gürültülü ortamda `config.json`'a `"vad_threshold": 250` gibi sabit değer koymak daha kararlıdır.

---

## 6. LLM katmanı (LM Studio)

| Konu | Ayrıntı |
|---|---|
| Model | `qwen3.8-27b-heretic-abliterated-uncensored` (config.json'da sabit) |
| Uç | `POST http://localhost:1234/v1/chat/completions`, `stream: true` |
| Model listesi | `GET /v1/models` |
| Otomatik seçim | Embedding/rerank modelleri (**`embed`, `rerank`, `bge-`, `gte-`, `e5-`, `minilm`**) elenir; `instruct`/`chat` içerenler tercih edilir, `uncensored`/`abliterated`/`roleplay` geri plana atılır |
| Payload | `model`, `messages`, `temperature`, `max_tokens`, `stream`, ve gerekirse `chat_template_kwargs: {"enable_thinking": false}` |
| SSE çözümü | Satırlar **elle UTF-8** decode edilir (charset belirtilmeyen yanıtlarda Türkçe bozulmasın diye), `data:` öneki temizlenir, `[DONE]` bitirir |
| İçerik | `delta.content` kullanılır; `delta.reasoning_content` **sayılır ama konuşulmaz** |
| Boş yanıt koruması | Yanıt yalnızca `reasoning` içeriyorsa: 2. deneme `tokens = min(max(tokens*3, 1024), 4096)` + mesaja ` /no_think` ipucu; hâlâ boşsa anlaşılır hata |
| Hata mesajı | Bağlantı yoksa LM Studio'yu açma/Server başlatma adımlarını içeren Türkçe rehber |

**Not (bu makine):** yüklü iki sohbet modeli de Qwen3 "reasoning" türevi. 9B modeli 512 token'ı
düşünmeyle tüketip **boş** yanıt veriyordu; bu yüzden `disable_thinking` + `max_tokens=1024` +
retry mekanizması eklendi ve hızlı cevap veren 27B modeli `config.json`'da sabitlendi.

---

## 7. TTS katmanı

| Konu | Ayrıntı |
|---|---|
| Birincil | `edge-tts` 7.3.1, ses **`tr-TR-EmelNeural`** |
| ⚠️ Bilinen sorun | **`tr-TR-AhmetNeural` Microsoft tarafında ses üretmiyor** (`NoAudioReceived`). Ses listesinde görünür ama boştur |
| Yedek ses zinciri | `tr-TR-EmelNeural` → `en-US-AvaMultilingualNeural` → `en-US-EmmaMultilingualNeural` → `de-DE-SeraphinaMultilingualNeural` (başarılı olan oturum boyunca hatırlanır) |
| Cümleleme | `MAX_CHUNK = 200` karakter; cümle sonu regex'i, uzun cümleler virgül/boşluktan bölünür |
| Metin temizliği | Kod blokları, markdown, emoji, `[link](url)` seslendirmeden önce temizlenir |
| Oynatma | `pygame.mixer` (44.1 kHz, buffer 512) → yoksa `ffplay` |
| Geçici dosyalar | `%TEMP%\jarvis_tts_*\part_XXXX.mp3`, çalındıktan sonra silinir |
| Çevrimdışı yedek | `pyttsx3` (SAPI5). Motor, **kullanıldığı iş parçacığında** `comtypes.CoInitialize()` sonrası kurulur (COM kuralı), hız 175 |
| Dayanıklılık | Art arda 3 hatada motor devre dışı bırakılır (sessiz mod) |

`Speaker` mimarisi: `feed()` cümleleri kuyruğa atar, ayrı bir daemon thread sentezleyip çalar;
`finish()` kuyruğun bitmesini bekler, `stop()` çalmayı keser ve kuyruğu boşaltır, `resume()` tekrar açar.

---

## 8. "Eklenen özellikler" — gerçek durum

| Özellik | Durum | Ayrıntı |
|---|---|---|
| Enerji tabanlı VAD + otomatik kalibrasyon | ✅ **Var** | §5'teki eşikler; eşik `max(90, taban*2+60)`, başlangıç 250 ms, bitiş 900 ms |
| CUDA/float16 STT + otomatik CPU düşüşü | ✅ **Var** | `_self_test()` + çalışma anı fallback |
| Reasoning model koruması | ✅ **Var** | §6 |
| Ses yedekleme zinciri | ✅ **Var** | §7 |
| Uyandırma kelimesi (`--wake-word`) | ✅ **Var** (transkript üzerinde) | Gerçek wake-word modeli değil |
| Hava durumu aracı (`get_weather`) | ✅ **Var** (varsayılan konum **Kırklareli/TR**) | LM Studio tool calling + open-meteo; model konumu kendisi çıkarır, “yarın” tahmini desteklenir |

### Hava durumu aracı — nasıl çalışıyor?

| Parça | Sınıf / fonksiyon | Not |
|---|---|---|
| Araç tanımı | `weather_tool_schema()` | `get_weather(konum?, gun?)`, `required: []` → konum verilmezse varsayılan kullanılır |
| Veri kaynağı | `WeatherService` | `geocoding-api.open-meteo.com` (isim → koordinat, `countryCode` desteği) + `api.open-meteo.com/v1/forecast`; anahtarsız; TZ=Europe/Istanbul; `forecast_days=2` |
| Konum çözümü | `WeatherService.resolve()` | `"Istanbul,TR"`, `"41.73,27.22"` veya boş (→ `settings.weather_location`); sonuçlar oturum boyunca önbellekte |
| Rapor | `WeatherReport` | `as_tool_result()` → modele JSON; `summary()` → konsol çıktısı |
| Kod→Türkçe | `describe_weather_code()` | WMO 0-99 → “açık”, “parçalı bulutlu”, “gök gürültülü fırtına” … |
| Model akışı | `LMStudioClient._iter_events()` | `tools` + `tool_choice: auto` gönderilir; `delta.tool_calls` parçaları `index`'e göre birleştirilir |
| Araç çalıştırma | `Assistant.run_tool()` | `get_weather` → `WeatherService.get()`; hata olursa `{"hata": ...}` |
| Takip turu | `LMStudioClient._tool_followup()` | `assistant(tool_calls)` + `role: tool` mesajları → model ikinci turda tek cümlelik yanıtı üretir (bu turda `tools` kapalı, döngü koruması) |

**Doğrulanmış örnek çıktı** (`python jarvis.py --once "Kırklareli'de hava nasıl?"`):

```
model: get_weather({"konum":"Kirklareli,TR","gun":"bugun"})
veri : Kırklareli, TR: 11.8°C, parçalı bulutlu (en düşük 9.8 / en yüksek 20.3, nem %89, rüzgar 4.6 km/s)
yanıt: "Kırklareli'de bugün parçalı bulutlu bir hava var. Sıcaklık 11.8 derece ve yağmur beklenmiyor."
```

**Sınırlar:** araç turu ikinci bir LLM çağrısı gerektirir (27B modelde toplam ~15-27 s).
Model aracı çağırmazsa yanıt “hava durumuna erişimim yok” olabilir; `--verbose` ile
tool çağrısını görüp sistemi prompttan (`DEFAULT_SYSTEM_PROMPT`) güçlendirebilirsiniz.

**Yeni araç eklemek:** `weather_tool_schema()`'ya yeni bir `function` ekleyin,
`Assistant.run_tool()` içine `name` dalı yazın. Akış aynısıdır; ayrı bir mimari gerekmez.

---

## 9. Ölçülen performans (bu makine)

**Donanım**: RTX 3060 12 GB (compute 8.6), LM Studio'da 27B model ~10.5 GB VRAM kullanıyor.

| Ölçüm | Değer |
|---|---|
| STT model yükleme (cuda/float16) | ~1.1 s (hayatın ilk CUDA başlatmasında ~17 s) |
| STT model yükleme (cpu/int8) | ~2.8 s |
| 4.94 sn Türkçe sesi çevirme | **cuda: 0.17 s** / cpu: 1.25 s |
| LLM ilk token | ~2.9 s |
| LLM üretim hızı | ~25 karakter/s (27B model, kısmi GPU) |
| Uçtan uca `--once` (ses çalma dahil) | ~16 s |
| Gerçek sesli tur | `Sen: Merhaba. (1.03s)` → `Jarvis: Merhaba, buyurun...` |

**GPU'yu daha da hızlandırma fikirleri**: LM Studio'da 27B yerine 7-8B instruct model (tamamen
VRAM'e sığar, ~60-80 karakter/s), `--whisper-model medium` (GPU'da ~0.4 s).

---

## 10. Dosya yapısı

```
C:\Jarvis\
├── jarvis.py                  Ana uygulama (2016 satır, 75 KB) — TÜM mantık burada
├── test_jarvis.py             Birim testleri (36 test, mikrofon/LLM/ağ gerekmez)
├── config.json                Bu makineye özel ayarlar (model, whisper_device=auto, ses)
├── config.example.json        Dokümante örnek ayar (yorum alanı `_aciklama` filtrelenir)
├── requirements.txt           Çekirdek bağımlılıklar (av pinli)
├── requirements-gpu.txt       requirements.txt + nvidia-cublas/cudnn
├── run_jarvis.bat             Çalıştırıcı (venv varsa onunla), UTF-8 kod sayfası
├── setup.bat                  venv oluştur + pip install
├── kur_ve_test_et.bat         Kurulum + tüm testler + ortam kontrolü → kurulum_sonuc.log
├── test_et.bat                Sadece testler (hızlı) → kurulum_sonuc.log + Notepad
├── README.md                  Kullanıcı dokümanı (kurulum, sorun giderme)
├── PROJECT_CONTEXT.md         BU DOSYA — geliştirici bağlamı
├── .gitignore                 .venv/, __pycache__/, config.json, *.log
└── .venv\                     Sanal ortam (Python 3.12.6 + tüm paketler + CUDA DLL'leri)
```

### `jarvis.py` iç bölüm haritası (sıra ile)

1. Opsiyonel importlar (`numpy`, `sounddevice`, `requests` → yoksa `None`)
2. `setup_console()`, `c()`, `_as_float/_as_int/_as_bool`
3. `_prepare_cuda_dlls()` — **CUDA DLL PATH enjeksiyonu**
4. `to_whisper_pcm()` — int16/float → 16 kHz float32 (48k kutu filtre, 44.1k lineer enterpolasyon)
5. `Settings` + `load_settings()` + env/CLI eşlemesi
6. Hatalar: `JarvisError` / `LLMError` / `TTSError`
7. Metin yardımcıları: `strip_for_speech`, `split_into_chunks`, `_pop_sentences`,
   `normalize_for_match`, `is_exit_command`, `EXIT_WORDS`
8. `MicrophoneRecorder` (+ `list_audio_devices`)
9. `Transcriber` (`_build`, `_self_test`, `_transcribe_once`, `transcribe`)
10. Hava durumu: `WMO_DESCRIPTIONS`, `WeatherReport`, `WeatherService`, `weather_tool_schema()`,
    `is_weather_question()`
11. `LLMEvent` + `LMStudioClient` (`list_models`, `chat_models`, `_preferred`, `resolve_model`,
    `_with_hint`, `_iter_events`, `_content_round`, `_tool_followup`, `stream_chat`, `check`)
12. TTS: `_run_async`, `AudioPlayer`, `EdgeTTSEngine`, `Pyttsx3Engine`, `NullTTSEngine`,
    `Speaker`, `build_speaker`, `list_edge_voices`, `speak_text`
13. `Assistant` (bellek + akış + TTS besleme + `run_tool`)
14. Çalışma modları: `banner`, `run_text_mode`, `run_voice_loop`, `say_goodbye`
15. `run_check` + `list_lmstudio_models` + `show_weather`
16. `parse_args`, `configure_logging`, `main`

---

## 11. Bilinen tuzaklar (tekrar düşmemek için)

| Tuzak | Sebep / çözüm |
|---|---|
| `tr-TR-AhmetNeural` sessiz | Microsoft ücretsiz uçtan kaldırdı/bozdu. Emel kullan veya yedek zincir devrede |
| `Library cublas64_12.dll is not found` | CUDA DLL'leri PATH'te değil → `_prepare_cuda_dlls()` hallediyor; `requirements-gpu.txt` kurulu olmalı |
| CUDA yüklü ama çıkarım patlıyor | Model yükleme başarılı olsa da ilk çıkarımda hata çıkar → `_self_test()` bunu yakalar, CPU'ya geçer |
| Boş LLM yanıtı | Qwen3 reasoning modelleri `reasoning_content` üretir → `disable_thinking` + retry (bkz. §6) |
| `decode_audio` / `metadata_errors` hatası | `av` 15+ → `av>=13,<14` pinli |
| Ses hiç çıkmıyor ama hata yok | `pygame` kurulu mu? Değilse `ffplay` gerekir |
| Windows'ta pyttsx3 çalışmıyor | COM iş parçacığı kuralı → motor worker thread'de `comtypes.CoInitialize()` ile kurulur |
| Batch dosyası `goto` ile sapıtıyor | Projede yazılan `.bat` dosyaları **LF satır sonlu**; bu yüzden `goto`/etiket kullanılmıyor, akış doğrusal |
| Log/pipe'a yazınca çıktı kayboluyor | `setup_console()` `line_buffering=True` yapar |
| Embedding modeli sohbet sanılıyor | `NON_CHAT_HINTS` filtresi + `--list-models` |
| Türkçe karakterler bozuk | Konsol kod sayfası; batch dosyaları `chcp 65001` yapar. Dosya içerikleri UTF-8'dir |
| İlk çalıştırma çok yavaş | Whisper modeli HF'den indirilir (~480 MB, tek seferlik) |

---

## 12. Test ve doğrulama

```bat
:: Hızlı (mikrofon/LLM/hoparlör gerekmez) — 36 test, ~0.05 s
.venv\Scripts\python.exe -m unittest -v test_jarvis.py

:: Sözdizimi
.venv\Scripts\python.exe -m py_compile jarvis.py test_jarvis.py

:: Uçtan uca
.venv\Scripts\python.exe jarvis.py --check --check-tts --check-stt
.venv\Scripts\python.exe jarvis.py --say "Merhaba efendim"
.venv\Scripts\python.exe jarvis.py --once "Tek cümleyle kendini tanıt"
.venv\Scripts\python.exe jarvis.py            :: gerçek mikrofon turu
```

Test kapsamı (`test_jarvis.py`): markdown temizliği, cümleleme, `_pop_sentences`, çıkış komutları,
ayar önceliği/doğrulama, `Speaker` sıralama + stop/resume, `Assistant` akış/geçmiş/hata durumu,
ses örnekleme dönüşümleri (16k/44.1k/48k/boş).

**STT'yi hoparlör olmadan doğrulama numarası** (mikrofon gerekmez):

```python
import asyncio, os, edge_tts
from faster_whisper import decode_audio           # av<14 gerekir
import jarvis
p = os.path.join(os.environ["TEMP"], "t.mp3")
asyncio.run(edge_tts.Communicate("Yarin saat dokuzda toplanti var.", "tr-TR-EmelNeural").save(p))
audio = decode_audio(p, sampling_rate=16000)
t = jarvis.Transcriber(jarvis.Settings(language="tr").normalize())
print(t.transcribe(audio, 16000), "| aygıt:", t._device)
```

---

## 13. Geliştirici notları / yol haritası

**Nereye dokunulur?**

| Yapmak istediğin | Dosya / sınıf | Kanca |
|---|---|---|
| Yanıt üslubunu değiştir | `DEFAULT_SYSTEM_PROMPT` veya `--system-prompt` | — |
| Daha doğru STT | `--whisper-model medium` / `large-v3` | model HF'den iner, GPU'da hızlı |
| Farklı ses | `--voice`, `EdgeTTSEngine.FALLBACK_VOICES` | `--list-voices` |
| Yeni araç (uygulama açma, takvim…) | `weather_tool_schema()` + `Assistant.run_tool()` | §8'deki desen birebir kopyalanır |
| Sözünü kesme (barge-in) | `Speaker.stop()` | TTS çalarken mikrofonu düşük eşikle dinle |
| Gerçek wake-word | `run_voice_loop` | `openwakeword`/`pvporcupine` entegre et |
| Kalıcı bellek | `Assistant.messages` | SQLite'a yaz, açılışta özetle yükle |
| VAD'i model tabanlı yap | `MicrophoneRecorder.listen` | `webrtcvad` / Silero |

**Kod stili / kurallar (mevcut dosyaya uy):**
- Python 3.9+ uyumlu sözdizimi, `from __future__ import annotations`, `typing` jenerikleri.
- Kullanıcıya görünen tüm mesajlar **Türkçe ve ASCII-güvenli** (Türkçe karakter yalnızca yorum/doküman).
- Her dış çağrı (ağ, ses, model) kendi hatasını yakalar ve Türkçe ipucu verir; çökme yok.
- Yeni bir yedek yol eklerken: önce logla (`LOG.warning`), sonra sessizce düş.
- `py_compile` + `unittest` her değişiklikten sonra çalıştırılmalı (hızlı ve bağımsız).

**Yapılmadı / bilinçli boşluklar:** araç çağırma (tool calling), barge-in, gerçek wake-word modeli,
kalıcı bellek, çoklu dil, GUI/tray uygulaması, paketleme (PyInstaller).

---

## 14. Hızlı teşhis komutları

```bat
.venv\Scripts\python.exe jarvis.py --check --check-stt   :: her şeyin özeti (aygıt dahil)
.venv\Scripts\python.exe jarvis.py --list-models         :: hangi model seçili?
.venv\Scripts\python.exe jarvis.py --list-devices        :: mikrofon indeksi
.venv\Scripts\python.exe jarvis.py --verbose             :: VAD eşiği, kayıt süresi, SSE detayı
curl http://localhost:1234/v1/models                     :: LM Studio ayakta mı?
nvidia-smi                                               :: VRAM / sürücü
```

`--check` çıktısındaki tipik satırlar:

```
[ OK ] faster-whisper     1.2.1
[ OK ] ses calma (pygame/ffplay)   pygame
[ OK ] mikrofon           NGENUITY - Microphone (HyperX) (indeks 1)
[ OK ] 16000 Hz mono kayit
[ OK ] LM Studio          http://localhost:1234/v1 -> qwen3.8-27b-... (2 sohbet modeli yuklu)
[ OK ] STT modeli         1.1 sn | aygit: cuda
```
