# Jarvis — Tamamen Yerel Sesli Asistan

Mikrofon → **faster-whisper** → **LM Studio** (yerel LLM) → **edge-tts / pyttsx3** → hoparlör.
Bulut yok, API anahtarı yok; tüm konuşma tanıma ve dil modeli çalışması kendi makinenizde.

```
            ┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐   ┌──────────────┐
  🎙  ────► │ VAD + kayıt  │──►│ faster-whisper│──►│ LM Studio /v1/chat/  │──►│  TTS motoru  │──► 🔊
            │ (sounddevice)│   │  (STT, 16 kHz)│   │ completions (SSE)    │   │ edge / pyttsx3│
            └──────────────┘   └───────────────┘   └──────────────────────┘   └──────────────┘
                   ▲                                                                    │
                   └──────────────── konuşma bitince tekrar dinlemeye döner ◄─────────────┘
```

---

## 1. Mimari ve modüller

`jarvis.py` tek dosyada, birbirinden bağımsız katmanlardan oluşur:

| Katman | Sınıf / fonksiyon | Görev |
|---|---|---|
| Kayıt + VAD | `MicrophoneRecorder` | 16 kHz mono kayıt, enerji tabanlı konuşma algılama, otomatik gürültü kalibrasyonu, pre-roll, sessizlikte kesme |
| STT | `Transcriber` | `faster-whisper` modeli (CUDA → CPU otomatik düşüş), segment filtreleme, halüsinasyon temizliği |
| LLM | `LMStudioClient` | `GET /v1/models` ile model keşfi, `POST /v1/chat/completions` ile **akışlı** yanıt (SSE) |
| TTS | `EdgeTTSEngine`, `Pyttsx3Engine`, `Speaker` | Cümle bazlı sentez + arka plan iş parçacığında oynatma, kesme (stop) desteği |
| Oynatma | `AudioPlayer` | `pygame` varsa o, yoksa `ffplay`, hiçbiri yoksa anlaşılır hata |
| Orkestrasyon | `Assistant` | Sohbet geçmişi (son N tur), akışlı yanıtı cümlelere bölüp TTS kuyruğuna verme |
| Ayarlar | `Settings`, `load_settings` | `config.json` < ortam değişkenleri < CLI sırasıyla öncelik |
| CLI | `parse_args`, `main` | `--check`, `--text`, `--once`, `--say`, `--list-devices` … |

**Gecikme neden düşük?** LLM yanıtı akışlı (stream) okunur; ilk cümle tamamlanır tamamlanmaz
TTS'e verilir. Konuşma başlarken model hâlâ yanıt üretmeye devam eder.

---

## 2. Gereksinimler

* Windows 10/11 (macOS ve Linux'ta da çalışır; ses oynatma için `ffplay` gerekebilir)
* Python **3.9 – 3.12**
* [LM Studio](https://lmstudio.ai/) kurulu ve çalışıyor olmalı
* Mikrofon + hoparlör/kulaklık
* İlk çalıştırmada Whisper modeli indirilir (~500 MB `small` için) → **bir kez** internet gerekir
* `edge-tts` internet ister (Microsoft'un ücretsiz ses servisi). Tamamen çevrimdışı olmak için `--tts pyttsx3`

### Python kütüphaneleri

| Paket | Sürüm | Neden |
|---|---|---|
| `faster-whisper` | ≥ 1.0.3 | CTranslate2 tabanlı hızlı Whisper; `ctranslate2`, `tokenizers`, `huggingface-hub` bağımlılıklarını kendisi getirir |
| `sounddevice` | ≥ 0.4.6 | Mikrofon akışı (içinde PortAudio ikilisi gelir) |
| `numpy` | ≥ 1.24 | Ses dizileri, RMS tabanlı VAD, resampling |
| `requests` | ≥ 2.31 | LM Studio HTTP + SSE |
| `edge-tts` | ≥ 6.1.9 | Doğal Türkçe sesler |
| `pygame` | ≥ 2.5.2 | Üretilen mp3'ü çalmak |
| `pyttsx3` | ≥ 2.90 | *(opsiyonel)* Çevrimdışı yedek motor (Windows SAPI5) |

Kurulum:

```bat
:: Kolay yol (venv + tüm paketler):
setup.bat

:: Elle:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: GPU'lu STT istiyorsanız (NVIDIA, CUDA 12):
pip install -r requirements-gpu.txt
```

### GPU ile STT (opsiyonel, önerilir)

`faster-whisper`'ın motoru CTranslate2, cuBLAS/cuDNN'i çalışma zamanında `LoadLibrary` ile arar.
`requirements-gpu.txt` bu DLL'leri venv içine kurar, `jarvis.py` da açılışta ilgili klasörleri
otomatik PATH'e ekler (`_prepare_cuda_dlls`) — ayrıca ayar gerekmez.

Ölçülen değerler (RTX 3060, `small` model, 4.94 sn Türkçe ses):

| Aygıt | Model yükleme | 5 sn sesi çevirme |
|---|---|---|
| `cuda / float16` | ~1 s (ilk CUDA başlatmada ~17 s) | **0.17 s** |
| `cpu / int8` | ~2.8 s | 1.25 s |

GPU bulunamazsa veya kütüphane eksikse Jarvis sessizce CPU/int8'e düşer; hiçbir şey bozulmaz.
`whisper_device` varsayılanı `auto`: önce CUDA, olmazsa CPU.

### Çift GPU (2× RTX 3060 12 GB)

İki kart **NVLink olmadan** bellek havuzu oluşturmaz; LM Studio modelin katmanlarını iki karta
böler → toplam **24 GB VRAM**. Yani kazanç “compute” değil **kapasite**: 27B Q4 modeli 12 GB'a
sığmadığı için CPU'ya taşarken, 24 GB'da tamamen GPU'da kalır (yaklaşık 2-3× hız).
Whisper zaten tek GPU kullanır ve 0.17 s'dir; çift kart orada hız kazandırmaz ama artık
`--whisper-model large-v3` gibi daha doğru bir modeli VRAM'den çalmadan çalıştırabilirsin.

```bat
nvidia-smi -L                       :: iki karti dogrula
python jarvis.py --check            :: "CUDA aygitlari" bolumunde GPU 0 / GPU 1 ve Whisper'in karti
python jarvis.py --whisper-gpu 1    :: Whisper 2. karta (LM Studio 1. karti kullanir)
```

Kalıcı olsun istersen `config.json`: `"whisper_gpu": 1`. Kurulum notları: 750 W PSU, iki ayrı
8-pin kablo, BIOS'ta *Above 4G Decoding* + *Resizable BAR*; ikinci yuva genelde x4'tür (LLM
çıkarımı için sorun değil, yalnızca model yüklemesi biraz yavaşlar).

---

## 3. LM Studio ayarı

1. LM Studio'yu açın → **Developer** (eski sürümlerde *Local Server*) sekmesi.
2. Bir **instruct/chat** modeli yükleyin (Türkçe için örn. `Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct`, `gemma-2-9b-it`).
3. **Start Server** → adres `http://localhost:1234`. OpenAI uyumlu uç: `http://localhost:1234/v1`.
4. GPU kullanacaksanız model ayarlarında *GPU Offload*'u artırın (gecikmeyi ciddi düşürür).

Doğrulama:

```bat
curl http://localhost:1234/v1/models
:: veya
run_jarvis.bat --check
```

`--model` vermezseniz `/v1/models` içindeki **ilk** model otomatik seçilir.

---

## 4. Kullanım

```bat
run_jarvis.bat                    :: sesli mod (varsayılan)
python jarvis.py                  :: aynısı, venv aktifse
python jarvis.py --check          :: ortam + LM Studio + mikrofon kontrolü
python jarvis.py --check --check-tts   :: + hoparlör testi
python jarvis.py --check --check-stt   :: + Whisper modelini yükle ve hangi aygıtta çalıştığını gör
python jarvis.py --text           :: mikrofon yerine klavye (hızlı deneme)
python jarvis.py --once "Merhaba, kendini tanıt"
python jarvis.py --say "Ses testi bir, iki, üç"
python jarvis.py --list-devices   :: mikrofon indeksleri
python jarvis.py --list-voices    :: Türkçe edge-tts sesleri
python jarvis.py --list-models    :: LM Studio'daki sohbet modelleri
python jarvis.py --weather        :: varsayılan konumun hava durumu (LLM'siz)
python jarvis.py --weather --weather-location "Istanbul,TR"
```

Sesli modda: konuşun, sustuğunuzda (~0.9 sn) kayıt biter, metne çevrilir, yanıt seslendirilir.
Çıkmak için **“çıkış”** deyin veya `Ctrl+C`.

### Sık kullanılan seçenekler

| Seçenek | Açıklama |
|---|---|
| `--model "Qwen2.5-7B-Instruct"` | LM Studio model kimliği (`--list-models` adayları gösterir) |
| `--whisper-model base\|small\|medium\|large-v3` | Hız/doğruluk dengesi (`small` dengeli) |
| `--whisper-device cuda --whisper-compute-type float16` | GPU'da STT |
| `--whisper-gpu 1` | Çoklu GPU: Whisper'ın hangi kartta çalışacağı (varsayılan `0`) |
| `--tts pyttsx3` | Tamamen çevrimdışı seslendirme |
| `--no-tts` | Sessiz (sadece yazı) |
| `--voice tr-TR-EmelNeural` | Türkçe kadın ses (**varsayılan**). `tr-TR-AhmetNeural` şu an Microsoft tarafında ses üretmiyor |
| `--list-models` | Yüklü modelleri sohbet / embed olarak sınıflandırıp seçili olanı işaretler |
| `--think` | Reasoning (Qwen3 vb.) modellerde düşünmeyi açar — varsayılan **kapalı**, çünkü sesli asistan için yavaş |
| `--max-tokens 1024` | Yanıt bütçesi (reasoning modelleri için yükseltin) |
| `--device 2` | Mikrofon indeksi |
| `--silence-ms 700` | Daha hızlı yanıt (biraz daha erken keser) |
| `--wake-word jarvis` | Sadece “jarvis” geçen komutları işler |
| `--weather-location "Kirklareli,TR"` | Hava durumu için varsayılan konum (**Kırklareli/TR**). “Istanbul,TR” veya koordinat `"41.73,27.22"` de olur |
| `--weather` | Hava durumunu LLM'e girmeden yazdırır (hızlı kontrol) |
| `--no-weather` | Hava durumu aracını tamamen kapatır |
| `--temperature 0.3` | Daha kararlı yanıtlar |
| `--verbose` | Ayrıntılı log (VAD eşiği, kayıt süresi, SSE) |

### Hava durumu aracı

Jarvis'in yerleşik bir `get_weather` aracı vardır; model bunu **OpenAI uyumlu tool calling**
ile çağırır (LM Studio bunu destekler). Veri kaynağı [open-meteo](https://open-meteo.com)'dur —
**API anahtarı gerekmez**, internet ister.

* Varsayılan konum: **Kırklareli / TR** (`weather_location`)
* Konum söylenirse model onu çıkarır: “İstanbul'da hava nasıl?”, “Antalya yarın yağmurlu mu?”
* “yarın” için günlük en düşük/en yüksek + yağış olasılığı döner
* Akış: model `tool_calls` üretir → araç çalışır → sonuç `role: tool` mesajı olarak modele verilir →
  model tek cümlelik doğal yanıt üretir ve seslendirilir
* Ağ hatasında araç `{"hata": ...}` döndürür; Jarvis konuşmayı bozmaz

```bat
python jarvis.py --weather                              :: Kırklareli bugün + yarın (ham veri)
python jarvis.py --once "Kırklareli'de hava nasıl?"       :: tam akış (araç + ses)
python jarvis.py --once "Yarın İstanbul'da yağmur var mı?"
```

### `config.json` (opsiyonel)

```bat
copy config.example.json config.json
```

`jarvis.py` çalıştığı klasördeki `config.json` dosyasını otomatik okur.
Öncelik sırası: **CLI > ortam değişkenleri > config.json > varsayılanlar**.

Bu makinede hazır gelen `config.json`, hızlı cevap veren modeli sabitler:
`qwen3.8-27b-heretic-abliterated-uncensored`. Farklı model için ya bu dosyayı düzenleyin
(`--list-models` ile adaylara bakın) ya da `--model` verin.

Ortam değişkenleri: `JARVIS_BASE_URL`, `JARVIS_API_KEY`, `JARVIS_MODEL`, `JARVIS_SYSTEM_PROMPT`,
`JARVIS_LANGUAGE`, `JARVIS_WHISPER_MODEL`, `JARVIS_TTS_ENGINE`, `JARVIS_VOICE`, `JARVIS_WAKE_WORD`.

---

## 5. Test

Mikrofon, LM Studio ve hoparlör gerektirmeyen birim testleri:

```bat
python -m unittest -v test_jarvis.py
```

Kapsam: metin temizleme/cümleleme, ayar önceliği, TTS kuyruğu ve stop/resume, asistan akışı ve
hata durumunda geçmişin bozulmaması, 44.1/48 kHz → 16 kHz dönüşümü. **41 test.**

Uçtan uca test sırası:

```bat
run_jarvis.bat --check --check-tts --check-stt   :: 1) ortam + ses + STT
run_jarvis.bat --say "Merhaba efendim"      :: 2) TTS + hoparlör
run_jarvis.bat --once "İki cümlede kendini tanıt"   :: 3) LM Studio + TTS
run_jarvis.bat                              :: 4) mikrofon + tam döngü
```

Hepsini tek tıkla çalıştırıp `kurulum_sonuc.log` üretmek için: `kur_ve_test_et.bat`
(veya hızlı sürüm: `test_et.bat`).

### Bu makinede ölçülen sonuçlar

| Adım | Sonuç |
|---|---|
| `py_compile` | OK |
| `unittest` | 23/23 OK |
| `--check` | Tümü OK — mikrofon `NGENUITY Microphone (HyperX)`, 16 kHz destekli |
| `--say` | edge-tts `tr-TR-EmelNeural` ile seslendirildi |
| `--once` | `qwen3.8-27b…` → Türkçe yanıt → seslendirildi. İlk token 2.9 s, 140 karakterlik yanıt 5.5 s (25 kr/s), toplam süre ses oynatması dahil |
| STT | edge-tts ile üretilen 4.94 sn Türkçe ses → `small` model, **GPU'da 0.17 sn** (CPU'da 1.25 sn) |
| Mikrofon | 2 sn kayıt, gürültü tabanı RMS ≈ 3 |
| Gerçek sesli tur | `Sen: Merhaba. (1.03s)` → `Jarvis: Merhaba, buyurun. Size nasıl yardımcı olabilirim?` |
| CUDA | `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` kurulu; Whisper `cuda/float16` ile çalışıyor (RTX 3060) |

---

## 6. Sorun giderme

| Belirti | Çözüm |
|---|---|
| `LM Studio sunucusuna ulasilamadi` | LM Studio → Developer → **Start Server**; model yüklü mü? `--base-url` doğru mu? |
| `400/404` model hatası | `--model <tam-kimlik>` verin; kimliği `curl http://localhost:1234/v1/models` ile görün |
| `Mikrofon acilamadi` | `--list-devices` ile indeks bulup `--device N` verin; Windows gizlilik ayarlarından mikrofon iznini açın |
| Kayıt hiç bitmiyor / erken kesiliyor | `--silence-ms 1200` (daha sabırlı) ya da `--silence-ms 600`; gürültülü ortamda `vad_threshold` değerini `config.json`'da 200-400 arası sabitleyin |
| Kendi sesini duyup tekrar yanıtlıyor | Mikrofon açıkken hoparlör yok; Jarvis konuşurken dinlemez, ama yankı süresi uzunsa `silence_ms` artırın veya kulaklık kullanın |
| `edge-tts sentezleme hatasi` | İnternet yok veya ses adı yanlış → `--list-voices`; ısrarcıysa `--tts pyttsx3` |
| Ses gelmiyor ama hata da yok | `tr-TR-AhmetNeural` şu an Microsoft tarafında sessiz dönüyor. `--voice tr-TR-EmelNeural` (varsayılan) kullanın |
| `Jarvis:` boş kalıyor / yanıt gelmiyor | Model yalnızca `reasoning` üretmiş olabilir: `--list-models` ile düşünmeyen bir instruct model seçin, ya da `--think --max-tokens 2048` deneyin |
| Ses çıkmıyor (`pygame` yok) | `pip install pygame` ya da ffmpeg kurup `ffplay`'i PATH'e ekleyin |
| Whisper yavaş | `--whisper-model base` (veya `tiny`), `--whisper-device cuda --whisper-compute-type float16` |
| Çift GPU'da Jarvis ile LM Studio VRAM için yarışıyor | `--whisper-gpu 1` verin (Whisper 2. karta geçer) veya LM Studio'da offload'u azaltın |
| İlk çalıştırma çok yavaş | Whisper modeli indiriliyor; sonrasında önbellekten yüklenir |
| Türkçe karakterler bozuk | `chcp 65001` (batch dosyaları yapıyor) veya Windows Terminal kullanın |
| `faster-whisper` kurulmuyor | Python 3.13 desteklenmiyor olabilir → 3.11/3.12 kullanın |
| `metadata_errors` / `decode_audio` hatası | `av` sürümü fazla yeni: `pip install "av>=13,<14"` |
| `Library cublas64_12.dll is not found` | CUDA kütüphaneleri eksik: `pip install -r requirements-gpu.txt`. Jarvis GPU'yu kullanamazsa otomatik CPU/int8'e düşer |

---

## 7. Genişletme fikirleri

* **Uyandırma kelimesi (offline):** `openwakeword` / `pvporcupine` ile gerçek wake-word; şu anki `--wake-word` transkript üzerinde çalışır.
* **Sözünü kesme (barge-in):** `Speaker` çalarken mikrofonu düşük eşikle dinleyip `speaker.stop()` çağırın.
* **Araç çağırma:** LM Studio'nun OpenAI uyumlu *tools* desteği ile `Assistant.stream_chat` içinde `tool_calls` işleyin (hava durumu, uygulama açma, dosya arama…).
* **Kalıcı bellek:** `Assistant.messages`'ı SQLite'a yazın; açılışta özetle geri yükleyin.
* **VAD iyileştirmesi:** `webrtcvad` veya Silero VAD ile enerji eşiği yerine olasılık tabanlı tespit.
* **Streaming TTS yerine tek parça:** gecikme yerine daha doğal tonlama istiyorsanız `MAX_CHUNK`'ı büyütün.

---

## 8. Dosyalar

```
jarvis.py            ana uygulama (tek dosya, katmanlı)
requirements.txt     Python bağımlılıkları
requirements-gpu.txt requirements.txt + CUDA 12 kütüphaneleri (GPU'lu STT)
config.json          bu makineye özel ayarlar (model seçimi vb.)
config.example.json  örnek/dokümante ayar dosyası
test_jarvis.py       mikrofon/LM Studio gerektirmeyen birim testleri
PROJECT_CONTEXT.md   geliştirici bağlamı (mimari, parametreler, tuzaklar, yol haritası)
setup.bat            venv oluşturur + bağımlılıkları kurar
kur_ve_test_et.bat   kurulum + tüm testler + ortam kontrolü (log: kurulum_sonuc.log)
test_et.bat          sadece testler (hızlı, ~10 sn)
run_jarvis.bat       venv varsa onunla çalıştırır, yoksa sistem Python'u
.gitignore
```

---

## Lisans

MIT © 2026 [Levent Dost](https://github.com/levo-tech7400) — ayrıntılar için [LICENSE](LICENSE).

> Bu proje tamamen yerel çalışır: konuşma tanıma (faster-whisper), dil modeli (LM Studio)
> ve seslendirme (edge-tts / pyttsx3). Hava durumu aracı dışında hiçbir veri dışarı çıkmaz.
