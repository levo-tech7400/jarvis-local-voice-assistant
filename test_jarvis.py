#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jarvis - mikrofon / LM Studio / hoparlor gerektirmeyen birim testleri.

Calistirma:
    python -m unittest -v test_jarvis.py
    python test_jarvis.py
"""

import io
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import jarvis

# Testleri kullanicinin config.json dosyasindan bagimsiz kilmak icin var olmayan yol
NO_CONFIG = str(Path(tempfile.gettempdir()) / "jarvis_test_missing_config.json")


def parse(*argv):
    return jarvis.parse_args(["--config", NO_CONFIG] + list(argv))


class FakeEngine:
    """Ses cikarmayan, sadece metinleri kaydeden TTS motoru."""

    name = "fake"

    def __init__(self):
        self.spoken = []
        self._lock = threading.Lock()

    def speak(self, text, stop_event=None):
        with self._lock:
            self.spoken.append(text)

    def stop(self):
        pass

    def close(self):
        pass


class FakeLLM:
    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.calls = 0

    def stream_chat(self, messages, tool_runner=None):
        self.calls += 1
        for piece in self.pieces:
            yield piece


class FakeFailingLLM:
    def stream_chat(self, messages, tool_runner=None):
        raise jarvis.LLMError("test hatasi")
        yield ""  # generator olsun


class TextTests(unittest.TestCase):
    def test_markdown_temizligi(self):
        metin = "**Merhaba** `dunya`, [link](http://x) ve emoji \U0001F600"
        temiz = jarvis.strip_for_speech(metin)
        self.assertIn("Merhaba", temiz)
        self.assertIn("dunya", temiz)
        self.assertNotIn("**", temiz)
        self.assertNotIn("`", temiz)
        self.assertNotIn("http", temiz)
        self.assertNotIn("\U0001F600", temiz)

    def test_cumleleme(self):
        chunks = jarvis.split_into_chunks("Merhaba efendim. Bugun hava cok guzel!")
        self.assertEqual(chunks, ["Merhaba efendim.", "Bugun hava cok guzel!"])

    def test_uzun_cumle_bolunur(self):
        chunks = jarvis.split_into_chunks("kelime " * 120)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), jarvis.MAX_CHUNK)

    def test_pop_sentences(self):
        chunks, rest = jarvis._pop_sentences("Bir. Iki. Uc")
        self.assertEqual([c.strip() for c in chunks], ["Bir.", "Iki."])
        self.assertEqual(rest, "Uc")

    def test_uzun_parca_erken_gonderilir(self):
        chunks, rest = jarvis._pop_sentences("a" * 300)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(rest), 300 - jarvis.MAX_CHUNK)

    def test_cikis_komutlari(self):
        self.assertTrue(jarvis.is_exit_command("Çıkış!"))
        self.assertTrue(jarvis.is_exit_command("görüşürüz"))
        self.assertTrue(jarvis.is_exit_command("QUIT"))
        self.assertFalse(jarvis.is_exit_command("kapatma"))
        self.assertFalse(jarvis.is_exit_command(""))


class SettingsTests(unittest.TestCase):
    def test_varsayilanlar(self):
        settings = jarvis.load_settings(parse())
        self.assertEqual(settings.base_url, "http://localhost:1234/v1")
        self.assertEqual(settings.language, "tr")
        self.assertEqual(settings.tts_engine, "edge")
        self.assertEqual(settings.whisper_model, "small")
        self.assertIsNone(settings.input_device)

    def test_cli_ezerse_uygulanir(self):
        settings = jarvis.load_settings(
            parse("--model", "test-model", "--voice", "tr-TR-EmelNeural", "--device", "3")
        )
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.voice, "tr-TR-EmelNeural")
        self.assertEqual(settings.input_device, 3)

    def test_no_tts(self):
        settings = jarvis.load_settings(parse("--no-tts"))
        self.assertEqual(settings.tts_engine, "none")

    def test_bozuk_degerler_sabitlenir(self):
        settings = jarvis.Settings(temperature="abc", max_tokens="x", silence_ms=10, base_url="")
        settings = settings.normalize()
        self.assertEqual(settings.temperature, 0.6)
        self.assertEqual(settings.max_tokens, 1024)
        self.assertEqual(settings.silence_ms, 200)
        self.assertEqual(settings.base_url, "http://localhost:1234/v1")


class SpeakerTests(unittest.TestCase):
    def test_cumleleri_sirayla_soyler(self):
        engine = FakeEngine()
        speaker = jarvis.Speaker(engine)
        try:
            speaker.feed("Merhaba efendim. Bugun hava cok guzel!")
            speaker.finish(timeout=5)
            self.assertEqual(engine.spoken, ["Merhaba efendim.", "Bugun hava cok guzel!"])
        finally:
            speaker.close()

    def test_stop_ve_resume(self):
        engine = FakeEngine()
        speaker = jarvis.Speaker(engine)
        try:
            speaker.stop()
            speaker.feed("bu soylenmemeli")
            speaker.finish(timeout=2)
            self.assertEqual(engine.spoken, [])

            speaker.resume()
            speaker.feed("tekrar dinliyoruz")
            speaker.finish(timeout=5)
            self.assertEqual(engine.spoken, ["tekrar dinliyoruz"])
        finally:
            speaker.close()

    def test_kapali_motor_sessiz(self):
        speaker = jarvis.Speaker(jarvis.NullTTSEngine())
        try:
            jarvis.speak_text(speaker, "sessiz test")
        finally:
            speaker.close()


class AssistantTests(unittest.TestCase):
    def test_yanit_akisli_ve_gecmis_kaydedilir(self):
        engine = FakeEngine()
        speaker = jarvis.Speaker(engine)
        settings = jarvis.Settings().normalize()
        llm = FakeLLM(["Merhaba efendim.", " Nasilsiniz?"])
        assistant = jarvis.Assistant(settings, llm, speaker)
        try:
            with redirect_stdout(io.StringIO()):
                answer = assistant.reply("selam")
            self.assertEqual(answer, "Merhaba efendim. Nasilsiniz?")
            self.assertEqual(assistant.messages[0]["role"], "system")
            self.assertEqual(assistant.messages[1], {"role": "user", "content": "selam"})
            self.assertEqual(assistant.messages[2]["role"], "assistant")
            self.assertEqual(engine.spoken, ["Merhaba efendim.", "Nasilsiniz?"])
        finally:
            speaker.close()

    def test_hata_durumunda_gecmis_kirlenmez(self):
        engine = FakeEngine()
        speaker = jarvis.Speaker(engine)
        settings = jarvis.Settings().normalize()
        assistant = jarvis.Assistant(settings, FakeFailingLLM(), speaker)
        try:
            with redirect_stdout(io.StringIO()):
                answer = assistant.reply("selam")
            self.assertEqual(answer, "")
            self.assertEqual(len(assistant.messages), 1)  # sadece sistem promptu
            self.assertTrue(engine.spoken)                # kullaniciya bilgi verildi
        finally:
            speaker.close()

    def test_hafiza_sinirlanir(self):
        engine = FakeEngine()
        speaker = jarvis.Speaker(engine)
        settings = jarvis.Settings(history_turns=1).normalize()
        assistant = jarvis.Assistant(settings, FakeLLM(["tamam."]), speaker)
        try:
            with redirect_stdout(io.StringIO()):
                for _ in range(5):
                    assistant.reply("deneme")
            self.assertEqual(len(assistant.messages), 3)  # sistem + 1 tur
        finally:
            speaker.close()


@unittest.skipIf(jarvis.np is None, "numpy kurulu degil")
class AudioTests(unittest.TestCase):
    def test_16k_olcekleme(self):
        audio = jarvis.np.full(1600, 16384, dtype=jarvis.np.int16)
        pcm = jarvis.to_whisper_pcm(audio, 16000)
        self.assertEqual(pcm.dtype, jarvis.np.float32)
        self.assertEqual(len(pcm), 1600)
        self.assertAlmostEqual(float(pcm[0]), 0.5, places=3)

    def test_48k_indirgeme(self):
        audio = jarvis.np.ones(4800, dtype=jarvis.np.int16)
        pcm = jarvis.to_whisper_pcm(audio, 48000)
        self.assertEqual(len(pcm), 1600)

    def test_44k_enterpolasyon(self):
        audio = jarvis.np.ones(4410, dtype=jarvis.np.int16)
        pcm = jarvis.to_whisper_pcm(audio, 44100)
        self.assertEqual(len(pcm), 1600)
        self.assertAlmostEqual(float(pcm[0]), 1.0 / 32768.0, places=6)

    def test_bos_ses(self):
        pcm = jarvis.to_whisper_pcm(jarvis.np.zeros(0, dtype=jarvis.np.int16), 16000)
        self.assertEqual(len(pcm), 0)


class HelperTests(unittest.TestCase):
    def test_normalize_for_match(self):
        self.assertEqual(jarvis.normalize_for_match("Çıkış, lütfen!"), "cikis lutfen")

    def test_cikis_kumesi_bos_degil(self):
        self.assertIn("cikis", jarvis._EXIT_NORMALIZED)

    def test_renk_fonksiyonu(self):
        self.assertIsInstance(jarvis.c("x", "92"), str)


class WeatherTests(unittest.TestCase):
    def test_wmo_kodlari(self):
        self.assertEqual(jarvis.describe_weather_code(0), "açık")
        self.assertEqual(jarvis.describe_weather_code(3), "kapalı")
        self.assertEqual(jarvis.describe_weather_code("61"), "hafif yağmurlu")
        self.assertEqual(jarvis.describe_weather_code(95), "gök gürültülü fırtına")
        self.assertEqual(jarvis.describe_weather_code(999), "bilinmiyor")
        self.assertEqual(jarvis.describe_weather_code(None), "bilinmiyor")

    def test_niyet_tespiti(self):
        self.assertTrue(jarvis.is_weather_question("Kırklareli'de hava nasıl?"))
        self.assertTrue(jarvis.is_weather_question("yarın yağmur var mı"))
        self.assertFalse(jarvis.is_weather_question("bana bir şiir oku"))
        self.assertFalse(jarvis.is_weather_question(""))

    def test_rapor_ozeti(self):
        report = jarvis.WeatherReport(
            location="Kırklareli, TR", day="bugun", description="parçalı bulutlu",
            temperature=11.8, apparent=11.1, humidity=89, wind_kmh=4.6,
            t_min=9.8, t_max=20.3, rain_probability=0,
        )
        ozet = report.summary()
        self.assertIn("Kırklareli, TR", ozet)
        self.assertIn("11.8°C", ozet)
        self.assertIn("parçalı bulutlu", ozet)

        data = report.as_tool_result()
        self.assertEqual(data["konum"], "Kırklareli, TR")
        self.assertEqual(data["sicaklik_c"], 11.8)
        self.assertEqual(data["en_dusuk_c"], 9.8)
        self.assertEqual(data["yagis_olasiligi_yuzde"], 0)

    def test_yarin_ozeti_gunluk(self):
        report = jarvis.WeatherReport(
            location="Kırklareli, TR", day="yarin", description="az bulutlu",
            temperature=21.3, t_min=9.3, t_max=21.3,
        )
        self.assertTrue(report.summary().startswith("Kırklareli, TR: en yüksek 21.3°C"))

    def test_tool_semasi(self):
        schema = jarvis.weather_tool_schema()
        self.assertEqual(len(schema), 1)
        function = schema[0]["function"]
        self.assertEqual(function["name"], "get_weather")
        self.assertIn("konum", function["parameters"]["properties"])
        self.assertEqual(function["parameters"]["required"], [])

    def test_varsayilan_konum_kirklareli(self):
        settings = jarvis.Settings().normalize()
        self.assertEqual(settings.weather_location, "Kirklareli,TR")
        self.assertTrue(settings.weather_enabled)

    def test_no_weather(self):
        settings = jarvis.load_settings(parse("--no-weather"))
        self.assertFalse(settings.weather_enabled)

    def test_konum_cli(self):
        settings = jarvis.load_settings(parse("--weather-location", "Istanbul,TR"))
        self.assertEqual(settings.weather_location, "Istanbul,TR")

    def test_koordinat_ayristirma(self):
        service = jarvis.WeatherService(jarvis.Settings().normalize())
        label, lat, lon = service.resolve("41.73, 27.22")
        self.assertAlmostEqual(lat, 41.73, places=2)
        self.assertAlmostEqual(lon, 27.22, places=2)
        self.assertIn("41.73", label)

    def test_bilinmeyen_arac_hata_verir(self):
        assistant = jarvis.Assistant(
            jarvis.Settings().normalize(), FakeLLM([]), FakeSpeaker(), weather=None
        )
        with self.assertRaises(jarvis.JarvisError):
            assistant.run_tool("bilinmeyen_arac", "{}")

    def test_arac_sonucu_mesaji(self):
        class FakeWeather:
            def get(self, location=None, day="bugun"):
                return jarvis.WeatherReport(
                    location="Kırklareli, TR", day=day, description="açık", temperature=21.0,
                )

        assistant = jarvis.Assistant(
            jarvis.Settings().normalize(), FakeLLM([]), FakeSpeaker(), weather=FakeWeather()
        )
        result = assistant.run_tool("get_weather", '{"konum": "Kirklareli", "gun": "bugun"}')
        self.assertEqual(result["durum"], "açık")
        self.assertEqual(result["sicaklik_c"], 21.0)

    def test_arac_takip_mesajlari(self):
        settings = jarvis.Settings().normalize()
        client = jarvis.LMStudioClient(settings)
        conversation = [{"role": "user", "content": "hava nasıl?"}]
        calls = [
            jarvis.LLMEvent(
                "tool_call", tool_id="call_1", tool_name="get_weather",
                tool_arguments='{"gun": "bugun"}',
            )
        ]
        messages = client._tool_followup(conversation, calls, lambda name, args: {"durum": "açık"})
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertIn("açık", messages[2]["content"])

    def test_arac_hatasi_tool_mesajina_yazilir(self):
        settings = jarvis.Settings().normalize()
        client = jarvis.LMStudioClient(settings)
        calls = [jarvis.LLMEvent("tool_call", tool_id="c2", tool_name="get_weather", tool_arguments="{}")]

        def patla(name, args):
            raise RuntimeError("ag yok")

        messages = client._tool_followup([{"role": "user", "content": "x"}], calls, patla)
        self.assertIn("hata", messages[-1]["content"])
        self.assertIn("ag yok", messages[-1]["content"])


class FakeSpeaker:
    """Assistant testlerinde ses cikarmayan speaker yerine gecen hafif nesne."""

    engine_name = "fake"

    def feed(self, text):
        pass

    def finish(self, timeout=None):
        pass

    def stop(self):
        pass

    def resume(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
