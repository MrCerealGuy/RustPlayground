// -----------------------------------------------------------------------------
// ollama_chat2 - Your personal AI chat assistant.
//
// by Andreas Zahnleiter <a.zahnleiter@gmx.de>
// -----------------------------------------------------------------------------
// 2026-05-17 - az - created
// 2026-05-18 - az - added ollama installation process
// 2026-05-19 - az - added colored text and unicode symbols
// 2026-05-20 - az - check for Windows 11 
// 2026-06-06 - az - replaced stdin input with speech-to-text (whisper.cpp)
// -----------------------------------------------------------------------------

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use owo_colors::OwoColorize;

//-----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Message {
    role: String,
    content: String,
}

//-----------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OllamaChunk {
    message: Option<OllamaMessage>,
    done: Option<bool>,
}

//-----------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: Option<String>,
}

//-----------------------------------------------------------------------------

const WHISPER_MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin";
const WHISPER_MODEL_PATH: &str = "ggml-large-v3.bin";
const WHISPER_EXE_URL: &str =
    "https://github.com/ggerganov/whisper.cpp/releases/download/v1.8.6/whisper-bin-x64.zip";
const WHISPER_EXE_PATH: &str = "whisper.exe";
const WHISPER_ZIP_PATH: &str = "whisper-bin-x64.zip";
const TARGET_SAMPLE_RATE: u32 = 16000;

const PIPER_EXE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip";
const PIPER_DIR: &str = "piper";
const PIPER_ZIP_PATH: &str = "piper_windows_amd64.zip";
const PIPER_VOICE_URL: &str =
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/ramona/low/de_DE-ramona-low.onnx";
const PIPER_VOICE_JSON_URL: &str =
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/ramona/low/de_DE-ramona-low.onnx.json";

//-----------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn is_windows_11() -> bool {
    use winreg::enums::*;
    use winreg::RegKey;

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);

    let current_version = hklm
        .open_subkey("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion");

    if let Ok(key) = current_version {
        let build: Result<String, _> = key.get_value("CurrentBuild");

        if let Ok(build) = build {
            if let Ok(build_number) = build.parse::<u32>() {
                return build_number >= 22000;
            }
        }
    }

    return false;
}

//-----------------------------------------------------------------------------

fn ollama_is_installed() -> bool {
    let output = Command::new("ollama")
        .arg("--version")
        .output();

    match output {
        Ok(o) => o.status.success(),
        Err(_) => false,
    }
}

//-----------------------------------------------------------------------------

fn install_ollama() -> bool {
    let status = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://ollama.com/install.ps1 | iex",
        ])
        .status();

    match status {
        Ok(s) if s.success() => true,
        _ => false,
    }
}

//-----------------------------------------------------------------------------

fn phi4_is_installed() -> bool {
    let output = Command::new("ollama")
        .args(["list"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout.contains("phi4")
        }
        _ => false,
    }
}

//-----------------------------------------------------------------------------

fn install_phi4() -> bool {
    let status = Command::new("ollama")
        .args(["pull", "phi4"])
        .status();

    match status {
        Ok(s) => s.success(),
        Err(_) => false,
    }
}

//-----------------------------------------------------------------------------

async fn download_whisper_model(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new(WHISPER_MODEL_PATH);
    if model_path.exists() {
        println!("✅ Whisper model '{}' found.", WHISPER_MODEL_PATH);
        return Ok(());
    }

    // Remove partial download if present
    let _ = std::fs::remove_file(model_path);

    println!("{} Downloading Whisper model (~1.5 GB)...", "Model missing.".red());
    println!("   {}", WHISPER_MODEL_URL);

    let response = client.get(WHISPER_MODEL_URL).send().await?;
    let total = response.content_length().unwrap_or(0);

    let mut file = std::fs::File::create(model_path)?;
    let mut downloaded: u64 = 0;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            let pct = downloaded as f64 / total as f64 * 100.0;
            print!("\r{:.0}% ({:.1} MB / {:.1} MB)", pct, downloaded as f64 / 1_048_576.0, total as f64 / 1_048_576.0);
        } else {
            print!("\r{:.1} MB downloaded", downloaded as f64 / 1_048_576.0);
        }
        io::stdout().flush()?;
    }

    println!("\n✅ Whisper model downloaded.");
    Ok(())
}

//-----------------------------------------------------------------------------

async fn download_whisper_binary(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let exe_path = Path::new(WHISPER_EXE_PATH);
    if exe_path.exists() {
        println!("✅ Whisper binary found.");
        return Ok(());
    }

    println!("{} Downloading whisper.cpp binary...", "Binary missing.".red());
    println!("   {}", WHISPER_EXE_URL);

    let response = client.get(WHISPER_EXE_URL).send().await?;
    let bytes = response.bytes().await?;
    std::fs::write(WHISPER_ZIP_PATH, &bytes)?;

    println!("{} Extracting...", "Extracting zip.".yellow());

    let out_dir = "whisper_extracted";
    let extract = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!("Expand-Archive -Path '{}' -DestinationPath '{}' -Force", WHISPER_ZIP_PATH, out_dir),
        ])
        .output()?;

    if !extract.status.success() {
        let err = String::from_utf8_lossy(&extract.stderr);
        let _ = std::fs::remove_file(WHISPER_ZIP_PATH);
        return Err(format!("Extraction failed: {}", err).into());
    }

    // In v1.8.6, binaries are in a Release/ subfolder with whisper-cli.exe
    let release_dir = Path::new(out_dir).join("Release");
    let source_exe = release_dir.join("whisper-cli.exe");
    let source_dlls = ["whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"];

    if source_exe.exists() {
        std::fs::rename(&source_exe, WHISPER_EXE_PATH)?;
    } else {
        let _ = std::fs::remove_file(WHISPER_ZIP_PATH);
        let _ = std::fs::remove_dir_all(out_dir);
        return Err("whisper-cli.exe not found in extracted archive.".into());
    }

    // Copy needed DLLs alongside the exe
    for dll in &source_dlls {
        let src = release_dir.join(dll);
        if src.exists() {
            let _ = std::fs::copy(&src, dll);
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(WHISPER_ZIP_PATH);
    let _ = std::fs::remove_dir_all(out_dir);

    println!("✅ Whisper binary ready.");
    Ok(())
}

//-----------------------------------------------------------------------------

async fn download_piper_binary(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let piper_dir = Path::new(PIPER_DIR);
    let exe_path = piper_dir.join("piper.exe");

    if exe_path.exists() {
        println!("✅ Piper binary found.");
        return Ok(());
    }

    println!("{} Downloading Piper TTS binary (~22 MB)...", "Binary missing.".red());
    println!("   {}", PIPER_EXE_URL);

    let response = client.get(PIPER_EXE_URL).send().await?;
    let bytes = response.bytes().await?;
    std::fs::write(PIPER_ZIP_PATH, &bytes)?;

    println!("{} Extracting...", "Extracting zip.".yellow());

    let out_dir = "piper_extracted";
    let extract = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!("Expand-Archive -Path '{}' -DestinationPath '{}' -Force", PIPER_ZIP_PATH, out_dir),
        ])
        .output()?;

    if !extract.status.success() {
        let err = String::from_utf8_lossy(&extract.stderr);
        let _ = std::fs::remove_file(PIPER_ZIP_PATH);
        return Err(format!("Extraction failed: {}", err).into());
    }

    let extracted_piper = Path::new(out_dir).join("piper");

    if extracted_piper.exists() {
        if piper_dir.exists() {
            std::fs::remove_dir_all(piper_dir)?;
        }
        std::fs::rename(&extracted_piper, PIPER_DIR)?;
    } else {
        let _ = std::fs::remove_file(PIPER_ZIP_PATH);
        let _ = std::fs::remove_dir_all(out_dir);
        return Err("piper/ directory not found in extracted archive.".into());
    }

    let _ = std::fs::remove_file(PIPER_ZIP_PATH);
    let _ = std::fs::remove_dir_all(out_dir);

    println!("✅ Piper binary ready.");
    Ok(())
}

//-----------------------------------------------------------------------------

async fn download_piper_voice(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let piper_dir = Path::new(PIPER_DIR);
    let voice_path = piper_dir.join("de_DE-ramona-low.onnx");

    if voice_path.exists() {
        println!("✅ Piper German female voice model found.");
        return Ok(());
    }

    std::fs::create_dir_all(piper_dir)?;

    println!("{} Downloading German voice model (~63 MB)...", "Voice model missing.".red());
    println!("   {}", PIPER_VOICE_URL);

    let response = client.get(PIPER_VOICE_URL).send().await?;
    let bytes = response.bytes().await?;
    std::fs::write(&voice_path, &bytes)?;

    let json_path = piper_dir.join("de_DE-ramona-low.onnx.json");
    if !json_path.exists() {
        println!("   Downloading voice config...");
        if let Ok(resp) = client.get(PIPER_VOICE_JSON_URL).send().await {
            if let Ok(bytes) = resp.bytes().await {
                let _ = std::fs::write(&json_path, &bytes);
            }
        }
    }

    println!("✅ German voice model downloaded.");
    Ok(())
}

//-----------------------------------------------------------------------------

fn record_audio_push_to_talk() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    println!("{}", "\n🎤 Listening... (speak to record)".yellow());
    io::stdout().flush()?;

    record_audio_inner(None)
}

//-----------------------------------------------------------------------------

fn record_audio_interrupt(
    cancel: Arc<AtomicBool>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    record_audio_inner(Some(cancel))
}

//-----------------------------------------------------------------------------

fn record_audio_inner(
    cancel: Option<Arc<AtomicBool>>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device found")?;

    let supported_config = device.default_input_config()?;
    let sample_format = supported_config.sample_format();
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;
    let stream_config: cpal::StreamConfig = supported_config.into();

    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let is_speaking = Arc::new(AtomicBool::new(false));
    let utterance_done = Arc::new(AtomicBool::new(false));
    let last_speech = Arc::new(Mutex::new(std::time::Instant::now()));

    let c_ref = cancel.as_ref().map(Arc::clone);
    let c1 = c_ref.clone();
    let c2 = c_ref.clone();
    let c3 = c_ref;

    let s1 = (
        Arc::clone(&samples),
        Arc::clone(&is_speaking),
        Arc::clone(&utterance_done),
        Arc::clone(&last_speech),
        c1,
    );
    let s2 = (
        Arc::clone(&samples),
        Arc::clone(&is_speaking),
        Arc::clone(&utterance_done),
        Arc::clone(&last_speech),
        c2,
    );
    let s3 = (
        Arc::clone(&samples),
        Arc::clone(&is_speaking),
        Arc::clone(&utterance_done),
        Arc::clone(&last_speech),
        c3,
    );

    const VAD_THRESHOLD: f32 = 0.04;
    const CANCEL_THRESHOLD: f32 = 0.08;
    const SILENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1500);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let (buf, spk, done, last, cancel_flag) = s1;
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let peak = data.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
                    if peak > VAD_THRESHOLD {
                        if let Some(ref c) = cancel_flag {
                            if peak > CANCEL_THRESHOLD {
                                c.store(true, Ordering::Relaxed);
                            }
                        }
                        spk.store(true, Ordering::Relaxed);
                        *last.lock().unwrap() = std::time::Instant::now();
                        buf.lock().unwrap().extend_from_slice(data);
                    } else if spk.load(Ordering::Relaxed) {
                        buf.lock().unwrap().extend_from_slice(data);
                        if last.lock().unwrap().elapsed() >= SILENCE_TIMEOUT {
                            done.store(true, Ordering::Relaxed);
                        }
                    }
                },
                move |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }

        cpal::SampleFormat::I16 => {
            let (buf, spk, done, last, cancel_flag) = s2;
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let peak = data
                        .iter()
                        .map(|&s| (s.abs() as f32) / i16::MAX as f32)
                        .fold(0.0f32, f32::max);
                    if peak > VAD_THRESHOLD {
                        if let Some(ref c) = cancel_flag {
                            if peak > CANCEL_THRESHOLD {
                                c.store(true, Ordering::Relaxed);
                            }
                        }
                        spk.store(true, Ordering::Relaxed);
                        *last.lock().unwrap() = std::time::Instant::now();
                        let mut b = buf.lock().unwrap();
                        for &s in data {
                            b.push(s as f32 / i16::MAX as f32);
                        }
                    } else if spk.load(Ordering::Relaxed) {
                        let mut b = buf.lock().unwrap();
                        for &s in data {
                            b.push(s as f32 / i16::MAX as f32);
                        }
                        if last.lock().unwrap().elapsed() >= SILENCE_TIMEOUT {
                            done.store(true, Ordering::Relaxed);
                        }
                    }
                },
                move |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }

        cpal::SampleFormat::U16 => {
            let (buf, spk, done, last, cancel_flag) = s3;
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let peak = data
                        .iter()
                        .map(|&s| {
                            ((s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)).abs()
                        })
                        .fold(0.0f32, f32::max);
                    if peak > VAD_THRESHOLD {
                        if let Some(ref c) = cancel_flag {
                            if peak > CANCEL_THRESHOLD {
                                c.store(true, Ordering::Relaxed);
                            }
                        }
                        spk.store(true, Ordering::Relaxed);
                        *last.lock().unwrap() = std::time::Instant::now();
                        let mut b = buf.lock().unwrap();
                        for &s in data {
                            b.push((s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0));
                        }
                    } else if spk.load(Ordering::Relaxed) {
                        let mut b = buf.lock().unwrap();
                        for &s in data {
                            b.push((s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0));
                        }
                        if last.lock().unwrap().elapsed() >= SILENCE_TIMEOUT {
                            done.store(true, Ordering::Relaxed);
                        }
                    }
                },
                move |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }

        other => return Err(format!("Unsupported sample format: {:?}", other).into()),
    };

    stream.play()?;

    while !utterance_done.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    drop(stream);
    let recorded = samples.lock().unwrap().clone();

    if recorded.is_empty() {
        return Err("No audio recorded.".into());
    }

    let mono: Vec<f32> = if channels > 1 {
        recorded
            .chunks_exact(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        recorded
    };

    let resampled = if sample_rate != TARGET_SAMPLE_RATE {
        resample(&mono, sample_rate, TARGET_SAMPLE_RATE)
    } else {
        mono
    };

    Ok(resampled)
}

//-----------------------------------------------------------------------------

fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (input.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = (i as f64 / ratio) as usize;
        let src_idx = src_idx.min(input.len().saturating_sub(1));
        let frac = (i as f64 / ratio) - src_idx as f64;

        let next_idx = (src_idx + 1).min(input.len().saturating_sub(1));
        let sample = input[src_idx] * (1.0 - frac as f32) + input[next_idx] * frac as f32;
        output.push(sample);
    }

    output
}

//-----------------------------------------------------------------------------

fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16)?;
    }

    writer.finalize()?;
    Ok(())
}

//-----------------------------------------------------------------------------

fn transcribe_via_whisper(wav_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let whisper_path = cwd.join(WHISPER_EXE_PATH);
    let model_path = cwd.join(WHISPER_MODEL_PATH);

    let output = Command::new(&whisper_path)
        .args([
            "-m", &model_path.to_string_lossy(),
            "-f", wav_path,
            "-nt",
            "-np",
            "-l", "de",
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("whisper.cpp failed: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let text = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();

    Ok(text)
}

//-----------------------------------------------------------------------------

fn compute_vu_levels(wav_path: &str) -> Vec<f32> {
    let mut levels = Vec::new();
    let mut reader = match hound::WavReader::open(wav_path) {
        Ok(r) => r,
        Err(_) => return levels,
    };
    let spec = reader.spec();
    let frame_samples = (spec.sample_rate as usize / 20).max(1) * spec.channels as usize;
    let mut chunk = Vec::with_capacity(frame_samples);

    match spec.sample_format {
        hound::SampleFormat::Int => {
            let divisor = match spec.bits_per_sample {
                8 => i16::from(i8::MAX) as f32,
                16 => i16::MAX as f32,
                24 => 8388607.0f32,
                32 => 2147483647.0f32,
                _ => i16::MAX as f32,
            };
            for sample in reader.samples::<i16>() {
                if let Ok(s) = sample {
                    chunk.push(s as f32 / divisor);
                    if chunk.len() >= frame_samples {
                        let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        levels.push(peak);
                        chunk.clear();
                    }
                }
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                if let Ok(s) = sample {
                    chunk.push(s);
                    if chunk.len() >= frame_samples {
                        let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        levels.push(peak);
                        chunk.clear();
                    }
                }
            }
        }
    }

    if !chunk.is_empty() {
        let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        levels.push(peak);
    }
    levels
}

fn vu_meter_bar(level: f32, width: usize) -> String {
    let filled = (level * width as f32).min(width as f32).round() as usize;
    let empty = width - filled;
    let bar: String = std::iter::repeat('█').take(filled).collect();
    let space: String = std::iter::repeat('░').take(empty).collect();
    format!("{}{}", bar, space)
}

fn speak_text(text: &str, cancel: &AtomicBool) -> Result<(), Box<dyn std::error::Error>> {
    if text.trim().is_empty() {
        return Ok(());
    }

    let (tx_stop, rx_stop) = std::sync::mpsc::channel::<()>();
    let _listener = std::thread::spawn(move || {
        let mut _buf = String::new();
        if io::stdin().read_line(&mut _buf).is_ok() {
            let _ = tx_stop.send(());
        }
    });

    let interrupted = || -> bool {
        rx_stop.try_recv().is_ok() || cancel.load(Ordering::Relaxed)
    };

    let cwd = std::env::current_dir()?;
    let piper_exe = cwd.join(PIPER_DIR).join("piper.exe");
    let piper_voice = cwd.join(PIPER_DIR).join("de_DE-ramona-low.onnx");

    if piper_exe.exists() && piper_voice.exists() {
        let temp_wav = std::env::temp_dir().join("ollama_chat_tts.wav");
        let wav_path = temp_wav.to_string_lossy().to_string();

        let mut piper_proc = Command::new(&piper_exe)
            .args([
                "--model", &piper_voice.to_string_lossy(),
                "--output-file", &wav_path,
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        if let Some(mut stdin) = piper_proc.stdin.take() {
            stdin.write_all(text.as_bytes())?;
        }
        let _ = piper_proc.wait();

        let mut play = Command::new("powershell")
            .args([
                "-NoProfile", "-NonInteractive", "-Command",
                &format!("(New-Object System.Media.SoundPlayer '{}').PlaySync()", wav_path),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let vu_levels = compute_vu_levels(&wav_path);
        let vu_total = vu_levels.len();
        let vu_start = std::time::Instant::now();
        let mut last_display_vu = 0usize;
        let mut vu_printed = false;

        loop {
            if let Ok(Some(_)) = play.try_wait() {
                break;
            }
            if interrupted() {
                let _ = play.kill();
                break;
            }

            if vu_total > 0 {
                let playback_frame = (vu_start.elapsed().as_millis() as usize) / 50;
                let target_idx = playback_frame.min(vu_total.saturating_sub(1));

                if target_idx > last_display_vu {
                    last_display_vu = target_idx;
                    if !vu_printed {
                        vu_printed = true;
                        println!();
                        print!("\x1b[s");
                    }
                    let level = vu_levels[target_idx];
                    let bar = vu_meter_bar(level, 24);
                    let pct = (level * 100.0).min(100.0) as u8;
                    let display: String = if pct > 70 {
                        bar.red().to_string()
                    } else if pct > 40 {
                        bar.yellow().to_string()
                    } else {
                        bar.green().to_string()
                    };
                    print!("\x1b[u\x1b[2K🔊 {} {}%", display, pct);
                    io::stdout().flush()?;
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        print!("\r{}", " ".repeat(40));

        let _ = std::fs::remove_file(&temp_wav);
    } else {
        let mut child = Command::new("powershell")
            .args([
                "-NoProfile", "-NonInteractive", "-Command",
                "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak([Console]::In.ReadToEnd())",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes())?;
        }

        loop {
            if let Ok(Some(_)) = child.try_wait() {
                break;
            }
            if interrupted() {
                let _ = child.kill();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    Ok(())
}

//-----------------------------------------------------------------------------

fn is_meaningful_speech(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    let lower = t.to_lowercase();
    if lower.contains("[blank_audio]") || lower.contains("[music") || lower.contains("[laughter]")
        || lower.contains("[sound") || lower.contains("[noise]")
    {
        return false;
    }
    true
}

//-----------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Check for Windows 11
    #[cfg(target_os = "windows")]
    {
        if is_windows_11() {
            println!("✅ Windows 11 detected.");
        } else {
            println!("{}", "❌ No Windows 11!".red());
            std::process::exit(1);
        }
    }

    // Check for ollama
    if ollama_is_installed() {
        println!("✅ Ollama is already installed.");
    } else {
        println!("{} Installing Ollama!", "Ollama not found!".red());

        if install_ollama() {
            println!("✅ Installation successful!");
        } else {
            eprintln!("{}", "❌ Installation failed!".red());
            std::process::exit(1);
        }        
    }

    // Check for AI model 'phi4'
    if phi4_is_installed() {
        println!("✅ Model 'phi4' is already installed.");
    } else {
        println!("{} Installing model!", "Model 'phi4' is missing.".red());

        if install_phi4() {
            println!("✅ Model 'phi4' successfully installed.");
        } else {
            eprintln!("{}", "❌ Installation failed!".red());
            std::process::exit(1);
        }
    }

    // Download whisper model + binary for STT
    let client = Client::new();
    download_whisper_model(&client).await?;
    download_whisper_binary(&client).await?;

    // Download Piper TTS binary and German voice model (optional; falls back to System.Speech)
    if let Err(e) = download_piper_binary(&client).await {
        eprintln!("{} {} {}", "Warning:".yellow(), "Piper TTS binary download failed:".yellow(), e);
    }
    if let Err(e) = download_piper_voice(&client).await {
        eprintln!("{} {} {}", "Warning:".yellow(), "Piper voice model download failed:".yellow(), e);
    }

    println!("\nOllama Streaming Chat (Speech-to-Text)");
    println!("Type 'exit' to quit.");

    let system_content = "Du bist ein hilfreicher Assistent. Antworte immer in natürlicher, gesprächsorientierter Sprache wie ein Mensch. Vermeide Aufzählungen, Listen, Programmcode, mathematische Formeln, Tabellen und jede Art von strukturierter Darstellung. Deine Antworten sollen sich anhören wie ein normales Gespräch unter Freunden.";

    let mut history: Vec<Message> = vec![
        Message {
            role: "system".to_string(),
            content: system_content.to_string(),
        }
    ];

    let client = Client::new();

    let mut audio = record_audio_push_to_talk()?;

    loop {
        let wav_path = std::env::temp_dir().join("ollama_chat_input.wav");
        let wav_str = wav_path.to_str().ok_or("Invalid temp path")?;

        if let Err(e) = save_wav(wav_str, &audio, TARGET_SAMPLE_RATE) {
            eprintln!("{} {}", "WAV error:".red(), e);
            audio = record_audio_push_to_talk()?;
            continue;
        }

        print!("{}", "Transcribing...".blue());
        io::stdout().flush()?;

        let input = match transcribe_via_whisper(wav_str) {
            Ok(text) => {
                println!("\r{}", " ".repeat(80));
                println!("{} {}", "You:".green(), text);
                text
            }
            Err(e) => {
                eprintln!("\r{} {}", "STT error:".red(), e);
                audio = record_audio_push_to_talk()?;
                continue;
            }
        };

        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        if !is_meaningful_speech(&input) {
            std::thread::sleep(std::time::Duration::from_millis(500));
            audio = record_audio_push_to_talk()?;
            continue;
        }

        history.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Ollama request
        let body = json!({
            "model": "phi4",
            "stream": true,
            "messages": history
        });

        let response = client
            .post("http://localhost:11434/api/chat")
            .json(&body)
            .send()
            .await?;

        let mut stream = response.bytes_stream();

        println!("\n{}\n", "Assistant:".blue());

        let mut full_response = String::new();

        // Stream processing
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;

            let text = String::from_utf8_lossy(&chunk);

            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<OllamaChunk>(&line) {
                    if let Some(msg) = parsed.message {
                        if let Some(content) = msg.content {
                            print!("{content}");
                            io::stdout().flush()?;

                            full_response.push_str(&content);
                        }
                    }

                    if parsed.done == Some(true) {
                        break;
                    }
                }
            }
        }

        println!("\n");

        // Start TTS in background, then listen for next user input
        let tts_cancel = Arc::new(AtomicBool::new(false));
        let tts_text = full_response.clone();
        let tts_cancel_vad = Arc::clone(&tts_cancel);
        let tts_handle = std::thread::spawn(move || {
            if let Err(e) = speak_text(&tts_text, &tts_cancel) {
                eprintln!("{} {}", "TTS error:".red(), e);
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(500));

        print!("{}", "\n🎤 Listening... (speak to interrupt or respond)".yellow());
        io::stdout().flush()?;

        audio = match record_audio_interrupt(Arc::clone(&tts_cancel_vad)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("{} {}", "Recording error:".red(), e);
                tts_cancel_vad.store(true, Ordering::Relaxed);
                let _ = tts_handle.join();
                audio = record_audio_push_to_talk()?;
                continue;
            }
        };

        let _ = tts_handle.join();

        // Save assistant message
        history.push(Message {
            role: "assistant".to_string(),
            content: full_response,
        });
    }

    println!("\nGoodbye! 👋");

    Ok(())
}

//-----------------------------------------------------------------------------
