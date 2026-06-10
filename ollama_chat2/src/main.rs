use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, Paragraph, Wrap};
use ratatui::Terminal;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use owo_colors::OwoColorize;

//-----------------------------------------------------------------------------
// Constants
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
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/kerstin/low/de_DE-kerstin-low.onnx";
const PIPER_VOICE_JSON_URL: &str =
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/kerstin/low/de_DE-kerstin-low.onnx.json";

//const SYSTEM_PROMPT: &str = "Du bist ein hilfreicher Assistent, der ausschließlich in natürlich gesprochener Sprache antwortet. STRIKTES VERBOT von: Programmcode (egal welche Sprache), JavaScript, Python, HTML, CSS, Shell-Befehle, SQL, mathematische Formeln, LaTeX, Aufzählungen, Listen, nummerierte Schritte, Tabellen, Sternchen-Aufzählungen, Gedankenstriche, Sonderzeichen oder Formatierungen. Erkläre Konzepte in ganzen Sätzen ohne Beispiele in Code. Antworte als würdest du mit einem Freund sprechen – fließend, natürlich und direkt vorlesbar.";
const SYSTEM_PROMPT: &str = "";

//-----------------------------------------------------------------------------
// Types
//-----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChunk {
    message: Option<OllamaMessage>,
    done: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: Option<String>,
}

#[derive(Debug, Clone)]
struct ChatEntry {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Listening,
    Transcribing,
    Thinking,
    Speaking,
}

struct AppState {
    messages: Vec<ChatEntry>,
    phase: Phase,
    status_text: String,
    vad_level: f32,
    vu_level: f32,
    transcription_progress: u16,
    error: Option<String>,
    search_in_progress: bool,
    search_query: String,
    search_result: String,
    search_show_until: Option<std::time::Instant>,
}

impl AppState {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            phase: Phase::Listening,
            status_text: "Listening...".into(),
            vad_level: 0.0,
            vu_level: 0.0,
            transcription_progress: 0,
            error: None,
            search_in_progress: false,
            search_query: String::new(),
            search_result: String::new(),
            search_show_until: None,
        }
    }
}

enum UiCommand {
    Exit,
}

//-----------------------------------------------------------------------------
// System checks
//-----------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn is_windows_11() -> bool {
    use winreg::enums::*;
    use winreg::RegKey;
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    if let Ok(key) = hklm.open_subkey("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion") {
        if let Ok(build) = key.get_value::<String, _>("CurrentBuild") {
            if let Ok(build_number) = build.parse::<u32>() {
                return build_number >= 22000;
            }
        }
    }
    false
}

fn ollama_is_installed() -> bool {
    Command::new("ollama").arg("--version").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn install_ollama() -> bool {
    Command::new("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
               "irm https://ollama.com/install.ps1 | iex"])
        .status().map(|s| s.success()).unwrap_or(false)
}

fn phi4_is_installed() -> bool {
    Command::new("ollama").args(["list"]).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("phi4")).unwrap_or(false)
}

fn install_phi4() -> bool {
    Command::new("ollama").args(["pull", "phi4"])
        .status().map(|s| s.success()).unwrap_or(false)
}

//-----------------------------------------------------------------------------
// Downloads
//-----------------------------------------------------------------------------

async fn download_whisper_model(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new(WHISPER_MODEL_PATH);
    if model_path.exists() {
        println!("✅ Whisper model '{}' found.", WHISPER_MODEL_PATH);
        return Ok(());
    }
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
        .args(["-NoProfile", "-Command",
               &format!("Expand-Archive -Path '{}' -DestinationPath '{}' -Force", WHISPER_ZIP_PATH, out_dir)])
        .output()?;
    if !extract.status.success() {
        let err = String::from_utf8_lossy(&extract.stderr);
        let _ = std::fs::remove_file(WHISPER_ZIP_PATH);
        return Err(format!("Extraction failed: {}", err).into());
    }
    let release_dir = Path::new(out_dir).join("Release");
    let source_exe = release_dir.join("whisper-cli.exe");
    if source_exe.exists() {
        std::fs::rename(&source_exe, WHISPER_EXE_PATH)?;
    } else {
        let _ = std::fs::remove_file(WHISPER_ZIP_PATH);
        let _ = std::fs::remove_dir_all(out_dir);
        return Err("whisper-cli.exe not found in extracted archive.".into());
    }
    for dll in &["whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"] {
        let src = release_dir.join(dll);
        if src.exists() {
            let _ = std::fs::copy(&src, dll);
        }
    }
    let _ = std::fs::remove_file(WHISPER_ZIP_PATH);
    let _ = std::fs::remove_dir_all(out_dir);
    println!("✅ Whisper binary ready.");
    Ok(())
}

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
        .args(["-NoProfile", "-Command",
               &format!("Expand-Archive -Path '{}' -DestinationPath '{}' -Force", PIPER_ZIP_PATH, out_dir)])
        .output()?;
    if !extract.status.success() {
        let err = String::from_utf8_lossy(&extract.stderr);
        let _ = std::fs::remove_file(PIPER_ZIP_PATH);
        return Err(format!("Extraction failed: {}", err).into());
    }
    let extracted_piper = Path::new(out_dir).join("piper");
    if extracted_piper.exists() {
        if piper_dir.exists() { std::fs::remove_dir_all(piper_dir)?; }
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

async fn download_piper_voice(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let piper_dir = Path::new(PIPER_DIR);
    let voice_path = piper_dir.join("de_DE-kerstin-low.onnx");
    if voice_path.exists() {
        println!("✅ Piper Kerstin voice model found.");
        return Ok(());
    }
    std::fs::create_dir_all(piper_dir)?;
    println!("{} Downloading Kerstin voice model (~63 MB)...", "Voice model missing.".red());
    println!("   {}", PIPER_VOICE_URL);
    let response = client.get(PIPER_VOICE_URL).send().await?;
    let bytes = response.bytes().await?;
    std::fs::write(&voice_path, &bytes)?;
    let json_path = piper_dir.join("de_DE-kerstin-low.onnx.json");
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
// Audio helpers
//-----------------------------------------------------------------------------

fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate { return input.to_vec(); }
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

fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    let amplitude = i16::MAX as f32;
    for &sample in samples {
        writer.write_sample((sample * amplitude) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn transcribe_via_whisper(wav_path: &str, state: Arc<Mutex<AppState>>) -> Result<String, Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let whisper_path = cwd.join(WHISPER_EXE_PATH);
    let model_path = cwd.join(WHISPER_MODEL_PATH);
    let mut child = Command::new(&whisper_path)
        .args(["-m", &model_path.to_string_lossy(), "-f", wav_path, "-nt", "-pp", "-l", "de"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    // Read stdout for transcription text
    let stdout = child.stdout.take().unwrap();
    let read_stdout = std::thread::spawn(move || {
        let reader = io::BufReader::new(stdout);
        let mut lines = Vec::new();
        for line in reader.lines() {
            if let Ok(line) = line { lines.push(line); }
        }
        lines
    });

    // Read stderr for progress (whisper.cpp prints progress to stderr with -pp flag)
    let stderr = child.stderr.take().unwrap();
    let progress_state = Arc::clone(&state);
    let read_stderr = std::thread::spawn(move || {
        let reader = io::BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                if let Some(pct) = line.split("progress =").nth(1)
                    .and_then(|s| s.trim().trim_end_matches('%').parse::<u16>().ok())
                    .map(|v| v.min(100))
                {
                    if let Ok(mut s) = progress_state.lock() {
                        s.transcription_progress = pct;
                    }
                }
            }
        }
    });

    let status = child.wait()?;
    let trans_lines = read_stdout.join().unwrap();
    read_stderr.join().unwrap();

    if !status.success() {
        return Err(format!("whisper.cpp failed with exit code {:?}", status.code()).into());
    }

    let text = trans_lines.iter().map(|s| s.as_str()).filter(|l| !l.trim().is_empty()).collect::<Vec<_>>().join(" ").trim().to_string();
    Ok(text)
}

fn is_meaningful_speech(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() { return false; }
    let lower = t.to_lowercase();
    !(lower.contains("[blank_audio]") || lower.contains("[music") || lower.contains("[laughter]")
        || lower.contains("[sound") || lower.contains("[noise]"))
}

//-----------------------------------------------------------------------------
// Web search
//-----------------------------------------------------------------------------

fn search_web(query: &str, client: &Client, rt: &tokio::runtime::Runtime) -> String {
    // WEBFETCH: prefix → fetch and extract text from a URL
    if let Some(url) = query.strip_prefix("WEBFETCH:") {
        return fetch_url_text(url, client, rt);
    }

    let lower = query.to_lowercase();

    // Weather query → use Open-Meteo API (free, no key)
    if lower.starts_with("wetter ") || lower.starts_with("weather ") {
        let location = query.splitn(2, ' ').nth(1).unwrap_or("").trim();
        if !location.is_empty() {
            return search_weather(location, client, rt);
        }
    }

    // General query → DuckDuckGo Instant Answer API
    let encoded = urlencoding::encode(query);
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        encoded
    );

    let result = rt.block_on(async { client.get(&url).send().await });
    let body = match result {
        Ok(resp) => match rt.block_on(async { resp.text().await }) {
            Ok(t) => t,
            Err(e) => return format!("Fehler beim Lesen der Suchergebnisse: {}", e),
        },
        Err(e) => return format!("Fehler bei der Suchanfrage: {}", e),
    };

    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return format!("Keine Suchergebnisse für '{}' gefunden.", query),
    };

    let mut parts: Vec<String> = Vec::new();

    if let Some(answer) = json["Answer"].as_str() {
        if !answer.is_empty() {
            parts.push(format!("Antwort: {}", answer));
        }
    }

    if let Some(abstract_text) = json["Abstract"].as_str() {
        if !abstract_text.is_empty() {
            parts.push(format!("Zusammenfassung: {}", abstract_text));
            if let Some(source) = json["AbstractSource"].as_str() {
                if !source.is_empty() {
                    parts.push(format!("Quelle: {}", source));
                }
            }
        }
    }

    if let Some(topics) = json["RelatedTopics"].as_array() {
        let mut count = 0;
        for topic in topics {
            if count >= 5 { break; }
            if let Some(text) = topic["Text"].as_str() {
                parts.push(format!("- {}", text));
                count += 1;
            }
            if let Some(subtopics) = topic["Topics"].as_array() {
                for sub in subtopics {
                    if count >= 5 { break; }
                    if let Some(text) = sub["Text"].as_str() {
                        parts.push(format!("- {}", text));
                        count += 1;
                    }
                }
            }
        }
    }

    if parts.is_empty() {
        format!("Keine Suchergebnisse für '{}' gefunden.", query)
    } else {
        parts.join("\n")
    }
}

fn fetch_url_text(url: &str, client: &Client, rt: &tokio::runtime::Runtime) -> String {
    let result = rt.block_on(async { client.get(url).send().await });
    let body = match result {
        Ok(resp) => match rt.block_on(async { resp.text().await }) {
            Ok(t) => t,
            Err(e) => return format!("Fehler beim Abrufen der Seite: {}", e),
        },
        Err(e) => return format!("Fehler beim Abrufen der Seite: {}", e),
    };

    let text = strip_html(&body);
    let max_len = 8000;
    if text.len() > max_len {
        format!("{}\n\n[Text gekürzt auf {} Zeichen]", &text[..max_len], max_len)
    } else {
        text
    }
}

fn strip_html(html: &str) -> String {
    let re = regex::Regex::new(r"(?i)<script[^>]*>[\s\S]*?</script>").unwrap();
    let s = re.replace_all(html, " ");
    let re = regex::Regex::new(r"(?i)<style[^>]*>[\s\S]*?</style>").unwrap();
    let s = re.replace_all(&s, " ");
    let re = regex::Regex::new(r"<[^>]+>").unwrap();
    let s = re.replace_all(&s, " ");
    let s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", "\"").replace("&nbsp;", " ")
        .replace("&auml;", "ä").replace("&ouml;", "ö").replace("&uuml;", "ü")
        .replace("&Auml;", "Ä").replace("&Ouml;", "Ö").replace("&Uuml;", "Ü")
        .replace("&szlig;", "ß");
    let re = regex::Regex::new(r"\s+").unwrap();
    let s = re.replace_all(&s, " ");
    s.trim().to_string()
}

fn is_likely_domain(s: &str) -> bool {
    s.contains('.') && !s.contains(' ') && s.len() >= 4
}

fn search_weather(location: &str, client: &Client, rt: &tokio::runtime::Runtime) -> String {
    // Geocoding
    let encoded = urlencoding::encode(location);
    let geo_url = format!(
        "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=de&format=json",
        encoded
    );

    let body = match rt.block_on(async { client.get(&geo_url).send().await }) {
        Ok(r) => match rt.block_on(async { r.text().await }) {
            Ok(t) => t,
            Err(e) => return format!("Fehler bei der Geokodierung: {}", e),
        },
        Err(e) => return format!("Fehler bei der Geokodierung: {}", e),
    };

    let geo: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return format!("Konnte '{}' nicht finden.", location),
    };

    let results = match geo["results"].as_array() {
        Some(r) if !r.is_empty() => r,
        _ => return format!("Konnte '{}' nicht finden.", location),
    };

    let lat = results[0]["latitude"].as_f64().unwrap_or(0.0);
    let lon = results[0]["longitude"].as_f64().unwrap_or(0.0);
    let name = results[0]["name"].as_str().unwrap_or(location);
    let country = results[0]["country"].as_str().unwrap_or("");
    let region = results[0]["admin1"].as_str().unwrap_or("");

    // Weather forecast
    let weather_url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode&timezone=Europe/Berlin&forecast_days=3",
        lat, lon
    );

    let body = match rt.block_on(async { client.get(&weather_url).send().await }) {
        Ok(r) => match rt.block_on(async { r.text().await }) {
            Ok(t) => t,
            Err(e) => return format!("Fehler bei der Wetterabfrage: {}", e),
        },
        Err(e) => return format!("Fehler bei der Wetterabfrage: {}", e),
    };

    let weather: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return format!("Fehler beim Parsen der Wetterdaten."),
    };

    let mut parts = Vec::new();
    parts.push(format!("Wetter für {}, {} ({})", name, region, country));

    if let Some(current) = weather["current_weather"].as_object() {
        let temp = current["temperature"].as_f64().unwrap_or(0.0);
        let windspeed = current["windspeed"].as_f64().unwrap_or(0.0);
        let wcode = current["weathercode"].as_i64().unwrap_or(0);
        parts.push(format!(
            "Aktuell: {}°C, {}, Wind: {} km/h",
            temp, weather_code_desc(wcode), windspeed
        ));
    }

    if let Some(daily) = weather["daily"].as_object() {
        let times = daily["time"].as_array();
        let max_temps = daily["temperature_2m_max"].as_array();
        let min_temps = daily["temperature_2m_min"].as_array();
        let precip = daily["precipitation_sum"].as_array();
        let wcodes = daily["weathercode"].as_array();

        if let Some(times) = times {
            for i in 0..times.len().min(3) {
                let date = times[i].as_str().unwrap_or("");
                let t_max = max_temps.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let t_min = min_temps.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let p = precip.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let wc = wcodes.and_then(|a| a.get(i)).and_then(|v| v.as_i64()).unwrap_or(0);
                parts.push(format!(
                    "{}: {}°C - {}°C, {}, Niederschlag: {} mm",
                    date, t_min, t_max, weather_code_desc(wc), p
                ));
            }
        }
    }

    parts.join("\n")
}

fn weather_code_desc(code: i64) -> &'static str {
    match code {
        0 => "Klarer Himmel",
        1 => "Überwiegend klar",
        2 => "Teilweise bewölkt",
        3 => "Bedeckt",
        45 | 48 => "Nebel",
        51 => "Leichter Nieselregen",
        53 => "Mäßiger Nieselregen",
        55 => "Starker Nieselregen",
        56 | 57 => "Gefrierender Nieselregen",
        61 => "Leichter Regen",
        63 => "Mäßiger Regen",
        65 => "Starker Regen",
        66 | 67 => "Gefrierender Regen",
        71 => "Leichter Schneefall",
        73 => "Mäßiger Schneefall",
        75 => "Starker Schneefall",
        77 => "Schneekörner",
        80 => "Leichte Regenschauer",
        81 => "Mäßige Regenschauer",
        82 => "Starke Regenschauer",
        85 => "Leichte Schneeschauer",
        86 => "Starke Schneeschauer",
        95 => "Gewitter",
        96 => "Gewitter mit leichtem Hagel",
        99 => "Gewitter mit starkem Hagel",
        _ => "Unbekannt",
    }
}

fn detect_search_intent(text: &str) -> Option<String> {
    let lower = text.to_lowercase();

    // Direct "such nach" / "suche nach" commands
    for keyword in &["such nach ", "suche nach ", "such mal nach ", "google "] {
        if let Some(idx) = lower.find(keyword) {
            let query = text[idx + keyword.len()..].trim().to_string();
            if !query.is_empty() { return Some(query); }
        }
    }

    // URL + summarize/fetch pattern: "fass ... auf <domain>" / "zusammenfassung ... <domain>"
    let wants_fetch = lower.contains("zusammenfass") || lower.contains("neuigkeit")
        || lower.contains("nachricht") || lower.contains("aktuell")
        || lower.contains("recherchier") || lower.contains("artikel");
    let has_url_prefix = [" auf ", " von ", " von der "];
    if wants_fetch {
        for prefix in &has_url_prefix {
            if let Some(idx) = lower.find(prefix) {
                let after = text[idx + prefix.len()..].trim().trim_end_matches('?').trim();
                let domain = after.split_whitespace().next().unwrap_or("");
                if is_likely_domain(domain) {
                    let scheme = if domain.starts_with("http") { "" } else { "https://" };
                    return Some(format!("WEBFETCH:{}{}", scheme, domain));
                }
            }
        }
    }

    // Direct URL mention without a summarize keyword
    for prefix in &[" auf ", " von ", " von der "] {
        if let Some(idx) = lower.find(prefix) {
            let after = text[idx + prefix.len()..].trim().trim_end_matches('?').trim();
            let domain = after.split_whitespace().next().unwrap_or("");
            if is_likely_domain(domain) {
                let scheme = if domain.starts_with("http") { "" } else { "https://" };
                return Some(format!("WEBFETCH:{}{}", scheme, domain));
            }
        }
    }

    // Weather questions with location extraction (first "in" match)
    if lower.contains("wetter") {
        if let Some(idx) = lower.find(" in ") {
            let location = text[idx + 4..].trim().trim_end_matches('?').trim();
            if !location.is_empty() {
                return Some(format!("Wetter {}", location));
            }
        }
        return Some(format!("Wetter {}", text.trim_end_matches('?')));
    }

    // Questions about current events, news (fallback – search via DuckDuckGo)
    if lower.contains("nachricht") || lower.contains("aktuell") || lower.contains("neuigkeit") {
        return Some(text.trim_end_matches('?').to_string());
    }

    // Explicit "recherchiere" / "suche" / "googel"
    if lower.contains("recherchier") || lower.contains("googel") || lower == "suche" {
        return Some(text.trim_end_matches('?').to_string());
    }

    None
}

fn record_audio_inner(cancel: Option<Arc<AtomicBool>>, state: Arc<Mutex<AppState>>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No input device found")?;
    let supported_config = device.default_input_config()?;
    let sample_format = supported_config.sample_format();
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;
    let stream_config: cpal::StreamConfig = supported_config.into();

    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let is_speaking = Arc::new(AtomicBool::new(false));
    let utterance_done = Arc::new(AtomicBool::new(false));
    let last_speech = Arc::new(Mutex::new(std::time::Instant::now()));

    const VAD_THRESHOLD: f32 = 0.04;
    const CANCEL_THRESHOLD: f32 = 0.08;
    const SILENCE_TIMEOUT: Duration = Duration::from_millis(1500);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let buf = Arc::clone(&samples);
            let spk = Arc::clone(&is_speaking);
            let done = Arc::clone(&utterance_done);
            let last = Arc::clone(&last_speech);
            let state_ref = Arc::clone(&state);
            let c = cancel.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let peak = data.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
                    if let Ok(mut s) = state_ref.try_lock() { s.vad_level = peak; }
                    if let Some(ref c) = c { if peak > CANCEL_THRESHOLD { c.store(true, Ordering::Relaxed); } }
                    if peak > VAD_THRESHOLD {
                        spk.store(true, Ordering::Relaxed);
                        *last.lock().unwrap() = std::time::Instant::now();
                        buf.lock().unwrap().extend_from_slice(data);
                    } else if spk.load(Ordering::Relaxed) {
                        buf.lock().unwrap().extend_from_slice(data);
                        if last.lock().unwrap().elapsed() >= SILENCE_TIMEOUT { done.store(true, Ordering::Relaxed); }
                    }
                },
                move |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let buf = Arc::clone(&samples);
            let spk = Arc::clone(&is_speaking);
            let done = Arc::clone(&utterance_done);
            let last = Arc::clone(&last_speech);
            let state_ref = Arc::clone(&state);
            let c = cancel.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let peak = data.iter().map(|&s| (s.abs() as f32) / i16::MAX as f32).fold(0.0f32, f32::max);
                    if let Ok(mut s) = state_ref.try_lock() { s.vad_level = peak; }
                    if let Some(ref c) = c { if peak > CANCEL_THRESHOLD { c.store(true, Ordering::Relaxed); } }
                    if peak > VAD_THRESHOLD {
                        spk.store(true, Ordering::Relaxed);
                        *last.lock().unwrap() = std::time::Instant::now();
                        let mut b = buf.lock().unwrap();
                        for &s in data { b.push(s as f32 / i16::MAX as f32); }
                    } else if spk.load(Ordering::Relaxed) {
                        let mut b = buf.lock().unwrap();
                        for &s in data { b.push(s as f32 / i16::MAX as f32); }
                        if last.lock().unwrap().elapsed() >= SILENCE_TIMEOUT { done.store(true, Ordering::Relaxed); }
                    }
                },
                move |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let buf = Arc::clone(&samples);
            let spk = Arc::clone(&is_speaking);
            let done = Arc::clone(&utterance_done);
            let last = Arc::clone(&last_speech);
            let state_ref = Arc::clone(&state);
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let peak = data.iter().map(|&s| ((s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)).abs()).fold(0.0f32, f32::max);
                    if let Ok(mut s) = state_ref.try_lock() { s.vad_level = peak; }
                    if peak > VAD_THRESHOLD {
                        spk.store(true, Ordering::Relaxed);
                        *last.lock().unwrap() = std::time::Instant::now();
                        let mut b = buf.lock().unwrap();
                        for &s in data { b.push((s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)); }
                    } else if spk.load(Ordering::Relaxed) {
                        let mut b = buf.lock().unwrap();
                        for &s in data { b.push((s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)); }
                        if last.lock().unwrap().elapsed() >= SILENCE_TIMEOUT { done.store(true, Ordering::Relaxed); }
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
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(stream);
    let recorded = samples.lock().unwrap().clone();
    if recorded.is_empty() { return Err("No audio recorded.".into()); }

    let mono: Vec<f32> = if channels > 1 {
        recorded.chunks_exact(channels).map(|chunk| chunk.iter().sum::<f32>() / channels as f32).collect()
    } else { recorded };

    let resampled = if sample_rate != TARGET_SAMPLE_RATE { resample(&mono, sample_rate, TARGET_SAMPLE_RATE) } else { mono };
    Ok(resampled)
}

//-----------------------------------------------------------------------------
// TTS
//-----------------------------------------------------------------------------

fn compute_vu_levels(wav_path: &str) -> Vec<f32> {
    let mut levels = Vec::new();
    let mut reader = match hound::WavReader::open(wav_path) {
        Ok(r) => r, Err(_) => return levels,
    };
    let spec = reader.spec();
    let frame_samples = (spec.sample_rate as usize / 20).max(1) * spec.channels as usize;
    let mut chunk = Vec::with_capacity(frame_samples);
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let divisor = match spec.bits_per_sample {
                8 => i16::from(i8::MAX) as f32, 16 => i16::MAX as f32,
                24 => 8388607.0f32, 32 => 2147483647.0f32, _ => i16::MAX as f32,
            };
            for sample in reader.samples::<i16>() {
                if let Ok(s) = sample {
                    chunk.push(s as f32 / divisor);
                    if chunk.len() >= frame_samples {
                        levels.push(chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max));
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
                        levels.push(chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max));
                        chunk.clear();
                    }
                }
            }
        }
    }
    if !chunk.is_empty() {
        levels.push(chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max));
    }
    levels
}

fn speak_text(text: &str, cancel: &AtomicBool, state: Arc<Mutex<AppState>>) -> Result<(), Box<dyn std::error::Error>> {
    if text.trim().is_empty() { return Ok(()); }

    let (tx_stop, rx_stop) = mpsc::channel::<()>();
    let _listener = std::thread::spawn(move || {
        let mut buf = String::new();
        if io::stdin().read_line(&mut buf).is_ok() { let _ = tx_stop.send(()); }
    });

    let interrupted = || -> bool { rx_stop.try_recv().is_ok() || cancel.load(Ordering::Relaxed) };

    let cwd = std::env::current_dir()?;
    let piper_exe = cwd.join(PIPER_DIR).join("piper.exe");
    let piper_voice = cwd.join(PIPER_DIR).join("de_DE-kerstin-low.onnx");

    if piper_exe.exists() && piper_voice.exists() {
        let temp_wav = std::env::temp_dir().join("ollama_chat_tts.wav");
        let wav_path = temp_wav.to_string_lossy().to_string();

        let mut piper_proc = Command::new(&piper_exe)
            .args(["--model", &piper_voice.to_string_lossy(), "--output-file", &wav_path])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        if let Some(mut stdin) = piper_proc.stdin.take() { stdin.write_all(text.as_bytes())?; }
        let _ = piper_proc.wait();

        let vu_levels = compute_vu_levels(&wav_path);
        let total_frames = vu_levels.len();

        let mut play = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command",
                   &format!("(New-Object System.Media.SoundPlayer '{}').PlaySync()", wav_path)])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let vu_start = std::time::Instant::now();
        loop {
            if let Ok(Some(_)) = play.try_wait() { break; }
            if interrupted() { let _ = play.kill(); break; }

            let elapsed = vu_start.elapsed();
            if total_frames > 0 {
                let frame = (elapsed.as_secs_f64() * 20.0) as usize;
                let idx = frame.min(total_frames - 1);
                if let Ok(mut s) = state.try_lock() { s.vu_level = vu_levels[idx]; }
            }

            std::thread::sleep(Duration::from_millis(50));
        }

        if let Ok(mut s) = state.try_lock() { s.vu_level = 0.0; }
        let _ = std::fs::remove_file(&temp_wav);
    } else {
        let mut child = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command",
                   "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak([Console]::In.ReadToEnd())"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        if let Some(mut stdin) = child.stdin.take() { stdin.write_all(text.as_bytes())?; }
        loop {
            if let Ok(Some(_)) = child.try_wait() { break; }
            if interrupted() { let _ = child.kill(); break; }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    Ok(())
}

//-----------------------------------------------------------------------------
// Conversation thread
//-----------------------------------------------------------------------------

fn conversation_loop(state: Arc<Mutex<AppState>>, cmd_rx: mpsc::Receiver<UiCommand>) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let client = Client::new();

    let mut history: Vec<Message> = vec![Message {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    loop {
        // Check for exit (non-blocking)
        match cmd_rx.try_recv() {
            Ok(UiCommand::Exit) | Err(mpsc::TryRecvError::Disconnected) => break,
            _ => {}
        }

        // Listening phase
        {
            let mut s = state.lock().unwrap();
            s.phase = Phase::Listening;
            s.status_text = "Listening...".into();
            s.vad_level = 0.0;
            s.error = None;
        }

        // Cooldown to avoid TTS echo re-triggering VAD
        std::thread::sleep(Duration::from_millis(600));

        // Record audio
        let audio = match record_audio_inner(None, Arc::clone(&state)) {
            Ok(a) => a,
            Err(_) => continue,
        };

        // Transcribing phase
        {
            let mut s = state.lock().unwrap();
            s.phase = Phase::Transcribing;
            s.status_text = "Transcribing...".into();
            s.transcription_progress = 0;
        }

        let wav_path = std::env::temp_dir().join("ollama_chat_input.wav");
        let wav_str = wav_path.to_str().unwrap_or("ollama_chat_input.wav");
        if let Err(e) = save_wav(wav_str, &audio, TARGET_SAMPLE_RATE) {
            let mut s = state.lock().unwrap();
            s.phase = Phase::Listening;
            s.status_text = "Listening...".into();
            s.error = Some(format!("WAV error: {}", e));
            continue;
        }

        let input = match transcribe_via_whisper(wav_str, Arc::clone(&state)) {
            Ok(t) => {
                {
                    let mut s = state.lock().unwrap();
                    s.transcription_progress = 100;
                }
                t
            }
            Err(e) => {
                let mut s = state.lock().unwrap();
                s.phase = Phase::Listening;
                s.status_text = "Listening...".into();
                s.error = Some(format!("STT error: {}", e));
                continue;
            }
        };
        let _ = std::fs::remove_file(&wav_path);

        if !is_meaningful_speech(&input) {
            {
                let mut s = state.lock().unwrap();
                s.phase = Phase::Listening;
                s.status_text = "Listening...".into();
            }
            std::thread::sleep(Duration::from_millis(500));
            continue;
        }

        // Add user message
        {
            let mut s = state.lock().unwrap();
            s.messages.push(ChatEntry { role: "user".to_string(), content: input.clone() });
        }

        // Check if web search is needed based on user input
        let search_query = detect_search_intent(&input);

        history.push(Message { role: "user".to_string(), content: input });

        if let Some(ref query) = search_query {
            {
                let mut s = state.lock().unwrap();
                s.search_in_progress = true;
                s.search_show_until = None;
                s.search_query = query.clone();
                let disp = if query.len() > 40 { format!("{}...", &query[..40]) } else { query.clone() };
                s.status_text = format!("Web search: {}", disp);
            }

            let result = search_web(query, &client, &rt);

            {
                let mut s = state.lock().unwrap();
                s.search_in_progress = false;
                s.search_result = result.clone();
                s.search_show_until = Some(std::time::Instant::now() + std::time::Duration::from_secs(3));
            }

            // Inject search result as context message
            history.push(Message {
                role: "system".to_string(),
                content: format!("Web search result for '{}':\n{}", query, result),
            });
        }

        // Thinking phase
        {
            let mut s = state.lock().unwrap();
            s.phase = Phase::Thinking;
            s.status_text = "Thinking...".into();
        }

        // Ollama request (async via tokio runtime)
        let body = json!({ "model": "phi4", "stream": true, "messages": history.clone() });

        let response = match rt.block_on(async {
            client.post("http://localhost:11434/api/chat").json(&body).send().await
        }) {
            Ok(r) => r,
            Err(e) => {
                let mut s = state.lock().unwrap();
                s.phase = Phase::Listening;
                s.status_text = "Listening...".into();
                s.error = Some(format!("Ollama error: {}", e));
                continue;
            }
        };

        // Stream processing + streaming TTS
        let tts_cancel = Arc::new(AtomicBool::new(false));

        let (tts_tx, tts_rx) = mpsc::channel::<String>();
        let tts_cancel_tts = Arc::clone(&tts_cancel);
        let tts_state = Arc::clone(&state);
        let tts_handle = std::thread::spawn(move || {
            while let Ok(segment) = tts_rx.recv() {
                if tts_cancel_tts.load(Ordering::Relaxed) { break; }
                if let Err(e) = speak_text(&segment, &tts_cancel_tts, Arc::clone(&tts_state)) {
                    eprintln!("TTS error: {}", e);
                    break;
                }
            }
        });

        let mut full_response = String::new();
        let mut tts_buffer = String::new();

        {
            let mut s = state.lock().unwrap();
            s.messages.push(ChatEntry { role: "assistant".to_string(), content: String::new() });
            s.phase = Phase::Speaking;
            s.status_text = "Speaking... (Enter to interrupt)".into();
        }

        let mut stream = response.bytes_stream();
        while let Some(chunk) = rt.block_on(stream.next()) {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.trim().is_empty() { continue; }
                if let Ok(parsed) = serde_json::from_str::<OllamaChunk>(line) {
                    if let Some(msg) = parsed.message {
                        if let Some(content) = msg.content {
                            full_response.push_str(&content);

                            // Skip TTS for markdown code blocks (```...```)
                            let in_code = full_response.matches("```").count() % 2 == 1;
                            if !in_code && !content.contains("```") {
                                let cleaned = content.replace('`', "");
                                tts_buffer.push_str(&cleaned);

                                if tts_buffer.len() >= 80
                                    && (tts_buffer.ends_with('.') || tts_buffer.ends_with('!') || tts_buffer.ends_with('?'))
                                {
                                    let segment = std::mem::take(&mut tts_buffer);
                                    let _ = tts_tx.send(segment);
                                }
                            }

                            // Update visible text (always show everything including code)
                            {
                                let mut s = state.lock().unwrap();
                                if let Some(last) = s.messages.last_mut() {
                                    last.content = full_response.clone();
                                }
                            }
                        }
                    }
                    if parsed.done == Some(true) { break; }
                }
            }
        }

        if !tts_buffer.is_empty() {
            let _ = tts_tx.send(std::mem::take(&mut tts_buffer));
        }
        drop(tts_tx);

        let _ = tts_handle.join();

        {
            let mut s = state.lock().unwrap();
            if full_response.is_empty() {
                s.messages.pop();
            }
            s.phase = Phase::Listening;
            s.status_text = "Listening...".into();
            s.vu_level = 0.0;
        }

        if !full_response.is_empty() {
            history.push(Message { role: "assistant".to_string(), content: full_response });
        }

        // Clean up injected search context (don't keep it for future turns)
        if search_query.is_some() {
            // Remove the system message we added (it was second-to-last message in history before assistant response)
            // The last is the assistant response, the second-to-last is the search context
            // Actually, we pushed: user message, search system message, assistant response
            // After assistant push: [system, user, search-system, assistant] - we want to keep system and user, remove search-system
            // After popping search: [system, user, assistant]
            if history.len() >= 4 {
                // Remove the search system message at index history.len() - 2
                let idx = history.len() - 2;
                if history[idx].role == "system" && history[idx].content.starts_with("Web search result") {
                    history.remove(idx);
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// TUI rendering
//-----------------------------------------------------------------------------

fn ui(f: &mut ratatui::Frame, state: &AppState, scroll_offset: &mut usize) {
    let size = f.size();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(3), Constraint::Min(1), Constraint::Length(4)])
        .split(size);

    // Title
    let title = Paragraph::new("Ollama Chat  —  Speech-to-Text  —  phi4 + whisper.cpp + Piper TTS")
        .block(Block::default().borders(Borders::ALL).title(" AI Chat "));
    f.render_widget(title, chunks[0]);

    // Chat area
    let chat_rect = chunks[1];
    let inner = Rect { x: chat_rect.x + 1, y: chat_rect.y + 1, width: chat_rect.width.saturating_sub(2), height: chat_rect.height.saturating_sub(2) };
    let block = Block::default().borders(Borders::ALL).title(" Conversation ");
    f.render_widget(block, chat_rect);

// Pre-wrap long lines at inner.width to keep lines.len() == visual lines
let max_width = inner.width.saturating_sub(2) as usize;
let mut lines: Vec<Line> = Vec::new();
for entry in &state.messages {
    let prefix = if entry.role == "user" { "You: " } else { "AI: " };
    let mut first = true;
    for line_text in entry.content.lines() {
        if line_text.is_empty() {
            lines.push(Line::from(Span::raw("")));
            continue;
        }
        let text = if first { format!("{}{}", prefix, line_text) } else { line_text.to_string() };
        first = false;
        if max_width > 0 {
            let mut start = 0;
            let len = text.len();
            while start < len {
                let end = std::cmp::min(start + max_width, len);
                // Try to break at a space if not at the end
                let break_at = if end < len {
                    if let Some(space) = text[start..end].rfind(' ') {
                        start + space + 1
                    } else {
                        end
                    }
                } else {
                    end
                };
                lines.push(Line::from(Span::raw(text[start..break_at].to_string())));
                start = break_at;
            }
        } else {
            lines.push(Line::from(Span::raw(text)));
        }
    }
    lines.push(Line::from(Span::raw("")));
}
    if let Some(ref err) = state.error {
        lines.push(Line::from(Span::raw(format!("Error: {}", err))));
    }

    if state.phase == Phase::Transcribing || state.phase == Phase::Thinking || state.phase == Phase::Speaking {
        *scroll_offset = 0;
    }
    let max_scroll = lines.len().saturating_sub(inner.height as usize);
    *scroll_offset = (*scroll_offset).min(max_scroll);
    let scroll = max_scroll.saturating_sub(*scroll_offset);
    let chat = Paragraph::new(lines.clone())
        .scroll((scroll as u16, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(chat, inner);

    // Status bar
    let status_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(5), Constraint::Length(20)])
        .split(chunks[2]);
    let status_block = Block::default().borders(Borders::ALL).title(" Status ");
    f.render_widget(status_block, chunks[2]);

    // Phase + status text
    let phase_style = match state.phase {
        Phase::Listening => Style::default().fg(Color::Green),
        Phase::Transcribing => Style::default().fg(Color::Yellow),
        Phase::Thinking => Style::default().fg(Color::Magenta),
        Phase::Speaking => Style::default().fg(Color::Blue),
    };
    let phase_text = Paragraph::new(Line::from(Span::styled(&state.status_text, phase_style)));
    f.render_widget(phase_text, status_chunks[0]);

    // VU meter
    let vu_label = format!("VU: {:>3}%", (state.vu_level * 100.0) as u8);
    let vu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::NONE))
        .gauge_style(Style::default().fg(if state.vu_level > 0.7 { Color::Red } else if state.vu_level > 0.4 { Color::Yellow } else { Color::Green }))
        .percent((state.vu_level * 100.0) as u16)
        .label(vu_label);
    f.render_widget(vu_gauge, status_chunks[1]);

    // VAD meter
    let vad_label = format!("VAD: {:>3}%", (state.vad_level * 100.0) as u8);
    let vad_gauge = Gauge::default()
        .block(Block::default().borders(Borders::NONE))
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent((state.vad_level * 100.0).min(100.0) as u16)
        .label(vad_label);
    f.render_widget(vad_gauge, status_chunks[2]);

    // Phase text shows Transcribing status in status bar

    // Search dialog overlay
    let show_search = state.search_in_progress
        || state.search_show_until.map(|t| t > std::time::Instant::now()).unwrap_or(false);
    if show_search {
        let area = centered_rect(70, 50, size);
        let status = if state.search_in_progress { " Web Search " } else { " Search Complete " };
        let display_text = if state.search_in_progress {
            format!("🔍 {}", state.search_query)
        } else {
            format!("🔍 {}\n\n{}", state.search_query, state.search_result)
        };
        let dialog = Paragraph::new(display_text)
            .block(Block::default().borders(Borders::ALL).title(status))
            .wrap(Wrap { trim: false });
        f.render_widget(Clear, area);
        f.render_widget(dialog, area);
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length((r.height * (100 - percent_y)) / 200),
            Constraint::Length((r.height * percent_y) / 100),
            Constraint::Length((r.height * (100 - percent_y)) / 200),
        ])
        .split(r)[1];
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length((r.width * (100 - percent_x)) / 200),
            Constraint::Length((r.width * percent_x) / 100),
            Constraint::Length((r.width * (100 - percent_x)) / 200),
        ])
        .split(popup)[1]
}

//-----------------------------------------------------------------------------
// TUI main loop
//-----------------------------------------------------------------------------

fn run_tui(state: Arc<Mutex<AppState>>, cmd_tx: mpsc::Sender<UiCommand>) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut scroll_offset = 0usize;

    let res = loop {
        {
            let state_guard = state.lock().unwrap();
            terminal.draw(|f| ui(f, &state_guard, &mut scroll_offset))?;
        }

        // Handle keyboard
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => {
                        let _ = cmd_tx.send(UiCommand::Exit);
                        break Ok(());
                    }
                    KeyCode::Up => scroll_offset = scroll_offset.saturating_add(1),
                    KeyCode::Down => scroll_offset = scroll_offset.saturating_sub(1),
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Ollama Chat 2 — Speech-to-Text Assistant".green().bold());

    // Windows 11 check
    #[cfg(target_os = "windows")]
    {
        if is_windows_11() {
            println!("✅ Windows 11 detected.");
        } else {
            println!("{}", "⚠️  Windows 11 recommended.".yellow());
        }
    }

    // Ollama
    if !ollama_is_installed() {
        println!("{} Installing Ollama...", "Ollama missing.".red());
        if !install_ollama() {
            eprintln!("{} Failed to install Ollama. Install manually: https://ollama.com", "Error:".red());
            std::process::exit(1);
        }
    } else {
        println!("✅ Ollama is already installed.");
    }

    // phi4
    if !phi4_is_installed() {
        println!("{} Downloading phi4 model (~3 GB)...", "Model missing.".red());
        if !install_phi4() {
            eprintln!("{}", "Failed to pull phi4 model.".red());
            std::process::exit(1);
        }
    } else {
        println!("✅ Model 'phi4' is already installed.");
    }

    let client = Client::new();
    download_whisper_model(&client).await?;
    download_whisper_binary(&client).await?;
    download_piper_binary(&client).await?;
    download_piper_voice(&client).await?;
    println!();

    // Start TUI
    let state = Arc::new(Mutex::new(AppState::new()));
    let (cmd_tx, cmd_rx) = mpsc::channel::<UiCommand>();

    // Spawn conversation thread
    let conv_state = Arc::clone(&state);
    std::thread::spawn(move || {
        conversation_loop(conv_state, cmd_rx);
    });

    // Run TUI
    if let Err(e) = run_tui(state, cmd_tx) {
        eprintln!("{} {}", "TUI error:".red(), e);
    }

    println!("\nGoodbye! 👋");
    Ok(())
}
