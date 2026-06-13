# ollama_chat2

Ein interaktiver KI-Chat-Assistent mit Sprachsteuerung (Speech-to-Text und Text-to-Speech), basierend auf **Ollama** + **phi4**, **whisper.cpp** (STT) und **Piper TTS** (Sprachausgabe). Modernes TUI mit Ratatui.

## Voraussetzungen

- **Windows 11** (wird beim Start geprüft)
- **Ollama** (wird automatisch installiert, falls fehlend)
- **Rust** (stable) – [rustup.rs](https://rustup.rs)

## Installation

### 1. Rust installieren

```powershell
winget install Rustlang.Rustup
# oder: https://rustup.rs
```

### 2. Repository klonen und bauen

```powershell
git clone <url>
cd ollama_chat2
cargo build
```

### 3. KI-Modell pullen (falls nicht automatisch geschehen)

```powershell
ollama pull phi4
```

## Erster Start

```powershell
cargo run
```

Beim ersten Start werden automatisch heruntergeladen:

| Komponente | Größe | Quelle |
|---|---|---|
| **Whisper STT Modell** (`ggml-large-v3.bin`) | ~1,5 GB | huggingface.co (ggerganov/whisper.cpp) |
| **Whisper Binary** (`whisper.exe` + DLLs) | ~4 MB | GitHub (whisper.cpp v1.8.6) |
| **Piper TTS Binary** (`piper.exe` + DLLs) | ~22 MB | GitHub (rhasspy/piper) |
| **Piper Deutsche Stimme** (`de_DE-kerstin-low.onnx`) | ~63 MB | huggingface.co (rhasspy/piper-voices) |

## Features

- **Always-on VAD** – Sprache wird automatisch erkannt, keine Taste nötig
- **Speech-to-Text** – whisper.cpp mit deutschem Large-v3-Modell (`-l de -pp`)
- **Text-to-Speech** – Piper TTS mit deutscher Stimme (Kerstin, weiblich)
- **VAD Barge-in** – Sprich einfach, um die KI während der Sprachausgabe zu unterbrechen
- **Enter-Taste** – Backup zum Stoppen der Sprachausgabe
- **Web Search** – DuckDuckGo Lite + Wikipedia, ausgelöst per Schlüsselwörtern
- **Wetterabfrage** – Open-Meteo API (kostenlos, kein API-Key)
- **URL-Fetching** – Webseiten-Inhalte abrufen und zusammenfassen
- **Gesprächs-Gedächtnis** – Konversation wird in `conversation_history.json` gespeichert (nach jedem Durchlauf)
- **Ratatui TUI** – Drei Fenster (Titel, Chat-Verlauf, Statusleiste)
- **Exit** – `Esc`, `q` oder `Ctrl+C`

## Bedienung

| Taste | Aktion |
|---|---|
| `Esc` / `q` / `Ctrl+C` | Programm beenden |
| `Enter` / `Space` | TTS-Stopp (Backup) |
| `↑` / `↓` | Im Chat-Verlauf scrollen |

Während der Sprachausgabe der KI: Einfach selbst zu sprechen beginnen → VAD Barge-in unterbricht die Ausgabe automatisch.

## Konfiguration

Das System-Prompt ist aktuell leer (`SYSTEM_PROMPT = ""` in `main.rs`). Zum Anpassen die Konstante in `main.rs` ändern.

## Abhängigkeiten (Cargo.toml)

- `cpal` – Mikrofonzugriff
- `hound` – WAV-Export
- `reqwest` – HTTP-Client (Download + Web Search + URL-Fetching)
- `serde` / `serde_json` – JSON-Kommunikation mit Ollama
- `futures-util` – Streaming der Ollama-Antwort
- `owo-colors` – Farbige Konsolenausgabe (Pre-TUI)
- `winreg` – Windows 11-Erkennung
- `tokio` – Async-Runtime
- `ratatui` – TUI-Framework
- `crossterm` – Terminal-Steuerung (raw mode, Events)
- `urlencoding` – URL-Encoding für Web Search
- `regex` – HTML-Parsing für DuckDuckGo Lite

## Projektstruktur

```
ollama_chat2/
├── src/
│   └── main.rs                # Hauptprogramm (gesamte Logik)
├── Cargo.toml
├── README.md
├── ggml-large-v3.bin          # Whisper Modell (automatisch)
├── whisper.exe                # Whisper Binary (automatisch)
├── whisper.dll / ggml*.dll    # Whisper DLLs (automatisch)
├── piper/                     # Piper TTS (automatisch)
│   ├── piper.exe
│   ├── de_DE-kerstin-low.onnx
│   └── espeak-ng-data/
├── conversation_history.json  # Gesprächsverlauf (automatisch)
└── target/                    # Build-Output
```
