# ollama_chat2

Ein interaktiver KI-Chat-Assistent mit Sprachsteuerung (Speech-to-Text und Text-to-Speech), basierend auf **Ollama** + **phi4**, **whisper.cpp** (STT) und **Piper TTS** (Sprachausgabe).

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
| **Whisper STT Modell** (`ggml-base.bin`) | ~145 MB | huggingface.co (ggerganov/whisper.cpp) |
| **Whisper Binary** (`whisper.exe`) | ~4 MB | GitHub (whisper.cpp v1.8.6) |
| **Piper TTS Binary** (`piper.exe` + DLLs) | ~22 MB | GitHub (rhasspy/piper) |
| **Piper Deutsche Stimme** (`de_DE-thorsten-medium.onnx`) | ~63 MB | huggingface.co (rhasspy/piper-voices) |

## Features

- **Voice Activity Detection (VAD)** – Sprache wird automatisch erkannt, keine Taste nötig
- **Speech-to-Text** – whisper.cpp mit deutschem Sprachmodell (`-l de`)
- **Text-to-Speech** – Piper TTS mit deutscher Neural-Stimme (Thorsten, medium)
- **Ins-Wort-Fallen** – Sprich einfach, um die KI zu unterbrechen
- **Enter-Taste** – Backup zum Stoppen der Sprachausgabe
- **Exit** – Sage "Exit" oder tippe es ein

## Konfiguration

Die Rolle des Assistenten wird beim Start per Spracheingabe oder Tastatur festgelegt.

## Abhängigkeiten (Cargo.toml)

- `cpal` – Mikrofonzugriff
- `hound` – WAV-Export
- `reqwest` – Download von Modellen
- `serde` / `serde_json` – JSON-Kommunikation mit Ollama
- `futures-util` – Streaming der Ollama-Antwort
- `owo-colors` – Farbige Konsolenausgabe
- `winreg` – Windows 11-Erkennung
- `tokio` – Async-Runtime

## Projektstruktur

```
ollama_chat2/
├── src/
│   └── main.rs          # Hauptprogramm
├── Cargo.toml
├── README.md
├── ggml-base.bin        # Whisper Modell (automatisch)
├── whisper.exe          # Whisper Binary (automatisch)
├── whisper.dll          # Whisper DLLs (automatisch)
├── piper/               # Piper TTS (automatisch)
│   ├── piper.exe
│   ├── de_DE-thorsten-medium.onnx
│   └── ...
└── target/              # Build-Output
```
