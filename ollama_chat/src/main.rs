// -----------------------------------------------------------------------------
// ollama_chat - Your personal AI chat assistant.
//
// by Andreas Zahnleiter <a.zahnleiter@gmx.de>
// -----------------------------------------------------------------------------
// 2026-05-17 - az - created
// 2026-05-18 - az - added ollama installation process
// 2026-05-19 - az - added colored text and unicode symbols
// 2026-05-20 - az - check for Windows 11 
// 2026-05-20 - az - user-defined role AI assistant
// -----------------------------------------------------------------------------

use std::io::{self, Write};
use std::process::Command;
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

    println!("\nOllama Streaming Chat");
    println!("Type 'exit' to quit.\n");

    // Role of AI assistant
    print!("Please describe the role of your AI assistant: ");
    io::stdout().flush()?;

    let mut initialcontent = String::new();
    io::stdin().read_line(&mut initialcontent)?;
    let initialcontent = initialcontent.trim();

    println!("\n");

    let mut history: Vec<Message> = vec![
        Message {
            role: "system".to_string(),
            content: initialcontent.to_string(),
        }
    ];

    let client = Client::new();

    loop {
        // User input
        print!("{}", "You: ".green());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.is_empty() {
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

        // Sava assistant message
        history.push(Message {
            role: "assistant".to_string(),
            content: full_response,
        });
    }

    println!("\nGoodbye! 👋");

    Ok(())
}

//-----------------------------------------------------------------------------