use std::io::{self, Write};
use std::process::Command;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

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
    let client = Client::new();

    let mut history: Vec<Message> = vec![
        Message {
            role: "system".to_string(),
            content: "You are a helpful Rust programming tutor. Be concise and show code examples.".to_string(),
        }
    ];

    // Check for ollama
    if ollama_is_installed() {
        println!("Ollama is already installed.");
    } else {
        println!("Ollama not found! Installing Ollama!");

        if install_ollama() {
            println!("Installation successful!");
        } else {
            eprintln!("Installation failed!");
            std::process::exit(1);
        }        
    }

    // Check for AI model 'phi4'
    if phi4_is_installed() {
        println!("Model 'phi4' is already installed.");
    } else {
        println!("Model 'phi4' is missing. Installing model!");

        if install_phi4() {
            println!("Model 'phi4' successfully installed.");
        } else {
            eprintln!("Installation failed!");
            std::process::exit(1);
        }
    }

    println!("Ollama Streaming Chat (Rust)");
    println!("Type 'exit' to quit.\n");

    loop {
        // USER INPUT
        print!("You: ");
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

        // REQUEST
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

        println!("\nAssistant:\n");

        let mut full_response = String::new();

        // STREAM PROCESSING (IMPORTANT PART)
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

        // SAVE ASSISTANT MESSAGE
        history.push(Message {
            role: "assistant".to_string(),
            content: full_response,
        });
    }

    println!("Goodbye!");

    Ok(())
}