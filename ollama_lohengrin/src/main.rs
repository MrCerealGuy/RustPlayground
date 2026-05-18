use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::io::{stdout, Write};

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: Option<Message>,
    done: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: Option<String>,
}

pub async fn stream_agent(
    client: &Client,
    model: &str,
    system_prompt: &str,
    input: &str,
    label: &str,
) -> Result<String> {
    let body = json!({
        "model": model,
        "stream": true,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": input }
        ]
    });

    let response = client
        .post("http://localhost:11434/api/chat")
        .json(&body)
        .send()
        .await?;

    let mut stream = response.bytes_stream();

    let mut full = String::new();

    println!("\n[{label}]\n");

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        // 🔥 WICHTIG: Ollama sendet JSON pro Zeile
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(parsed) = serde_json::from_str::<OllamaStreamChunk>(line) {
                if let Some(msg) = parsed.message {
                    if let Some(content) = msg.content {
                        print!("{content}");
                        stdout().flush().unwrap();

                        full.push_str(&content);
                    }
                }
            }
        }
    }

    println!();

    Ok(full)
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = reqwest::Client::new();

    let model = "phi4";

    let mut context =
        "Diskussion: Einführung in die Programmiersprache Rust anhand praxisnaher Beispiele.".to_string();

    for round in 1..=3 {
        println!("\n================ Runde {round} ================\n");

        let a = stream_agent(
            &client,
            model,
            "Du bist Lehrer: Ein Experte auf dem Gebiet der Rust Programmierung.",
            &context,
            "Lehrer",
        ).await?;

        let b = stream_agent(
            &client,
            model,
            "Du bist Schüler: Du hast Erfahrung in der Programmiersprache C.",
            &a,
            "Schüler",
        ).await?;

        context = format!("A:\n{a}\n\nB:\n{b}");
    }

    Ok(())
}