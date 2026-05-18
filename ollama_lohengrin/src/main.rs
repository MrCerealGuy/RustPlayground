use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::io::{stdout, Write};
use std::process::Command;

//-----------------------------------------------------------------------------

#[derive(Clone)]
struct Agent {
    name: String,
    system_prompt: String,
    memory: String,
}

//-----------------------------------------------------------------------------

impl Agent {
    fn new(name: &str, system_prompt: &str) -> Self {
        Self {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            memory: String::new(),
        }
    }

    fn build_input(&self, context: &str) -> String {
        format!(
            "KONTEXT:\n{}\n\nGEDÄCHTNIS:\n{}\n",
            context, self.memory
        )
    }

    fn update(&mut self, response: &str) {
        self.memory.push_str("\n");
        self.memory.push_str(response);
    }
}

//-----------------------------------------------------------------------------

#[derive(Deserialize)]
struct Chunk {
    message: Option<Message>,
}

//-----------------------------------------------------------------------------

#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}

//-----------------------------------------------------------------------------

async fn run_agent(
    client: &Client,
    model: &str,
    agent: &Agent,
    input: &str,
) -> Result<String> {
    let body = json!({
        "model": model,
        "stream": true,
        "messages": [
            { "role": "system", "content": agent.system_prompt },
            { "role": "user", "content": input }
        ]
    });

    let res = client
        .post("http://localhost:11434/api/chat")
        .json(&body)
        .send()
        .await?;

    let mut stream = res.bytes_stream();

    let mut full = String::new();

    print!("\n[{}]\n", agent.name);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if let Ok(parsed) = serde_json::from_str::<Chunk>(line) {
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

//-----------------------------------------------------------------------------

struct Orchestrator {
    teacher: Agent,
    student: Agent,
    critic: Agent,
}

//-----------------------------------------------------------------------------

impl Orchestrator {
    fn new() -> Self {
        Self {
            teacher: Agent::new(
                "Lehrer",
                "Du bist ein Rust-Lehrer. Erkläre klar und strukturiert.",
            ),
            student: Agent::new(
                "Schüler",
                "Du bist ein C-Programmierer und lernst Rust.",
            ),
            critic: Agent::new(
                "Kritiker",
                "Du bewertest Antworten auf Klarheit und Korrektheit.",
            ),
        }
    }
}

//-----------------------------------------------------------------------------

impl Orchestrator {
    async fn run(&mut self, client: &Client, model: &str) -> Result<()> {
        let mut context =
            "Thema: Rust Grundlagen für C Entwickler".to_string();

        for round in 1..=3 {
            println!("\n================ RUNDE {round} ================\n");

            // ───────── TEACHER ─────────
            let teacher_input = self.teacher.build_input(&context);

            let teacher_out = run_agent(
                client,
                model,
                &self.teacher,
                &teacher_input,
            )
            .await?;

            self.teacher.update(&teacher_out);

            // ───────── STUDENT ─────────
            let student_input = self.student.build_input(&teacher_out);

            let student_out = run_agent(
                client,
                model,
                &self.student,
                &student_input,
            )
            .await?;

            self.student.update(&student_out);

            // ───────── CRITIC (NEU!) ─────────
            let critic_input = format!(
                "Lehrer:\n{}\n\nSchüler:\n{}",
                teacher_out, student_out
            );

            let critic_out = run_agent(
                client,
                model,
                &self.critic,
                &critic_input,
            )
            .await?;

            self.critic.update(&critic_out);

            // ───────── CONTEXT UPDATE ─────────
            context = format!(
                "LEKTION:\n{}\n\nFEEDBACK:\n{}",
                teacher_out, critic_out
            );
        }

        Ok(())
    }
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
async fn main() -> Result<()> {
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

    let client = reqwest::Client::new();
    let mut orch = Orchestrator::new();

    orch.run(&client, "phi4").await?;

    Ok(())
}