use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::io::{stdout, Write};

#[derive(Clone)]
struct Agent {
    name: String,
    system_prompt: String,
    memory: String,
}

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
            "AKTUELLER KONTEXT:\n{}\n\nDEIN GEDÄCHTNIS:\n{}\n",
            context, self.memory
        )
    }

    fn update_memory(&mut self, response: &str) {
        self.memory.push_str("\n");
        self.memory.push_str(response);
    }
}

#[derive(Debug, Deserialize)]
struct Chunk {
    message: Option<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: Option<String>,
}

async fn stream_agent_response(
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

    let response = client
        .post("http://localhost:11434/api/chat")
        .json(&body)
        .send()
        .await?;

    let mut stream = response.bytes_stream();

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

#[tokio::main]
async fn main() -> Result<()> {
    let client = reqwest::Client::new();
    let model = "phi4";

    let mut teacher = Agent::new(
        "Lehrer",
        r#"
Du bist ein erfahrener Rust-Lehrer.
Du erklärst klar, strukturiert und korrekt.
Du wiederholst dich nicht.
Du baust auf vorherigen Antworten auf.
"#,
    );

    let mut student = Agent::new(
        "Schüler",
        r#"
Du bist ein Schüler mit C-Erfahrung.
Du stellst Fragen.
Du verstehst nicht alles sofort.
Du darfst Fehler machen.
Du wiederholst dich nicht.
"#,
    );

    let mut context =
        "Thema: Einführung in Rust für C-Programmierer".to_string();

    for round in 1..=3 {
        println!("\n================ RUNDE {round} ================\n");

        // ───── TEACHER ─────
        let teacher_input = teacher.build_input(&context);

        let teacher_response = stream_agent_response(
            &client,
            model,
            &teacher,
            &teacher_input,
        )
        .await?;

        teacher.update_memory(&teacher_response);

        // ───── STUDENT ─────
        let student_input = student.build_input(&teacher_response);

        let student_response = stream_agent_response(
            &client,
            model,
            &student,
            &student_input,
        )
        .await?;

        student.update_memory(&student_response);

        // ───── CONTEXT UPDATE ─────
        context = format!(
            "Lehrer:\n{}\n\nSchüler:\n{}",
            teacher_response, student_response
        );
    }

    Ok(())
}