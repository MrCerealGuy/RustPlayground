use std::io::{self, Write};

use ollama_rs::{
    generation::chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
    Ollama,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::default();

    let mut history = vec![
        ChatMessage::new(
            MessageRole::System,
            "You are a Rust programming tutor. Be concise and always show code examples."
                .into(),
        ),
    ];

    println!("Interactive Ollama Chat");
    println!("Type 'exit' to quit.\n");

    loop {
        // User input
        print!("You: ");
        io::stdout().flush()?; // important for prompt display

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        // Exit condition
        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Add user message to history
        history.push(ChatMessage::new(
            MessageRole::User,
            input.to_string(),
        ));

        // Send request
        let res = ollama
            .send_chat_messages(
                ChatMessageRequest::new(
                    "phi4".into(),
                    history.clone(),
                )
            )
            .await?;

        let reply = res.message.content.clone();

        // Print assistant reply
        println!("\nAssistant:\n{}\n", reply);

        // Save assistant reply in history
        history.push(ChatMessage::new(
            MessageRole::Assistant,
            reply,
        ));
    }

    println!("Goodbye!");

    Ok(())
}