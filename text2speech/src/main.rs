use std::{thread, time::Duration};
use tts::Tts;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tts = Tts::default()?;

    let voices = tts.voices()?;

    tts.set_voice(&voices[2])?;

    tts.speak("Hallo, wie kann ich dir helfen?", false)?;

    while tts.is_speaking()? {
        thread::sleep(Duration::from_millis(100));
    }

    Ok(())
}