use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use whisper_rs::{
    FullParams,
    SamplingStrategy,
    WhisperContext,
    WhisperContextParameters,
};

fn main() -> Result<()> {
    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));

    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .expect("Kein Mikrofon gefunden");

    println!("Mikrofon: {}", device.name()?);

    let config = device.default_input_config()?;

    println!(
        "Samplerate: {} Hz",
        config.sample_rate().0
    );

    let buffer_clone = audio_buffer.clone();

    let err_fn = |err| {
        eprintln!("Audiostream-Fehler: {err}");
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[f32], _| {
                    let mut buf =
                        buffer_clone.lock().unwrap();

                    buf.extend_from_slice(data);

                    // Maximal 30 Sekunden behalten
                    let max_samples = 48000 * 30;

                    if buf.len() > max_samples {
                        let remove =
                            buf.len() - max_samples;

                        buf.drain(..remove);
                    }
                },
                err_fn,
                None,
            )?
        }

        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[i16], _| {
                    let mut buf =
                        buffer_clone.lock().unwrap();

                    for s in data {
                        buf.push(
                            *s as f32
                                / i16::MAX as f32,
                        );
                    }
                },
                err_fn,
                None,
            )?
        }

        cpal::SampleFormat::U16 => {
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[u16], _| {
                    let mut buf =
                        buffer_clone.lock().unwrap();

                    for s in data {
                        let sample =
                            (*s as f32
                                / u16::MAX as f32)
                                * 2.0
                                - 1.0;

                        buf.push(sample);
                    }
                },
                err_fn,
                None,
            )?
        }

        _ => panic!("Nicht unterstütztes Audioformat"),
    };

    stream.play()?;

    println!("Whisper wird geladen ...");

    let ctx = WhisperContext::new_with_params(
        "models/ggml-base.bin",
        WhisperContextParameters::default(),
    )?;

    let mut state = ctx.create_state()?;

    println!("Bereit.");
    println!("Sprich ins Mikrofon ...");

    let mut last_output = String::new();

    loop {
        thread::sleep(Duration::from_secs(2));

        let audio = {
            let buf = audio_buffer.lock().unwrap();

            if buf.is_empty() {
                continue;
            }

            buf.clone()
        };

        if audio.len() < 16000 {
            continue;
        }

        let mut params =
            FullParams::new(
                SamplingStrategy::Greedy {
                    best_of: 1,
                },
            );

        params.set_language(Some("de"));
        params.set_n_threads(4);
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        if let Err(e) =
            state.full(params, &audio)
        {
            eprintln!("Whisper Fehler: {e}");
            continue;
        }

        let segments =
            state.full_n_segments()?;

        let mut text = String::new();

        for i in 0..segments {
            text.push_str(
                &state.full_get_segment_text(i)?,
            );
        }

        let text = text.trim();

        if !text.is_empty()
            && text != last_output
        {
            println!("\n{text}\n");

            last_output = text.to_string();
        }
    }
}