use candle_core::Tensor;

use hound::{Error as HoundError, WavReader};
#[cfg(not(target_arch = "wasm32"))]
use hound::{WavSpec, WavWriter};

use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
pub fn read_wav<P: AsRef<Path>>(path: P) -> anyhow::Result<(Tensor, u32)> {
    let reader = WavReader::open(path)?;
    read_wav_internal(reader)
}

/// Read any supported audio file to a mono `[1, T]` tensor, mirroring
/// upstream's `audio_read`: WAV always works via hound; other formats
/// (mp3, flac, ogg, m4a) need the optional `audio-formats` feature
/// (symphonia), the Rust counterpart of upstream's optional soundfile.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_audio<P: AsRef<Path>>(path: P) -> anyhow::Result<(Tensor, u32)> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "wav" || ext == "wave" {
        return read_wav(path);
    }

    #[cfg(feature = "audio-formats")]
    {
        read_audio_symphonia(path)
    }
    #[cfg(not(feature = "audio-formats"))]
    {
        anyhow::bail!(
            "reading .{ext} needs the `audio-formats` feature \
             (rebuild with --features audio-formats), or provide a WAV file"
        )
    }
}

/// Decode a non-WAV audio file with symphonia to mono f32.
#[cfg(all(not(target_arch = "wasm32"), feature = "audio-formats"))]
fn read_audio_symphonia(path: &Path) -> anyhow::Result<(Tensor, u32)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow::anyhow!("no audio track in {path:?}"))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow::anyhow!("no audio codec parameters in {path:?}"))?
        .clone();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())?;

    let mut sample_rate = params.sample_rate.unwrap_or(0);
    let mut mono: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            // End of stream.
            Ok(None) => break,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Recoverable per symphonia docs: skip the malformed packet.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = decoded.spec();
        sample_rate = spec.rate();
        let channels = spec.channels().count().max(1);
        decoded.copy_to_vec_interleaved(&mut interleaved);
        for frame in interleaved.chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }

    if mono.is_empty() || sample_rate == 0 {
        anyhow::bail!("no audio decoded from {path:?}");
    }
    let n = mono.len();
    let tensor = Tensor::from_vec(mono, (1, n), &candle_core::Device::Cpu)?;
    Ok((tensor, sample_rate))
}

pub fn read_wav_from_bytes(bytes: &[u8]) -> anyhow::Result<(Tensor, u32)> {
    let reader = WavReader::new(std::io::Cursor::new(bytes))?;
    read_wav_internal(reader)
}

fn read_wav_internal<R: std::io::Read + std::io::Seek>(
    mut reader: WavReader<R>,
) -> anyhow::Result<(Tensor, u32)> {
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            let mut samples = Vec::new();
            for s in reader.samples::<i32>() {
                match s {
                    Ok(v) => samples.push(v as f32 / max_val),
                    Err(e) => {
                        // If we hit an unexpected EOF but have read valid samples, we accept it.
                        if let HoundError::IoError(ref io_err) = e {
                            // Check for UnexpectedEof OR "Failed to read enough bytes" (which is Other in standard hound)
                            let is_unexpected_eof = io_err.kind() == io::ErrorKind::UnexpectedEof;
                            // Check string representation for the specific hound error message
                            let is_truncated_msg = io_err.kind() == io::ErrorKind::Other
                                && io_err.to_string().contains("enough bytes");

                            if (is_unexpected_eof || is_truncated_msg) && !samples.is_empty() {
                                break;
                            }
                        }
                        return Err(anyhow::Error::from(e));
                    }
                }
            }
            samples
        }
        hound::SampleFormat::Float => {
            let mut samples = Vec::new();
            for s in reader.samples::<f32>() {
                match s {
                    Ok(v) => samples.push(v),
                    Err(e) => {
                        if let HoundError::IoError(ref io_err) = e {
                            let is_unexpected_eof = io_err.kind() == io::ErrorKind::UnexpectedEof;
                            let is_truncated_msg = io_err.kind() == io::ErrorKind::Other
                                && io_err.to_string().contains("enough bytes");

                            if (is_unexpected_eof || is_truncated_msg) && !samples.is_empty() {
                                break;
                            }
                        }
                        return Err(anyhow::Error::from(e));
                    }
                }
            }
            samples
        }
    };

    let device = if cfg!(target_arch = "wasm32") {
        &candle_core::Device::Cpu
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            &candle_core::Device::Cpu
        }
        #[cfg(target_arch = "wasm32")]
        {
            &candle_core::Device::Cpu
        }
    };

    let tensor = if channels > 1 {
        // Downmix interleaved multichannel audio to mono, like upstream's
        // audio_read (the models are mono; a stereo prompt should not end up
        // as two "channels" of conditioning).
        let num_samples = samples.len() / channels;
        let mut mono = vec![0.0f32; num_samples];
        for (i, sample) in mono.iter_mut().enumerate() {
            let frame = &samples[i * channels..(i + 1) * channels];
            *sample = frame.iter().sum::<f32>() / channels as f32;
        }
        Tensor::from_vec(mono, (1, num_samples), device)?
    } else {
        let n = samples.len();
        Tensor::from_vec(samples, (1, n), device)?
    };

    Ok((tensor, sample_rate))
}

pub fn pcm_i16_le_bytes(audio: &Tensor) -> anyhow::Result<Vec<u8>> {
    let shape = audio.dims();
    if shape.len() != 2 {
        anyhow::bail!(
            "Expected audio tensor with shape [channels, samples], got {:?}",
            shape
        );
    }

    let data = audio.to_vec2::<f32>()?;
    let channel_slices: Vec<&[f32]> = data.iter().map(|channel| channel.as_slice()).collect();
    Ok(pcm_i16_le_bytes_from_slices(&channel_slices))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn pcm_i16_le_bytes_mono(samples: &[f32]) -> Vec<u8> {
    pcm_i16_le_bytes_from_slices(&[samples])
}

fn pcm_i16_le_bytes_from_slices(channels: &[&[f32]]) -> Vec<u8> {
    if channels.is_empty() {
        return Vec::new();
    }

    let num_samples = channels[0].len();
    let mut out = Vec::with_capacity(num_samples * channels.len() * 2);

    for i in 0..num_samples {
        for channel in channels {
            let val = channel[i].clamp(-1.0, 1.0);
            let val = (val * 32767.0) as i16;
            out.extend_from_slice(&val.to_le_bytes());
        }
    }

    out
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_wav<P: AsRef<Path>>(path: P, audio: &Tensor, sample_rate: u32) -> anyhow::Result<()> {
    let mut writer = std::fs::File::create(path)?;
    write_wav_to_writer(&mut writer, audio, sample_rate)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_wav_to_writer<W: std::io::Write + std::io::Seek>(
    writer: W,
    audio: &Tensor,
    sample_rate: u32,
) -> anyhow::Result<()> {
    let shape = audio.dims();
    if shape.len() != 2 {
        anyhow::bail!(
            "Expected audio tensor with shape [channels, samples], got {:?}",
            shape
        );
    }
    let channels = shape[0] as u16;
    let _num_samples = shape[1];

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut wav_writer = WavWriter::new(writer, spec)?;
    let pcm_bytes = pcm_i16_le_bytes(audio)?;
    for chunk in pcm_bytes.as_chunks::<2>().0 {
        wav_writer.write_sample(i16::from_le_bytes(*chunk))?;
    }
    wav_writer.finalize()?;
    Ok(())
}

pub fn normalize_peak(audio: &Tensor) -> anyhow::Result<Tensor> {
    let max_abs = audio.abs()?.max_all()?.to_scalar::<f32>()?;
    if max_abs > 0.0 {
        Ok(audio.affine(1.0 / max_abs as f64, 0.0)?)
    } else {
        Ok(audio.clone())
    }
}

/// End a voice prompt `[..., T]` on exactly 80 ms of silence (upstream
/// `end_on_pause`, #334).
///
/// Training prompts end inside the pause between two words. A prompt that
/// stops on speech makes the model continue that utterance (a burst at the
/// start of every chunk, or a wrong first word), and one that ends on a long
/// silence delays the onset. The trailing silence (20 ms frames more than
/// 35 dB below the loudest one) is cut, the last 20 ms of what remains is
/// faded out, and 80 ms of zeros is appended.
pub fn end_on_pause(wav: &Tensor, sample_rate: usize) -> anyhow::Result<Tensor> {
    const PAUSE_SEC: f64 = 0.08;
    const FADE_SEC: f64 = 0.02;
    const FLOOR_DB: f32 = 35.0;

    let dims = wav.dims().to_vec();
    let Some((&t, lead)) = dims.split_last() else {
        return Ok(wav.clone());
    };
    let frame = ((0.02 * sample_rate as f64) as usize).max(1);
    let n = t / frame;
    if n == 0 {
        return Ok(wav.clone());
    }
    let rows: usize = lead.iter().product();

    let data = wav
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // RMS of each 20 ms frame over every row (batch and channels), in dB.
    let mut db = Vec::with_capacity(n);
    for f in 0..n {
        let mut sum = 0.0f64;
        for r in 0..rows {
            let start = r * t + f * frame;
            sum += data[start..start + frame]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>();
        }
        let rms = (sum / (rows * frame) as f64).sqrt() as f32;
        db.push(20.0 * (rms + 1e-12).log10());
    }
    let loudest = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let last_loud = db
        .iter()
        .rposition(|&d| d > loudest - FLOOR_DB)
        .unwrap_or(n - 1);
    let end = (last_loud + 1) * frame;

    let fade = ((FADE_SEC * sample_rate as f64) as usize).min(end);
    let pause = (PAUSE_SEC * sample_rate as f64) as usize;
    let new_t = end + pause;
    let mut out = Vec::with_capacity(rows * new_t);
    for r in 0..rows {
        let row = &data[r * t..r * t + end];
        out.extend_from_slice(row);
        // torch.linspace(1, 0, fade) over the last `fade` kept samples.
        let base = out.len() - fade;
        for i in 0..fade {
            let gain = if fade == 1 {
                1.0
            } else {
                1.0 - i as f32 / (fade - 1) as f32
            };
            out[base + i] *= gain;
        }
        out.resize(out.len() + pause, 0.0);
    }

    let mut new_dims = lead.to_vec();
    new_dims.push(new_t);
    Ok(Tensor::from_vec(out, new_dims, &candle_core::Device::Cpu)?
        .to_dtype(wav.dtype())?
        .to_device(wav.device())?)
}

// Matches Python's scipy.signal.resample_poly behavior
pub fn resample(audio: &Tensor, from_rate: u32, to_rate: u32) -> anyhow::Result<Tensor> {
    if from_rate == to_rate {
        return Ok(audio.clone());
    }

    let shape = audio.dims();
    let channels = shape[0];
    let num_samples = shape[1];

    if num_samples == 0 {
        return Ok(audio.clone());
    }

    use rubato::audioadapter_buffers::direct::SequentialSlice;
    use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};

    let ratio = to_rate as f64 / from_rate as f64;

    // Candle holds the clip as [C][T] row-major, which is exactly rubato's
    // "sequential" layout, so the flattened tensor is handed over as is.
    let input = audio.flatten_all()?.to_vec1::<f32>()?;
    let input = SequentialSlice::new(&input, channels, num_samples)?;

    // Fixed-ratio polynomial (septic) interpolation; `process_all` feeds the
    // whole clip through in chunks and trims the resampler's startup delay.
    let mut resampler = Async::<f32>::new_poly(
        ratio,
        1.0, // max_resample_ratio_relative (1.0: the ratio never changes)
        PolynomialDegree::Septic,
        1024, // chunk size (input frames per internal call)
        channels,
        FixedAsync::Input,
    )?;
    let output = resampler.process_all(&input, num_samples, None)?;

    // rubato returns interleaved frames; regroup them per channel for candle.
    let interleaved = output.take_data();
    let out_samples = interleaved.len() / channels;
    let mut flat_data: Vec<f32> = Vec::with_capacity(channels * out_samples);
    for ch in 0..channels {
        flat_data.extend(interleaved.iter().skip(ch).step_by(channels));
    }

    Ok(Tensor::from_vec(
        flat_data,
        (channels, out_samples),
        audio.device(),
    )?)
}

#[deprecated(note = "Use resample() instead which provides higher quality.")]
pub fn resample_linear(audio: &Tensor, from_rate: u32, to_rate: u32) -> anyhow::Result<Tensor> {
    resample(audio, from_rate, to_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_end_on_pause() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let sr = 1000; // 20-sample frames, 20-sample fade, 80-sample pause
        // 100 samples of tone, then 100 of silence.
        let mut data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.3).sin() * 0.5).collect();
        data.extend(std::iter::repeat_n(0.0, 100));
        let t = Tensor::from_vec(data.clone(), (1, 1, 200), &device)?;
        let out = end_on_pause(&t, sr)?;
        assert_eq!(out.dims(), &[1, 1, 180]); // 100 kept + 80 pause
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        // Untouched before the fade, faded to zero at the cut, silent after.
        assert_eq!(v[..80], data[..80]);
        assert_eq!(v[99], 0.0);
        assert!(v[100..].iter().all(|&x| x == 0.0));

        // A prompt ending on speech keeps all of it and gains the pause.
        let loud = Tensor::ones((1, 1, 100), DType::F32, &device)?;
        assert_eq!(end_on_pause(&loud, sr)?.dims(), &[1, 1, 180]);
        // Shorter than one frame: unchanged.
        let tiny = Tensor::ones((1, 1, 10), DType::F32, &device)?;
        assert_eq!(end_on_pause(&tiny, sr)?.dims(), &[1, 1, 10]);
        Ok(())
    }

    #[test]
    fn test_normalize_peak() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![-0.5f32, 0.2, 0.5], (1, 3), &device)?;
        let normalized = normalize_peak(&t)?;
        let data = normalized.to_vec2::<f32>()?;
        assert_eq!(data[0], vec![-1.0, 0.4, 1.0]);
        Ok(())
    }

    #[test]
    fn test_pcm_i16_le_bytes_clamp_and_interleave() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let data = vec![-1.0f32, 0.0, 1.0, 0.5, -0.5, 2.0];
        let t = Tensor::from_vec(data, (2, 3), &device)?;

        let bytes = pcm_i16_le_bytes(&t)?;
        let samples: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| i16::from_le_bytes(*chunk))
            .collect();

        assert_eq!(samples, vec![-32767, 16383, 0, -16383, 32767, 32767]);
        Ok(())
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_resample() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // rubato works best with reasonable block sizes.
        // Let's use a larger sample count to be safe.
        let input_samples = 1024;
        let data: Vec<f32> = (0..input_samples).map(|i| (i as f32 * 0.1).sin()).collect();
        let t = Tensor::from_vec(data, (1, input_samples), &device)?;

        // Resample 100Hz to 200Hz (Ratio 2.0)
        let resampled = resample(&t, 100, 200)?;
        let out_samples = resampled.dims()[1];

        println!("Resample test: In={}, Out={}", input_samples, out_samples);

        // Expect approx double
        let expected = 2048;
        let diff = (out_samples as i64 - expected as i64).abs();

        assert!(
            diff <= 50,
            "Output samples {} deviates too much from expected {}",
            out_samples,
            expected
        );
        Ok(())
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wav_io() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // Use small values to avoid clipping
        // write_wav applies clamp(-1, 1) to match Python's behavior
        let t = Tensor::from_vec(vec![0.0f32, 0.5, -0.5, 0.1], (1, 4), &device)?;
        let path = "test_io.wav";
        write_wav(path, &t, 16000)?;

        let (read_t, sr) = read_wav(path)?;
        assert_eq!(sr, 16000);
        assert_eq!(read_t.dims(), t.dims());

        // Pre-calculate expected values (clamp doesn't change values in [-1, 1])
        let expected_data: Vec<f32> = vec![0.0, 0.5, -0.5, 0.1];
        let expected = Tensor::from_vec(expected_data, (1, 4), &device)?;

        // Tolerance for 16-bit quantization (1/32768 ~= 3e-5) plus float error
        let diff = (read_t - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "Diff was {}", diff);

        std::fs::remove_file(path)?;
        Ok(())
    }
}
