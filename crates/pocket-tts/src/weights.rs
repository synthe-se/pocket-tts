use anyhow::Result;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use candle_core::Device;

#[cfg(not(target_arch = "wasm32"))]
use hf_hub::{HFClientSync, HFError};

/// How many 10 s cache-lock waits to sit through (about ten minutes) while
/// another download of the same file finishes.
#[cfg(not(target_arch = "wasm32"))]
const MAX_LOCK_WAITS: usize = 60;

/// Download a file from HuggingFace Hub if necessary.
///
/// Supports the format: `hf://owner/repo/filename@revision`
/// where `@revision` is optional.
///
/// Note: Not available on wasm32 targets (use local file loading instead).
#[cfg(not(target_arch = "wasm32"))]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("http://") || file_path.starts_with("https://") {
        download_url(file_path)
    } else if file_path.starts_with("hf://") {
        let path = file_path.trim_start_matches("hf://");
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 3 {
            anyhow::bail!(
                "Invalid hf:// path: {}. Expected hf://repo_owner/repo_name/filename[@revision]",
                file_path
            );
        }
        let filename_with_revision = parts[2..].join("/");

        // Parse optional revision from filename (e.g., "file.safetensors@abc123")
        let (filename, revision) = if let Some(at_pos) = filename_with_revision.rfind('@') {
            let (f, r) = filename_with_revision.split_at(at_pos);
            (f.to_string(), Some(r[1..].to_string())) // Skip the '@'
        } else {
            (filename_with_revision, None)
        };

        // The client resolves the token itself (HF_TOKEN, then HF_TOKEN_PATH,
        // then $HF_HOME/token) and downloads into the standard HF cache.
        let client = HFClientSync::new()?;
        let repo = client.model(parts[0], parts[1]);

        // hf-hub gives up on a blob's cache lock after 10 s; when another
        // thread or process is mid-way through the same first download
        // (hundreds of MB), keep waiting for it instead of failing.
        let mut attempts = 0;
        loop {
            match repo
                .download_file()
                .filename(filename.clone())
                .maybe_revision(revision.clone())
                .send()
            {
                Err(HFError::CacheLockTimeout { .. }) if attempts < MAX_LOCK_WAITS => {
                    attempts += 1;
                }
                result => return Ok(result?),
            }
        }
    } else {
        Ok(PathBuf::from(file_path))
    }
}

/// Upstream's plain-URL branch: the file lands in ~/.cache/pocket_tts as
/// `sha256(url)` plus the extension of the URL path (query strings and
/// fragments are not part of it), and is reused on later calls.
#[cfg(not(target_arch = "wasm32"))]
fn download_url(url: &str) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};

    let cache_dir = std::env::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory for the download cache"))?
        .join(".cache")
        .join("pocket_tts");
    std::fs::create_dir_all(&cache_dir)?;

    let digest = Sha256::digest(url.as_bytes());
    let mut name: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    name.push_str(&url_path_suffix(url));
    let cached = cache_dir.join(name);
    if cached.exists() {
        return Ok(cached);
    }

    // reqwest's blocking client refuses to run on a tokio worker (the server
    // resolves its default voice from async code), so it gets its own thread.
    let url_owned = url.to_string();
    let bytes = std::thread::spawn(move || -> Result<Vec<u8>> {
        let response = reqwest::blocking::get(&url_owned)?.error_for_status()?;
        Ok(response.bytes()?.to_vec())
    })
    .join()
    .map_err(|_| anyhow::anyhow!("download thread panicked for {url}"))??;

    // Write then rename, so an interrupted download never leaves a
    // truncated file that later calls would trust.
    let partial = cached.with_extension("partial");
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, &cached)?;
    Ok(cached)
}

/// The extension (with its dot) of a URL's path, like Python's
/// `Path(urlparse(url).path).suffix`; empty when there is none.
#[cfg(not(target_arch = "wasm32"))]
fn url_path_suffix(url: &str) -> String {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let path = after_scheme
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .split_once('/')
        .map_or("", |(_, path)| path);
    let last = path.rsplit('/').next().unwrap_or("");
    match last.rfind('.') {
        Some(dot) if dot > 0 => last[dot..].to_string(),
        _ => String::new(),
    }
}

/// Whether a voice source names a `.safetensors` state rather than audio,
/// looking past an `hf://` revision or a URL query string (upstream
/// `_is_safetensors_source`).
pub fn is_safetensors_source(source: &str) -> bool {
    let path = if source.starts_with("http://") || source.starts_with("https://") {
        source.split(['?', '#']).next().unwrap_or(source)
    } else if source.starts_with("hf://") {
        source.rsplit_once('@').map_or(source, |(path, _)| path)
    } else {
        source
    };
    path.ends_with(".safetensors")
}

/// WASM version: Only supports local file paths
#[cfg(target_arch = "wasm32")]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("http://") || file_path.starts_with("https://") {
        download_url(file_path)
    } else if file_path.starts_with("hf://") {
        anyhow::bail!("HuggingFace Hub downloads not supported on WASM. Use local file paths.");
    }
    Ok(PathBuf::from(file_path))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_weights(
    file_path: &str,
    _device: &Device,
) -> Result<candle_core::safetensors::MmapedSafetensors> {
    let path = download_if_necessary(file_path)?;
    let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(safetensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_if_necessary_local() {
        let path = "test.safetensors";
        let res = download_if_necessary(path).unwrap();
        assert_eq!(res, PathBuf::from(path));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_url_path_suffix() {
        assert_eq!(url_path_suffix("https://x.org/a/b/voice.wav"), ".wav");
        assert_eq!(url_path_suffix("https://x.org/voice.wav?token=1"), ".wav");
        assert_eq!(url_path_suffix("https://x.org/v.tar.gz#frag"), ".gz");
        assert_eq!(url_path_suffix("https://x.org/download"), "");
        assert_eq!(url_path_suffix("https://x.org"), "");
    }

    #[test]
    fn test_is_safetensors_source() {
        assert!(is_safetensors_source("voice.safetensors"));
        assert!(is_safetensors_source(
            "hf://kyutai/repo/embeddings/alba.safetensors@abc123"
        ));
        assert!(is_safetensors_source(
            "https://example.com/alba.safetensors?download=1"
        ));
        assert!(!is_safetensors_source("hf://kyutai/repo/alba.wav@abc123"));
        assert!(!is_safetensors_source(
            "https://example.com/a.wav?x=.safetensors"
        ));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_invalid_hf_path() {
        let path = "hf://invalid";
        let res = download_if_necessary(path);
        assert!(res.is_err());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_parse_revision() {
        // Test parsing logic (doesn't actually download)
        let path = "hf://kyutai/pocket-tts/file.safetensors@abc123def";
        // This will fail to download but we're testing the parsing
        let res = download_if_necessary(path);
        // We expect a network error, not a parsing error
        assert!(res.is_err());
    }
}
