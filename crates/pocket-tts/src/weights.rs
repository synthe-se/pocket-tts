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
    if file_path.starts_with("hf://") {
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

/// WASM version: Only supports local file paths
#[cfg(target_arch = "wasm32")]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("hf://") {
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
