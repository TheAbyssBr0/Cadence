//! Configuration loading (`~/.config/cadence/config.toml`).
//!
//! Pure construction from TOML text is separated from filesystem access so
//! the parsing logic is unit-testable without touching the home directory.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Location of the production database, relative to [`Config::data_dir`].
pub const DB_FILE_NAME: &str = "cadence.db";
/// Lock file guarding exclusive DB access.
pub const LOCK_FILE_NAME: &str = "cadence.lock";
/// Config file name under the OS config directory.
pub const CONFIG_FILE_NAME: &str = "config.toml";

/// Runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Root directory for books, database, and file store.
    /// Defaults to `~/.cadence`.
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    /// When true (default), overdue work is served before new learning.
    #[serde(default = "default_catch_up_first")]
    pub catch_up_first: bool,
    /// Reserved new units per active day so review debt never starves learning.
    #[serde(default = "default_reserve_new_per_day")]
    pub reserve_new_per_day: u32,
}

const fn default_catch_up_first() -> bool {
    true
}

fn default_data_dir() -> PathBuf {
    Config::default_data_dir()
}

const fn default_reserve_new_per_day() -> u32 {
    1
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: Self::default_data_dir(),
            catch_up_first: true,
            reserve_new_per_day: 1,
        }
    }
}

impl Config {
    fn default_data_dir() -> PathBuf {
        dirs::home_dir().map_or_else(
            || PathBuf::from(".cadence"),
            |mut p| {
                p.push(".cadence");
                p
            },
        )
    }
    /// Parse TOML text into a [`Config`], filling defaults for missing keys.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the TOML is malformed.
    pub fn from_toml_text(text: &str) -> Result<Self> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let mut parsed: Self = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        if parsed.reserve_new_per_day == 0 {
            parsed.reserve_new_per_day = default_reserve_new_per_day();
        }
        Ok(parsed)
    }

    /// Load configuration from one filesystem path, falling back to defaults
    /// when the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] on malformed TOML and [`Error::Io`] on
    /// unreadable files (other than not-found).
    fn load_from(path: &std::path::Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml_text(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(Error::Io(e.to_string())),
        }
    }

    /// Load configuration from disk, falling back to defaults when the file
    /// does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] on malformed TOML and [`Error::Io`] on
    /// unreadable files (other than not-found).
    pub fn load() -> Result<Self> {
        Self::load_from(&Self::config_file_path())
    }

    /// Filesystem path of the config file (`$XDG_CONFIG_HOME/cadence/config.toml`
    /// or `~/.config/cadence/config.toml`).
    #[must_use]
    pub fn config_file_path() -> PathBuf {
        dirs::config_dir().map_or_else(
            || PathBuf::from("config.toml"),
            |mut p| {
                p.push("cadence");
                p.push(CONFIG_FILE_NAME);
                p
            },
        )
    }

    /// Filesystem path of the production SQLite database.
    #[must_use]
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join(DB_FILE_NAME)
    }

    /// Filesystem path of the advisory lock file.
    #[must_use]
    pub fn lock_path(&self) -> PathBuf {
        self.data_dir.join(LOCK_FILE_NAME)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_defaults() {
        let cfg = Config::from_toml_text("").unwrap();
        assert!(cfg.catch_up_first);
        assert_eq!(cfg.reserve_new_per_day, 1);
    }

    #[test]
    fn parses_partial_toml() {
        let cfg = Config::from_toml_text("catch_up_first = false\n").unwrap();
        assert!(!cfg.catch_up_first);
        assert_eq!(cfg.reserve_new_per_day, 1);
        // Missing keys resolve through their default fns, not empty values.
        let bare = Config::from_toml_text("reserve_new_per_day = 3\n").unwrap();
        assert!(bare.catch_up_first);
        assert!(bare.data_dir.ends_with(".cadence"));
    }

    #[test]
    fn rejects_malformed_toml() {
        let err = Config::from_toml_text("catch_up_first = [").unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn zero_reserve_falls_back_to_one() {
        let cfg = Config::from_toml_text("reserve_new_per_day = 0\n").unwrap();
        assert_eq!(cfg.reserve_new_per_day, 1);
    }

    #[test]
    fn db_and_lock_paths_append_filenames() {
        let cfg = Config {
            data_dir: PathBuf::from("/tmp/x"),
            catch_up_first: true,
            reserve_new_per_day: 1,
        };
        assert_eq!(cfg.db_path(), PathBuf::from("/tmp/x/cadence.db"));
        assert_eq!(cfg.lock_path(), PathBuf::from("/tmp/x/cadence.lock"));
    }

    #[test]
    fn config_file_path_names_the_file() {
        let path = Config::config_file_path();
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some(CONFIG_FILE_NAME)
        );
    }

    #[test]
    fn load_from_falls_back_or_errors_by_kind() {
        let dir = std::env::temp_dir().join(format!("cadence-cfg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Missing file: defaults (also pins `load`, which delegates here).
        let missing = dir.join("config.toml");
        assert_eq!(Config::load_from(&missing).unwrap(), Config::default());
        // Present file: parsed.
        std::fs::write(&missing, "catch_up_first = false\n").unwrap();
        assert!(!Config::load_from(&missing).unwrap().catch_up_first);
        // Unreadable path (a directory, not NotFound): Io error, not defaults.
        let blocking = dir.join("blocked.toml");
        std::fs::create_dir_all(&blocking).unwrap();
        assert!(matches!(
            Config::load_from(&blocking).unwrap_err(),
            Error::Io(_)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
