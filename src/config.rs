#![allow(dead_code, unused_imports)]

use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};
use rustc_hash::FxHasher;

#[derive(Clone, Debug, PartialEq)]
pub struct WisdomConfig {
    pub cpu_name: String,
    pub l1_data_cache_kb: u32,
    pub l2_cache_kb: u32,
    pub logical_cores: u32,
    pub has_adx: bool,
    pub has_avx2: bool,
    pub has_bmi2: bool,
    pub gpu_detected: String,
    pub gpu_dedicated: bool,
    pub karatsuba_cutoff_limbs: u32,
    pub toom3_cutoff_limbs: u32,
    pub ntt_cutoff_limbs: u32,
    pub calibrated_32limb_mul_ns: u64,
    pub calibrated_32limb_fmt_ns: u64,
    pub hw_fingerprint: u64,
    pub calib_sealed: bool,
    pub calibration_confidence: f32,
    pub calibration_cv: f32,
    pub calibration_runs: u32,
}

impl Default for WisdomConfig {
    fn default() -> Self {
        Self {
            cpu_name: "Generic x86-64 Host".to_string(),
            l1_data_cache_kb: 32,
            l2_cache_kb: 512,
            logical_cores: 8,
            has_adx: true,
            has_avx2: true,
            has_bmi2: true,
            gpu_detected: "None".to_string(),
            gpu_dedicated: false,
            karatsuba_cutoff_limbs: 24, // Conservative default: ~1,536 bits
            toom3_cutoff_limbs: 128,     // ~8,192 bits
            ntt_cutoff_limbs: 1024,      // ~65,536 bits
            calibrated_32limb_mul_ns: 750,
            calibrated_32limb_fmt_ns: 4_500,
            hw_fingerprint: 0,
            calib_sealed: false,
            calibration_confidence: 0.0,
            calibration_cv: 1.0,
            calibration_runs: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AppConfig {
    pub theme_index: usize,
    pub show_canonical_echo: bool,
    pub live_ghost_answering: bool,
    pub adaptive_debounce_ms: u32,
    pub max_live_calc_ms: u32,
    pub window_width: f32,
    pub window_height: f32,
    pub window_x: i32,
    pub window_y: i32,
    pub wisdom: WisdomConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            theme_index: 0, // Academic Manuscript Cream
            show_canonical_echo: true,
            live_ghost_answering: true,
            adaptive_debounce_ms: 180,
            max_live_calc_ms: 50,
            window_width: 1040.0,
            window_height: 680.0,
            window_x: -1,
            window_y: -1,
            wisdom: WisdomConfig::default(),
        }
    }
}

pub struct AppIdentity {
    pub exe_path: PathBuf,
    pub ini_path: PathBuf,
    pub history_path: PathBuf,
}

static IDENTITY: OnceLock<AppIdentity> = OnceLock::new();
static CONFIG: RwLock<AppConfig> = RwLock::new(AppConfig {
    theme_index: 0,
    show_canonical_echo: true,
    live_ghost_answering: true,
    adaptive_debounce_ms: 180,
    max_live_calc_ms: 50,
    window_width: 1040.0,
    window_height: 680.0,
    window_x: -1,
    window_y: -1,
    wisdom: WisdomConfig {
        cpu_name: String::new(),
        l1_data_cache_kb: 32,
        l2_cache_kb: 512,
        logical_cores: 8,
        has_adx: true,
        has_avx2: true,
        has_bmi2: true,
        gpu_detected: String::new(),
        gpu_dedicated: false,
        karatsuba_cutoff_limbs: 24,
        toom3_cutoff_limbs: 128,
        ntt_cutoff_limbs: 1024,
        calibrated_32limb_mul_ns: 750,
        calibrated_32limb_fmt_ns: 4_500,
        hw_fingerprint: 0,
        calib_sealed: false,
        calibration_confidence: 0.0,
        calibration_cv: 1.0,
        calibration_runs: 0,
    },
});

static LAST_INI_HASH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn get_identity() -> &'static AppIdentity {
    IDENTITY.get_or_init(|| {
        let app_name = env!("APP_DISPLAY_NAME");
        let app_slug = env!("APP_SLUG");
        let default_exe = format!("{app_name}.exe");
        let ini_filename = format!("{app_name}.ini");
        let hist_filename = format!("{app_name}_history.jsonl");

        let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from(&default_exe));
        let (ini_path, history_path) = if std::path::Path::new("Cargo.toml").exists() {
            (PathBuf::from(&ini_filename), PathBuf::from(&hist_filename))
        } else {
            let beside_ini = exe_path.with_extension("ini");
            let beside_hist = exe_path.with_file_name(&hist_filename);
            #[cfg(unix)]
            {
                let exe_dir_writable = exe_path
                    .parent()
                    .and_then(|d| fs::metadata(d).ok())
                    .map_or(false, |m| !m.permissions().readonly());
                if exe_dir_writable {
                    (beside_ini, beside_hist)
                } else {
                    let base = std::env::var_os("XDG_CONFIG_HOME")
                        .map(PathBuf::from)
                        .or_else(|| {
                            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))
                        })
                        .unwrap_or_else(|| PathBuf::from("/tmp"));
                    let dir = base.join(app_slug);
                    let _ = fs::create_dir_all(&dir);
                    (dir.join(&ini_filename), dir.join(&hist_filename))
                }
            }
            #[cfg(not(unix))]
            {
                let _ = app_slug;
                (beside_ini, beside_hist)
            }
        };
        AppIdentity {
            exe_path,
            ini_path,
            history_path,
        }
    })
}

pub fn get_config() -> AppConfig {
    CONFIG.read().map(|c| c.clone()).unwrap_or_default()
}

pub fn update_config<F: FnOnce(&mut AppConfig)>(updater: F) {
    if let Ok(mut lock) = CONFIG.write() {
        updater(&mut *lock);
    }
}

pub fn load_or_create_ini() -> AppConfig {
    let identity = get_identity();
    let mut config = AppConfig::default();
    let mut file_existed = false;

    if let Ok(bytes) = fs::read(&identity.ini_path) {
        file_existed = true;
        let mut hasher = FxHasher::default();
        bytes.hash(&mut hasher);
        LAST_INI_HASH.store(hasher.finish(), std::sync::atomic::Ordering::Relaxed);

        let content = String::from_utf8_lossy(&bytes);
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let key = k.trim().to_lowercase();
                let val = v.trim();

                let parse_bool = |s: &str| -> Option<bool> {
                    match s.to_lowercase().as_str() {
                        "1" | "true" | "yes" | "on" => Some(true),
                        "0" | "false" | "no" | "off" => Some(false),
                        _ => None,
                    }
                };

                match key.as_str() {
                    "theme_index" => { if let Ok(n) = val.parse() { config.theme_index = n; } }
                    "show_canonical_echo" => { if let Some(b) = parse_bool(val) { config.show_canonical_echo = b; } }
                    "live_ghost_answering" => { if let Some(b) = parse_bool(val) { config.live_ghost_answering = b; } }
                    "adaptive_debounce_ms" => { if let Ok(n) = val.parse() { config.adaptive_debounce_ms = n; } }
                    "max_live_calc_ms" => { if let Ok(n) = val.parse() { config.max_live_calc_ms = n; } }
                    "window_width" => { if let Ok(n) = val.parse() { config.window_width = n; } }
                    "window_height" => { if let Ok(n) = val.parse() { config.window_height = n; } }
                    "window_x" => { if let Ok(n) = val.parse() { config.window_x = n; } }
                    "window_y" => { if let Ok(n) = val.parse() { config.window_y = n; } }
                    // Wisdom entries
                    "cpu_name" => { config.wisdom.cpu_name = val.to_string(); }
                    "l1_data_cache_kb" => { if let Ok(n) = val.parse() { config.wisdom.l1_data_cache_kb = n; } }
                    "l2_cache_kb" => { if let Ok(n) = val.parse() { config.wisdom.l2_cache_kb = n; } }
                    "logical_cores" => { if let Ok(n) = val.parse() { config.wisdom.logical_cores = n; } }
                    "has_adx" => { if let Some(b) = parse_bool(val) { config.wisdom.has_adx = b; } }
                    "has_avx2" => { if let Some(b) = parse_bool(val) { config.wisdom.has_avx2 = b; } }
                    "has_bmi2" => { if let Some(b) = parse_bool(val) { config.wisdom.has_bmi2 = b; } }
                    "gpu_detected" => { config.wisdom.gpu_detected = val.to_string(); }
                    "gpu_dedicated" => { if let Some(b) = parse_bool(val) { config.wisdom.gpu_dedicated = b; } }
                    "karatsuba_cutoff_limbs" => { if let Ok(n) = val.parse() { config.wisdom.karatsuba_cutoff_limbs = n; } }
                    "toom3_cutoff_limbs" => { if let Ok(n) = val.parse() { config.wisdom.toom3_cutoff_limbs = n; } }
                    "ntt_cutoff_limbs" => { if let Ok(n) = val.parse() { config.wisdom.ntt_cutoff_limbs = n; } }
                    "calibrated_32limb_mul_ns" => { if let Ok(n) = val.parse() { config.wisdom.calibrated_32limb_mul_ns = n; } }
                    "calibrated_32limb_fmt_ns" => { if let Ok(n) = val.parse() { config.wisdom.calibrated_32limb_fmt_ns = n; } }
                    "hw_fingerprint" => {
                        let hex = val.strip_prefix("0x").or_else(|| val.strip_prefix("0X"));
                        if let Some(h) = hex {
                            if let Ok(n) = u64::from_str_radix(h, 16) {
                                config.wisdom.hw_fingerprint = n;
                            }
                        } else if let Ok(n) = val.parse() {
                            config.wisdom.hw_fingerprint = n;
                        }
                    }
                    "calib_sealed" => { if let Some(b) = parse_bool(val) { config.wisdom.calib_sealed = b; } }
                    "calibration_confidence" => { if let Ok(n) = val.parse() { config.wisdom.calibration_confidence = n; } }
                    "calibration_cv" => { if let Ok(n) = val.parse() { config.wisdom.calibration_cv = n; } }
                    "calibration_runs" => { if let Ok(n) = val.parse() { config.wisdom.calibration_runs = n; } }
                    _ => {}
                }
            }
        }
    }

    let live_fp = crate::engine::wisdom::HardwareProber::fast_cpu_fingerprint();
    let moved_ini = file_existed
        && config.wisdom.hw_fingerprint != 0
        && config.wisdom.hw_fingerprint != live_fp;
    if moved_ini {
        // Moved-.ini Sanitizer: preserve user UI theme/preferences, reset foreign machine geometry & cutoffs
        config.window_x = -1;
        config.window_y = -1;
        config.wisdom.calib_sealed = false;
        config.wisdom.calibration_runs = 0;
        config.wisdom.calibration_confidence = 0.0;
    }

    sanitize_window_geometry(&mut config);

    let needs_fresh_benchmark = !file_existed
        || moved_ini
        || config.wisdom.calibration_runs == 0
        || config.wisdom.cpu_name == "Generic x86-64 Host"
        || config.wisdom.gpu_detected == "None"
        || config.wisdom.gpu_detected.is_empty();

    if needs_fresh_benchmark {
        config.wisdom = crate::engine::wisdom::HardwareProber::probe_and_calibrate(config.wisdom);
        save_ini(&config);
    } else if config.wisdom.calib_sealed && config.wisdom.hw_fingerprint == live_fp {
        // Spec-First < 0.05 ms Warm Startup: restore sealed hardware wisdom directly from atomics
        crate::engine::wisdom::sync_cutoffs_from_config(&config.wisdom);
    } else {
        config.wisdom.hw_fingerprint = live_fp;
        config.wisdom = crate::engine::wisdom::HardwareProber::fast_calibrate_startup(config.wisdom);
        crate::engine::wisdom::sync_cutoffs_from_config(&config.wisdom);
        save_ini(&config);
    }

    if let Ok(mut lock) = CONFIG.write() {
        *lock = config.clone();
    }
    config
}

pub fn sanitize_window_geometry(config: &mut AppConfig) {
    if !config.window_width.is_finite() || config.window_width < 580.0 {
        config.window_width = 1040.0;
    } else {
        config.window_width = config.window_width.clamp(600.0, 3840.0);
    }
    if !config.window_height.is_finite() || config.window_height < 280.0 {
        config.window_height = 680.0;
    } else {
        config.window_height = config.window_height.clamp(290.0, 2160.0);
    }

    #[cfg(windows)]
    if config.window_x != -1
        && config.window_y != -1
        && config.window_x != i32::MIN
        && config.window_y != i32::MIN
    {
        #[link(name = "user32")]
        unsafe extern "system" {
            fn GetSystemMetrics(n_index: i32) -> i32;
        }
        unsafe {
            let virt_x = GetSystemMetrics(76); // SM_XVIRTUALSCREEN
            let virt_y = GetSystemMetrics(77); // SM_YVIRTUALSCREEN
            let virt_w = GetSystemMetrics(78); // SM_CXVIRTUALSCREEN
            let virt_h = GetSystemMetrics(79); // SM_CYVIRTUALSCREEN
            if virt_w > 0 && virt_h > 0 {
                let virt_r = virt_x + virt_w;
                let virt_b = virt_y + virt_h;
                if config.window_x < virt_x - 32
                    || config.window_y < virt_y - 32
                    || config.window_x + 100 > virt_r
                    || config.window_y + 100 > virt_b
                {
                    config.window_x = -1;
                    config.window_y = -1;
                }
            }
        }
    }
}

pub fn save_ini(config: &AppConfig) {
    let identity = get_identity();
    let text = format!(
        "; ==========================================================================\n\
         ; {} ({}) - Configuration & Hardware Wisdom Cache\n\
         ; ==========================================================================\n\n\
         [Interface]\n\
         theme_index = {}\n\
         show_canonical_echo = {}\n\
         live_ghost_answering = {}\n\
         adaptive_debounce_ms = {}\n\
         max_live_calc_ms = {}\n\
         window_width = {:.1}\n\
         window_height = {:.1}\n\
         window_x = {}\n\
         window_y = {}\n\n\
         [Hardware_Wisdom]\n\
         hw_fingerprint = 0x{:016X}\n\
         calib_sealed = {}\n\
         cpu_name = {}\n\
         l1_data_cache_kb = {}\n\
         l2_cache_kb = {}\n\
         logical_cores = {}\n\
         has_adx = {}\n\
         has_avx2 = {}\n\
         has_bmi2 = {}\n\
         gpu_detected = {}\n\
         gpu_dedicated = {}\n\
         karatsuba_cutoff_limbs = {}\n\
         toom3_cutoff_limbs = {}\n\
         ntt_cutoff_limbs = {}\n\
         calibrated_32limb_mul_ns = {}\n\
         calibrated_32limb_fmt_ns = {}\n\
         calibration_confidence = {:.3}\n\
         calibration_cv = {:.4}\n\
         calibration_runs = {}\n",
        env!("APP_DISPLAY_NAME"),
        env!("APP_PUBLISHER"),
        config.theme_index,
        config.show_canonical_echo,
        config.live_ghost_answering,
        config.adaptive_debounce_ms,
        config.max_live_calc_ms,
        config.window_width,
        config.window_height,
        config.window_x,
        config.window_y,
        config.wisdom.hw_fingerprint,
        config.wisdom.calib_sealed,
        config.wisdom.cpu_name,
        config.wisdom.l1_data_cache_kb,
        config.wisdom.l2_cache_kb,
        config.wisdom.logical_cores,
        config.wisdom.has_adx,
        config.wisdom.has_avx2,
        config.wisdom.has_bmi2,
        config.wisdom.gpu_detected,
        config.wisdom.gpu_dedicated,
        config.wisdom.karatsuba_cutoff_limbs,
        config.wisdom.toom3_cutoff_limbs,
        config.wisdom.ntt_cutoff_limbs,
        config.wisdom.calibrated_32limb_mul_ns,
        config.wisdom.calibrated_32limb_fmt_ns,
        config.wisdom.calibration_confidence,
        config.wisdom.calibration_cv,
        config.wisdom.calibration_runs,
    );

    let bytes = text.as_bytes();
    let mut hasher = FxHasher::default();
    bytes.hash(&mut hasher);
    let new_hash = hasher.finish();

    // SOTA Compare-Before-Write Invariant: eliminate flash SSD wear
    if LAST_INI_HASH.load(std::sync::atomic::Ordering::Relaxed) == new_hash {
        return;
    }

    if fs::write(&identity.ini_path, bytes).is_ok() {
        LAST_INI_HASH.store(new_hash, std::sync::atomic::Ordering::Relaxed);
    }
}
