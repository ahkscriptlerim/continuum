use egui::{Key, Label, RichText, ScrollArea, TextEdit, Ui, vec2};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver, RecvTimeoutError};
use std::sync::atomic::{AtomicU64, Ordering};
use rustc_hash::FxHashMap;

use crate::config::AppConfig;
use crate::engine::{
    execute_with_telemetry_with_counter, execute_with_telemetry_with_counter_mode,
    CalcValue, EvalResult, EpochGuard, CANCELLATION_EPOCH,
    classify_input_cost, ExecTier, InputClassification, is_task_aborted,
};
use crate::engine::diagnostics::{run_all_diagnostics, DiagnosticsReport};
use crate::engine::bignum::RadixMode;
use crate::engine::value::format_clean_float;
use super::paper_theme::{
    show_exact_card, ui_pill_toggle, ui_sym_hline, ui_sym_vline, PaperPalette, HEADER_ROW_HEIGHT,
    SIDEBAR_TOGGLE_BTN_SIZE,
};
use super::sidebar::SidebarState;

pub const RADIX_DEC_BIT: u8 = 1 << 0;
pub const RADIX_HEX_BIT: u8 = 1 << 1;
pub const RADIX_OCT_BIT: u8 = 1 << 2;
pub const RADIX_BIN_BIT: u8 = 1 << 3;

pub const ALL_RADICES: [(RadixMode, u8, &str, &str); 4] = [
    (
        RadixMode::Dec,
        RADIX_DEC_BIT,
        "DEC",
        "Decimal (Base 10) — Click to toggle display (at least 1 base remains active)",
    ),
    (
        RadixMode::Hex,
        RADIX_HEX_BIT,
        "HEX",
        "Hexadecimal (Base 16) — Click to toggle display (at least 1 base remains active)",
    ),
    (
        RadixMode::Oct,
        RADIX_OCT_BIT,
        "OCT",
        "Octal (Base 8) — Click to toggle display (at least 1 base remains active)",
    ),
    (
        RadixMode::Bin,
        RADIX_BIN_BIT,
        "BIN",
        "Binary (Base 2) — Click to toggle display (at least 1 base remains active)",
    ),
];

#[inline(always)]
pub const fn radix_to_bit(r: RadixMode) -> u8 {
    match r {
        RadixMode::Dec => RADIX_DEC_BIT,
        RadixMode::Hex => RADIX_HEX_BIT,
        RadixMode::Oct => RADIX_OCT_BIT,
        RadixMode::Bin => RADIX_BIN_BIT,
    }
}

#[inline(always)]
pub const fn radix_to_slot(r: RadixMode) -> usize {
    match r {
        RadixMode::Dec => 0,
        RadixMode::Hex => 1,
        RadixMode::Oct => 2,
        RadixMode::Bin => 3,
    }
}

#[cfg(windows)]
pub mod win_clipboard {
    use std::ffi::c_void;

    const CF_UNICODETEXT: u32 = 13;
    const GMEM_MOVEABLE: u32 = 0x0002;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn OpenClipboard(hWndNewOwner: *mut c_void) -> i32;
        fn CloseClipboard() -> i32;
        fn EmptyClipboard() -> i32;
        fn SetClipboardData(uFormat: u32, hMem: *mut c_void) -> *mut c_void;
        fn GetClipboardData(uFormat: u32) -> *mut c_void;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalAlloc(uFlags: u32, dwBytes: usize) -> *mut c_void;
        fn GlobalFree(hMem: *mut c_void) -> *mut c_void;
        fn GlobalLock(hMem: *mut c_void) -> *mut c_void;
        fn GlobalUnlock(hMem: *mut c_void) -> i32;
    }

    unsafe fn open_clipboard_with_retry() -> bool {
        for attempt in 0..5 {
            if unsafe { OpenClipboard(std::ptr::null_mut()) } != 0 {
                return true;
            }
            if attempt < 4 {
                std::thread::sleep( std::time::Duration::from_millis(1));
            }
        }
        false
    }

    pub fn set_text(text: &str) -> bool {
        let mut utf16: Vec<u16> = text.encode_utf16().collect();
        utf16.push(0);
        let byte_len = utf16.len() * std::mem::size_of::<u16>();
        unsafe {
            if !open_clipboard_with_retry() {
                return false;
            }
            let _ = EmptyClipboard();
            let hmem = GlobalAlloc(GMEM_MOVEABLE, byte_len);
            if hmem.is_null() {
                let _ = CloseClipboard();
                return false;
            }
            let ptr = GlobalLock(hmem) as *mut u16;
            if ptr.is_null() {
                let _ = GlobalFree(hmem);
                let _ = CloseClipboard();
                return false;
            }
            std::ptr::copy_nonoverlapping(utf16.as_ptr(), ptr, utf16.len());
            let _ = GlobalUnlock(hmem);
            let res = SetClipboardData(CF_UNICODETEXT, hmem);
            if res.is_null() {
                let _ = GlobalFree(hmem);
                let _ = CloseClipboard();
                return false;
            }
            let _ = CloseClipboard();
            true
        }
    }

    pub fn get_text() -> Option<String> {
        unsafe {
            if !open_clipboard_with_retry() {
                return None;
            }
            let hmem = GetClipboardData(CF_UNICODETEXT);
            if hmem.is_null() {
                let _ = CloseClipboard();
                return None;
            }
            let ptr = GlobalLock(hmem) as *const u16;
            if ptr.is_null() {
                let _ = CloseClipboard();
                return None;
            }
            let mut len = 0usize;
            while *ptr.add(len) != 0 {
                len += 1;
            }
            let slice = std::slice::from_raw_parts(ptr, len);
            let s = String::from_utf16_lossy(slice);
            let _ = GlobalUnlock(hmem);
            let _ = CloseClipboard();
            Some(s)
        }
    }
}

static CLIPBOARD_MIRROR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[cfg(unix)]
#[derive(Clone, Copy)]
struct UnixClipboardBackend {
    write_bin: &'static str,
    write_args: &'static [&'static str],
    read_bin: &'static str,
    read_args: &'static [&'static str],
}

#[cfg(unix)]
fn detect_unix_clipboard_backend() -> Option<UnixClipboardBackend> {
    static BACKEND: std::sync::OnceLock<Option<UnixClipboardBackend>> = std::sync::OnceLock::new();
    *BACKEND.get_or_init(|| {
        let has_bin = |name: &str| -> bool {
            std::env::var_os("PATH").map_or(false, |paths| {
                std::env::split_paths(&paths).any(|dir| dir.join(name).is_file())
            })
        };
        if has_bin("pbcopy") && has_bin("pbpaste") {
            Some(UnixClipboardBackend {
                write_bin: "pbcopy",
                write_args: &[],
                read_bin: "pbpaste",
                read_args: &[],
            })
        } else if std::env::var_os("WAYLAND_DISPLAY").is_some() && has_bin("wl-copy") && has_bin("wl-paste") {
            Some(UnixClipboardBackend {
                write_bin: "wl-copy",
                write_args: &[],
                read_bin: "wl-paste",
                read_args: &["--no-newline"],
            })
        } else if has_bin("xclip") {
            Some(UnixClipboardBackend {
                write_bin: "xclip",
                write_args: &["-selection", "clipboard"],
                read_bin: "xclip",
                read_args: &["-selection", "clipboard", "-o"],
            })
        } else if has_bin("xsel") {
            Some(UnixClipboardBackend {
                write_bin: "xsel",
                write_args: &["--clipboard", "--input"],
                read_bin: "xsel",
                read_args: &["--clipboard", "--output"],
            })
        } else {
            None
        }
    })
}

pub fn mirror_clipboard_text(text: &str) {
    if let Ok(mut guard) = CLIPBOARD_MIRROR.lock() {
        *guard = Some(text.to_owned());
    }
    #[cfg(unix)]
    if let Some(backend) = detect_unix_clipboard_backend() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        if let Ok(mut child) = Command::new(backend.write_bin)
            .args(backend.write_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
        }
    }
}

pub fn set_os_clipboard_text(ctx: &egui::Context, text: &str) {
    mirror_clipboard_text(text);
    #[cfg(windows)]
    {
        let _ = win_clipboard::set_text(text);
    }
    ctx.copy_text(text.to_owned());
    ctx.output_mut(|o| o.copied_text = text.to_owned());
}

pub fn get_os_clipboard_text() -> Option<String> {
    #[cfg(windows)]
    {
        if let Some(s) = win_clipboard::get_text() {
            return Some(s);
        }
    }
    #[cfg(unix)]
    if let Some(backend) = detect_unix_clipboard_backend() {
        use std::process::{Command, Stdio};
        if let Ok(out) = Command::new(backend.read_bin)
            .args(backend.read_args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
        {
            if out.status.success() && !out.stdout.is_empty() {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    return Some(s);
                }
            }
        }
    }
    CLIPBOARD_MIRROR.lock().ok().and_then(|g| g.clone())
}

#[inline]
pub fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map_or(s.len(), |(b, _)| b)
}

#[inline]
fn effective_input_slice(s: &str) -> &str {
    s.trim().trim_end_matches(|c: char| {
        c.is_whitespace() || matches!(c, '+' | '-' | '*' | '/' | '^')
    })
}

#[derive(Clone, Debug)]
pub struct PreparedDisplay {
    pub dec_str: String,
    pub radix_mode: RadixMode,
    pub radix_str: String,
    pub display_preview_str: String,
    pub multiline_str: String,
    pub per_radix: [Option<String>; 4],
    pub multiline_by_radix: [Option<String>; 4],
    pub formatted_mask: u8,
    pub rational_str: Option<String>,
    pub prime_factors_str: Option<String>,
}

impl PreparedDisplay {
    #[cfg(test)]
    pub fn from_value(val: &CalcValue, radix_mode: RadixMode) -> Option<Self> { // wiring-ok
        Self::from_value_mask(val, radix_to_bit(radix_mode))
    }

    /// Formats all active radices in `radix_mask` (plus `DEC` for `last_ans` & ratio dedup)
    /// concurrently in parallel when `>= 2` bases are requested and limb count exceeds the
    /// hardware-calibrated L2/core crossover (`radix_parallel_limbs_cutoff`), avoiding thread
    /// wake overhead on tiny integers while parallelizing multi-thousand-digit conversions!
    pub fn from_value_mask(val: &CalcValue, radix_mask: u8) -> Option<Self> {
        if is_task_aborted() {
            return None;
        }
        let clean_mask = if (radix_mask & 0b1111) == 0 {
            RADIX_DEC_BIT
        } else {
            radix_mask & 0b1111
        };
        let need_mask = clean_mask | RADIX_DEC_BIT;
        let primary = first_active_radix(clean_mask);

        let mut per_radix: [Option<String>; 4] = [None, None, None, None];
        let mut multiline_by_radix: [Option<String>; 4] = [None, None, None, None];
        let is_multi_box = matches!(val, CalcValue::Matrix { .. } | CalcValue::Logic { .. });
        let par_bits_cutoff =
            (crate::engine::wisdom::radix_parallel_limbs_cutoff() as u64).saturating_mul(64);

        if need_mask.count_ones() >= 2 && val.bit_length_estimate() >= par_bits_cutoff {
            // Concurrent zero-recalculation parallel conversion across requested radices
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(4);
                for (r, bit, _, _) in ALL_RADICES {
                    if (need_mask & bit) != 0 {
                        handles.push((
                            radix_to_slot(r),
                            s.spawn(move || {
                                let s_rad = val.format_radix(r);
                                let s_ml = if is_multi_box {
                                    val.format_multiline(r)
                                } else {
                                    s_rad.clone()
                                };
                                (s_rad, s_ml)
                            }),
                        ));
                    }
                }
                for (slot, h) in handles {
                    if let Ok((sr, sml)) = h.join() {
                        per_radix[slot] = Some(sr);
                        multiline_by_radix[slot] = Some(sml);
                    }
                }
            });
        } else {
            for (r, bit, _, _) in ALL_RADICES {
                if (need_mask & bit) != 0 {
                    let slot = radix_to_slot(r);
                    let s_rad = val.format_radix(r);
                    let s_ml = if is_multi_box {
                        val.format_multiline(r)
                    } else {
                        s_rad.clone()
                    };
                    per_radix[slot] = Some(s_rad);
                    multiline_by_radix[slot] = Some(s_ml);
                }
            }
        }

        let dec_str = per_radix[0].clone().unwrap_or_default();
        if is_task_aborted() || dec_str.is_empty() {
            return None;
        }
        let p_slot = radix_to_slot(primary);
        let radix_str = per_radix[p_slot].clone().unwrap_or_else(|| dec_str.clone());
        if is_task_aborted() || radix_str.is_empty() {
            return None;
        }
        let multiline_str = multiline_by_radix[p_slot]
            .clone()
            .unwrap_or_else(|| radix_str.clone());
        let display_preview_str = radix_str.clone();

        let rational_str = val.format_rational();
        let prime_factors_str = val.format_prime_factors();
        if is_task_aborted() {
            return None;
        }
        Some(Self {
            dec_str,
            radix_mode: primary,
            radix_str,
            display_preview_str,
            multiline_str,
            per_radix,
            multiline_by_radix,
            formatted_mask: need_mask,
            rational_str,
            prime_factors_str,
        })
    }

    /// Ensures all radices in `mask` are formatted & cached from the already-computed `CalcValue`
    /// in memory (`0` expression re-parsing or re-calculation; parallelized if `>= 2` missing).
    pub fn ensure_mask(&mut self, val: &CalcValue, mask: u8) {
        let clean = if (mask & 0b1111) == 0 {
            RADIX_DEC_BIT
        } else {
            mask & 0b1111
        };
        let missing = clean & !self.formatted_mask;
        if missing != 0 {
            let _guard = EpochGuard::new(0);
            let is_multi_box = matches!(val, CalcValue::Matrix { .. } | CalcValue::Logic { .. });
            let par_bits_cutoff =
                (crate::engine::wisdom::radix_parallel_limbs_cutoff() as u64).saturating_mul(64);
            if missing.count_ones() >= 2 && val.bit_length_estimate() >= par_bits_cutoff {
                let mut pairs: Vec<(usize, String, String)> = Vec::with_capacity(4);
                std::thread::scope(|s| {
                    let mut handles = Vec::with_capacity(4);
                    for (r, bit, _, _) in ALL_RADICES {
                        if (missing & bit) != 0 {
                            handles.push((
                                radix_to_slot(r),
                                s.spawn(move || {
                                    let sr = val.format_radix(r);
                                    let sml = if is_multi_box {
                                        val.format_multiline(r)
                                    } else {
                                        sr.clone()
                                    };
                                    (sr, sml)
                                }),
                            ));
                        }
                    }
                    for (slot, h) in handles {
                        if let Ok((sr, sml)) = h.join() {
                            pairs.push((slot, sr, sml));
                        }
                    }
                });
                for (slot, sr, sml) in pairs {
                    self.per_radix[slot] = Some(sr);
                    self.multiline_by_radix[slot] = Some(sml);
                }
            } else {
                for (r, bit, _, _) in ALL_RADICES {
                    if (missing & bit) != 0 {
                        let slot = radix_to_slot(r);
                        let sr = val.format_radix(r);
                        let sml = if is_multi_box {
                            val.format_multiline(r)
                        } else {
                            sr.clone()
                        };
                        self.per_radix[slot] = Some(sr);
                        self.multiline_by_radix[slot] = Some(sml);
                    }
                }
            }
            self.formatted_mask |= missing;
        }

        let primary = first_active_radix(clean);
        let p_slot = radix_to_slot(primary);
        self.radix_mode = primary;
        if let Some(ref s) = self.per_radix[p_slot] {
            self.radix_str = s.clone();
            self.display_preview_str = s.clone();
        }
        if let Some(ref ml) = self.multiline_by_radix[p_slot] {
            self.multiline_str = ml.clone();
        }
    }

    #[inline]
    pub fn str_for_radix(&self, r: RadixMode) -> &str {
        self.per_radix[radix_to_slot(r)]
            .as_deref()
            .unwrap_or(self.radix_str.as_str())
    }

    #[inline]
    pub fn multiline_for_radix(&self, r: RadixMode) -> &str {
        self.multiline_by_radix[radix_to_slot(r)]
            .as_deref()
            .unwrap_or_else(|| self.str_for_radix(r))
    }

    /// Returns the single deduplicated secondary equivalence subline (integer ratio / prime factors)
    /// to render once under the answer framebox, or `None` if empty or identical to the primary answer.
    pub fn answer_equiv_subline(&self) -> Option<String> {
        let mut parts: Vec<&str> = Vec::with_capacity(2);
        if let Some(ref frac) = self.rational_str {
            if !frac.is_empty() && frac != &self.radix_str && frac != &self.dec_str {
                parts.push(frac.as_str());
            }
        }
        if let Some(ref factors) = self.prime_factors_str {
            if !factors.is_empty()
                && factors != &self.radix_str
                && factors != &self.dec_str
                && !parts.contains(&factors.as_str())
            {
                parts.push(factors.as_str());
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("  ·  "))
        }
    }
}

#[inline(always)]
pub const fn first_active_radix(mask: u8) -> RadixMode {
    if (mask & RADIX_DEC_BIT) != 0 {
        RadixMode::Dec
    } else if (mask & RADIX_HEX_BIT) != 0 {
        RadixMode::Hex
    } else if (mask & RADIX_OCT_BIT) != 0 {
        RadixMode::Oct
    } else {
        RadixMode::Bin
    }
}


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeckCategory {
    MathTrig,
    Logic,
    LinearAlgebra,
    NumberTheory,
    Programmer,
    StatsUnits,
}

impl DeckCategory {
    pub fn label(&self) -> &'static str {
        match self {
            DeckCategory::MathTrig => "Math & Trig",
            DeckCategory::Logic => "Logic",
            DeckCategory::LinearAlgebra => "Linear Algebra",
            DeckCategory::NumberTheory => "Number Theory",
            DeckCategory::Programmer => "Bitwise",
            DeckCategory::StatsUnits => "Stats & Units",
        }
    }

    pub fn tooltip(&self) -> &'static str {
        match self {
            DeckCategory::MathTrig => {
                "Trigonometry, Hyperbolics, Logarithms, Roots, Factorials & Mathematical Constants"
            }
            DeckCategory::Logic => {
                "Propositional Boolean Algebra, Implication & Automated Truth-Table Solver"
            }
            DeckCategory::LinearAlgebra => {
                "Matrix & Vector Arithmetic, Determinants, Inverses, Eigenvalues & Linear Solvers"
            }
            DeckCategory::NumberTheory => {
                "Prime Factorization, Combinatorics, Symbolic Calculus & Polynomial Roots"
            }
            DeckCategory::Programmer => {
                "Exact 64-Bit Programmer Bitwise Gates, Shifts, Rotations & Base Prefixes"
            }
            DeckCategory::StatsUnits => {
                "Descriptive Statistics & Exact Rational SI / Imperial Unit Conversions"
            }
        }
    }
}

pub struct CalcTask {
    pub input: String,
    pub epoch: u64,
    pub radix_mask: u8,
    pub vars: FxHashMap<String, CalcValue>,
    pub hold_wall_ms: u64,
    pub deep_mode: bool,
}

pub struct CalcResponse {
    pub epoch: u64,
    pub result: Result<(EvalResult, PreparedDisplay), String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UndoEntry {
    pub text: String,
    pub cursor_char: usize,
}

pub struct EditorState {
    pub input_text: String,
    pub last_computed_input: String,
    pub last_effective_input: String,
    pub live_result: Option<EvalResult>,
    pub cached_display: Option<PreparedDisplay>,
    pub live_error: Option<String>,
    pub is_modified: bool,
    pub last_keystroke: Instant,
    pub keystroke_interval_ema: f32, // Adaptive typing liveliness EMA (ms)
    pub radix_mode: RadixMode,
    pub radix_mask: u8,
    #[allow(dead_code)]
    pub show_cmd_stream: bool,
    #[allow(dead_code)]
    pub stream_chunk_size: usize,
    pub show_function_deck: bool,
    pub deck_category: DeckCategory,
    pub last_ans: String,
    pub copy_notice_timer: Option<Instant>,
    pub copy_notice_slot: usize,
    pub memory_sum: f64,
    pub memory_val: Option<CalcValue>,
    pub request_focus: bool,
    pub current_epoch: u64,
    pub epoch_counter: Arc<AtomicU64>,
    pub task_sender: Sender<CalcTask>,
    pub response_receiver: Receiver<CalcResponse>,
    pub show_diagnostics: bool,
    pub diagnostics_report: Option<DiagnosticsReport>,
    pub task_in_flight: bool,
    pub pending_calc: bool,
    pub history_nav_idx: Option<usize>,
    pub request_cursor_pos: Option<usize>,
    pub selection_range: (usize, usize),
    pub request_selection: Option<(usize, usize)>,
    pub undo_stack: Vec<UndoEntry>,
    pub redo_stack: Vec<UndoEntry>,
    pub last_undo_snapshot: UndoEntry,
    pub calc_start_time: Option<Instant>,
    pub last_label_selection_sig: Option<u64>,
    pub last_show_sidebar_btn_rect: Option<egui::Rect>,
    pub last_card_rects: Vec<egui::Rect>,
}

impl EditorState {
    pub fn new() -> Self {
        let (tx_task, rx_task) = channel::<CalcTask>();
        let (tx_resp, rx_resp) = channel::<CalcResponse>();
        let epoch_counter = Arc::new(AtomicU64::new(0));
        let worker_counter = Arc::clone(&epoch_counter);

        std::thread::Builder::new()
            .name("LiveCalcWorker".into())
            .spawn(move || {
                while let Ok(mut task) = rx_task.recv() {
                    // 1. Drain and coalesce all queued intermediate tasks
                    while let Ok(newer) = rx_task.try_recv() {
                        task = newer;
                    }
                    // 2. Smart Input-Spam Wall: If this heavy calculation arrived during rapid typing,
                    //    hold at the wall for `hold_wall_ms`. Any new input arriving during this window
                    //    overwrites `task` before the heavy intermediate state ever starts!
                    while task.hold_wall_ms > 0 {
                        let wait_dur = Duration::from_millis(task.hold_wall_ms);
                        match rx_task.recv_timeout(wait_dur) {
                            Ok(newer) => {
                                task = newer;
                                while let Ok(latest) = rx_task.try_recv() {
                                    task = latest;
                                }
                            }
                            Err(RecvTimeoutError::Timeout) => break,
                            Err(RecvTimeoutError::Disconnected) => return,
                        }
                    }

                    // 3. Skip if superseded by a newer epoch (e.g. fast-path or clear)
                    if worker_counter.load(Ordering::Acquire) > task.epoch {
                        continue;
                    }

                    // 4. Execute calculation AND radix formatting under cooperative inner-loop preemption
                    let mut worker_vars = task.vars;
                    let outcome = {
                        let _guard = EpochGuard::with_counter_and_mode(
                            task.epoch,
                            &worker_counter,
                            task.deep_mode,
                        );
                        match execute_with_telemetry_with_counter_mode(
                            &task.input,
                            &mut worker_vars,
                            task.epoch,
                            &worker_counter,
                            task.deep_mode,
                        ) {
                            Ok(eval_res) => {
                                if let Some(disp) =
                                    PreparedDisplay::from_value_mask(&eval_res.value, task.radix_mask)
                                {
                                    Ok((eval_res, disp))
                                } else {
                                    Err("Aborted by newer input".to_string())
                                }
                            }
                            Err(e) => Err(e),
                        }
                    };

                    // 5. Only emit response if this task epoch is still the latest state of the input
                    if worker_counter.load(Ordering::Acquire) == task.epoch {
                        let _ = tx_resp.send(CalcResponse {
                            epoch: task.epoch,
                            result: outcome,
                        });
                    }
                }
            })
            .expect("Failed to spawn LiveCalcWorker thread");

        Self {
            input_text: String::new(),
            last_computed_input: String::new(),
            last_effective_input: String::new(),
            live_result: None,
            cached_display: None,
            live_error: None,
            is_modified: false,
            last_keystroke: Instant::now(),
            keystroke_interval_ema: 200.0,
            radix_mode: RadixMode::Dec,
            radix_mask: RADIX_DEC_BIT,
            show_cmd_stream: false,
            stream_chunk_size: 60,
            show_function_deck: true,
            deck_category: DeckCategory::MathTrig,
            last_ans: "0".to_string(),
            copy_notice_timer: None,
            copy_notice_slot: 0,
            memory_sum: 0.0,
            memory_val: None,
            request_focus: true,
            current_epoch: 0,
            epoch_counter,
            task_sender: tx_task,
            response_receiver: rx_resp,
            show_diagnostics: false,
            diagnostics_report: None,
            task_in_flight: false,
            pending_calc: false,
            history_nav_idx: None,
            request_cursor_pos: None,
            selection_range: (0, 0),
            request_selection: None,
            undo_stack: Vec::with_capacity(128),
            redo_stack: Vec::with_capacity(64),
            last_undo_snapshot: UndoEntry {
                text: String::new(),
                cursor_char: 0,
            },
            calc_start_time: None,
            last_label_selection_sig: None,
            last_show_sidebar_btn_rect: None,
            last_card_rects: Vec::with_capacity(8),
        }
    }

    /// Sets the active radix bitmask (`DEC=1, HEX=2, OCT=4, BIN=8`), enforcing `>= 1` active base.
    pub fn set_radix_mask(&mut self, mask: u8) {
        let clean = mask & 0b1111;
        self.radix_mask = if clean == 0 { RADIX_DEC_BIT } else { clean };
        self.radix_mode = first_active_radix(self.radix_mask);
    }

    /// Independently toggles a radix display pill while guaranteeing at least 1 active base remains.
    pub fn toggle_radix(&mut self, r: RadixMode) {
        let bit = radix_to_bit(r);
        if (self.radix_mask & bit) != 0 {
            // Only allow turning OFF if at least one other radix is currently active
            if (self.radix_mask & !bit) != 0 {
                self.radix_mask &= !bit;
            }
        } else {
            self.radix_mask |= bit;
        }
        self.radix_mode = first_active_radix(self.radix_mask);
    }

    /// Returns active radices in canonical `DEC -> HEX -> OCT -> BIN` order into a stack array.
    #[inline]
    pub fn active_radices(&self) -> ([RadixMode; 4], usize) {
        let mut out = [RadixMode::Dec; 4];
        let mut len = 0usize;
        for (r, bit, _, _) in ALL_RADICES {
            if (self.radix_mask & bit) != 0 {
                out[len] = r;
                len += 1;
            }
        }
        if len == 0 {
            (out, 1)
        } else {
            (out, len)
        }
    }

    pub fn sync_initial_input(&mut self) {
        let len = self.input_text.chars().count();
        self.selection_range = (len, len);
        self.request_selection = Some((len, len));
        self.last_undo_snapshot = UndoEntry {
            text: self.input_text.clone(),
            cursor_char: len,
        };
    }

    pub fn record_external_change(&mut self, prev_text: String) {
        if self.input_text != prev_text {
            let prev_cursor = self.selection_range.1.min(prev_text.chars().count());
            self.undo_stack.push(UndoEntry {
                text: prev_text,
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
            let new_len = self.input_text.chars().count();
            self.selection_range = (new_len, new_len);
            self.request_selection = Some((new_len, new_len));
            self.last_undo_snapshot = UndoEntry {
                text: self.input_text.clone(),
                cursor_char: new_len,
            };
            self.request_focus = true;
        }
    }

    #[inline]
    pub fn normalized_selection(&self) -> (usize, usize) {
        let total = self.input_text.chars().count();
        let lo = self.selection_range.0.min(self.selection_range.1).min(total);
        let hi = self.selection_range.0.max(self.selection_range.1).min(total);
        (lo, hi)
    }

    #[inline]
    #[allow(dead_code)]
    pub fn has_selection(&self) -> bool {
        let (lo, hi) = self.normalized_selection();
        lo < hi
    }

    /// Ctrl+A: Select entire input text as if on the user input control.
    pub fn action_select_all(&mut self) -> (usize, usize) {
        let total = self.input_text.chars().count();
        self.selection_range = (0, total);
        self.request_selection = Some((0, total));
        self.request_focus = true;
        (0, total)
    }

    /// Ctrl+C: When nothing is selected (`caret_start == caret_end` or unfocused), copies the whole
    /// input text to clipboard; when a selection exists (`caret_start != caret_end`), copies the selection.
    pub fn action_copy(&mut self) -> String {
        let (lo, hi) = self.normalized_selection();
        let copied = if lo < hi {
            let b_lo = char_to_byte_idx(&self.input_text, lo);
            let b_hi = char_to_byte_idx(&self.input_text, hi);
            self.input_text[b_lo..b_hi].to_string()
        } else {
            self.input_text.clone()
        };
        if !copied.is_empty() {
            self.copy_notice_timer = Some(Instant::now());
        }
        copied
    }

    /// Ctrl+X: Cuts selection if active, otherwise cuts the whole input text.
    pub fn action_cut(&mut self) -> String {
        let copied = self.action_copy();
        let (lo, hi) = self.normalized_selection();
        if lo < hi {
            let prev_text = self.input_text.clone();
            let prev_cursor = self.selection_range.1.min(prev_text.chars().count());
            let b_lo = char_to_byte_idx(&self.input_text, lo);
            let b_hi = char_to_byte_idx(&self.input_text, hi);
            self.input_text.replace_range(b_lo..b_hi, "");
            self.undo_stack.push(UndoEntry {
                text: prev_text,
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
            self.selection_range = (lo, lo);
            self.request_selection = Some((lo, lo));
            self.last_undo_snapshot = UndoEntry {
                text: self.input_text.clone(),
                cursor_char: lo,
            };
            self.request_focus = true;
            self.trigger_input_wall();
        } else if !self.input_text.is_empty() {
            self.clear();
        }
        copied
    }

    /// Ctrl+V / Text Insertion: Replaces active selection if `caret_start != caret_end`, or inserts
    /// at the current caret (or end of input) when nothing is selected, recording onto the limitless undo stack.
    pub fn action_paste(&mut self, raw_text: &str) {
        let trimmed = raw_text.trim_matches(|c| c == '\r' || c == '\n');
        if trimmed.is_empty() {
            return;
        }
        let normalized_owned;
        let clean: &str = if trimmed.contains(['\r', '\n']) {
            normalized_owned = trimmed
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            &normalized_owned
        } else {
            trimmed
        };
        if clean.is_empty() {
            return;
        }
        let prev_text = self.input_text.clone();
        let prev_cursor = self.selection_range.1.min(prev_text.chars().count());
        let (lo, hi) = self.normalized_selection();
        let b_lo = char_to_byte_idx(&self.input_text, lo);
        let b_hi = char_to_byte_idx(&self.input_text, hi);
        self.input_text.replace_range(b_lo..b_hi, clean);
        let new_caret = lo + clean.chars().count();
        if self.input_text != prev_text {
            self.undo_stack.push(UndoEntry {
                text: prev_text,
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
        }
        self.selection_range = (new_caret, new_caret);
        self.request_selection = Some((new_caret, new_caret));
        self.last_undo_snapshot = UndoEntry {
            text: self.input_text.clone(),
            cursor_char: new_caret,
        };
        self.request_focus = true;
        self.trigger_input_wall();
    }

    /// Ctrl+Z: Limitless undo across all input edits.
    pub fn action_undo(&mut self) -> bool {
        if let Some(prev) = self.undo_stack.pop() {
            let cur_cursor = self.selection_range.1.min(self.input_text.chars().count());
            self.redo_stack.push(UndoEntry {
                text: self.input_text.clone(),
                cursor_char: cur_cursor,
            });
            self.input_text = prev.text.clone();
            let target_cursor = prev.cursor_char.min(self.input_text.chars().count());
            self.selection_range = (target_cursor, target_cursor);
            self.request_selection = Some((target_cursor, target_cursor));
            self.last_undo_snapshot = UndoEntry {
                text: self.input_text.clone(),
                cursor_char: target_cursor,
            };
            self.request_focus = true;
            self.trigger_input_wall();
            true
        } else {
            false
        }
    }

    /// Ctrl+Y (or Ctrl+Shift+Z): Limitless redo across all undone edits.
    pub fn action_redo(&mut self) -> bool {
        if let Some(next) = self.redo_stack.pop() {
            let cur_cursor = self.selection_range.1.min(self.input_text.chars().count());
            self.undo_stack.push(UndoEntry {
                text: self.input_text.clone(),
                cursor_char: cur_cursor,
            });
            self.input_text = next.text.clone();
            let target_cursor = next.cursor_char.min(self.input_text.chars().count());
            self.selection_range = (target_cursor, target_cursor);
            self.request_selection = Some((target_cursor, target_cursor));
            self.last_undo_snapshot = UndoEntry {
                text: self.input_text.clone(),
                cursor_char: target_cursor,
            };
            self.request_focus = true;
            self.trigger_input_wall();
            true
        } else {
            false
        }
    }

    pub fn current_calc_value(&self) -> CalcValue {
        if let Some(ref res) = self.live_result {
            res.value.clone()
        } else if let Some((m, s)) = crate::engine::bignum::parse_exact_decimal(self.input_text.trim()) {
            CalcValue::normalize_decimal(m, s)
        } else if let Ok(n) = self.input_text.trim().parse::<f64>() {
            CalcValue::from_f64(n)
        } else {
            CalcValue::FastInt(0)
        }
    }

    pub fn memory_clear(&mut self) {
        self.memory_sum = 0.0;
        self.memory_val = None;
    }

    pub fn memory_add_current(&mut self) {
        let cur = self.current_calc_value();
        let base = self
            .memory_val
            .take()
            .unwrap_or_else(|| CalcValue::from_f64(self.memory_sum));
        let next = base
            .add(&cur)
            .unwrap_or_else(|_| CalcValue::from_f64(base.to_f64() + cur.to_f64()));
        self.memory_sum = next.to_f64();
        self.memory_val = Some(next);
    }

    pub fn memory_sub_current(&mut self) {
        let cur = self.current_calc_value();
        let base = self
            .memory_val
            .take()
            .unwrap_or_else(|| CalcValue::from_f64(self.memory_sum));
        let next = base
            .sub(&cur)
            .unwrap_or_else(|_| CalcValue::from_f64(base.to_f64() - cur.to_f64()));
        self.memory_sum = next.to_f64();
        self.memory_val = Some(next);
    }

    pub fn memory_is_active(&self) -> bool {
        if let Some(ref mv) = self.memory_val {
            mv.exact_sgn() != 0
        } else {
            self.memory_sum.abs() > 1e-12
        }
    }

    pub fn memory_display_str(&self) -> String {
        if let Some(ref mv) = self.memory_val {
            if (mv.to_f64() - self.memory_sum).abs() <= 1e-9 * self.memory_sum.abs().max(1.0) {
                return mv.to_string();
            }
        }
        format_clean_float(self.memory_sum)
    }

    /// Erect an immediate Input Wall: advances the cancellation epoch when the effective expression
    /// changes (preempting any in-flight calculation in microseconds), updates typing velocity
    /// telemetry, and marks the latest input state as pending.
    #[inline]
    pub fn trigger_input_wall(&mut self) {
        let now = Instant::now();
        let delta_ms = now.duration_since(self.last_keystroke).as_secs_f32() * 1000.0;
        self.last_keystroke = now;
        self.keystroke_interval_ema = (0.65 * self.keystroke_interval_ema) + (0.35 * delta_ms.min(500.0));
        self.is_modified = true;
        self.pending_calc = true;

        let effective = effective_input_slice(&self.input_text);
        if effective.is_empty() || effective != self.last_effective_input {
            let next_epoch = self.epoch_counter.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
            self.current_epoch = next_epoch;
            CANCELLATION_EPOCH.fetch_add(1, Ordering::AcqRel);
        }
        if self.input_text.trim().is_empty() {
            self.last_computed_input.clear();
            self.last_effective_input.clear();
            self.live_result = None;
            self.cached_display = None;
            self.live_error = None;
            self.task_in_flight = false;
            self.calc_start_time = None;
        }
    }

    pub fn insert_text(&mut self, text: &str) {
        self.action_paste(text);
    }

    pub fn insert_func_template(&mut self, func: &str) {
        let prev_text = self.input_text.clone();
        let prev_cursor = self.selection_range.1.min(prev_text.chars().count());
        let (lo, hi) = self.normalized_selection();
        let b_lo = char_to_byte_idx(&self.input_text, lo);
        let b_hi = char_to_byte_idx(&self.input_text, hi);
        let new_caret = if func.ends_with('(') {
            let mut combined = String::with_capacity(func.len() + 1);
            combined.push_str(func);
            combined.push(')');
            self.input_text.replace_range(b_lo..b_hi, &combined);
            lo + func.chars().count()
        } else {
            self.input_text.replace_range(b_lo..b_hi, func);
            lo + func.chars().count()
        };
        if self.input_text != prev_text {
            self.undo_stack.push(UndoEntry {
                text: prev_text,
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
        }
        self.selection_range = (new_caret, new_caret);
        self.request_selection = Some((new_caret, new_caret));
        self.last_undo_snapshot = UndoEntry {
            text: self.input_text.clone(),
            cursor_char: new_caret,
        };
        self.request_focus = true;
        self.trigger_input_wall();
    }

    pub fn backspace(&mut self) {
        if self.input_text.is_empty() {
            return;
        }
        let prev_text = self.input_text.clone();
        let prev_cursor = self.selection_range.1.min(prev_text.chars().count());
        let (lo, hi) = self.normalized_selection();
        let new_caret = if lo < hi {
            let b_lo = char_to_byte_idx(&self.input_text, lo);
            let b_hi = char_to_byte_idx(&self.input_text, hi);
            self.input_text.replace_range(b_lo..b_hi, "");
            lo
        } else if lo > 0 {
            let b_prev = char_to_byte_idx(&self.input_text, lo - 1);
            let b_lo = char_to_byte_idx(&self.input_text, lo);
            self.input_text.replace_range(b_prev..b_lo, "");
            lo - 1
        } else {
            return;
        };
        if self.input_text != prev_text {
            self.undo_stack.push(UndoEntry {
                text: prev_text,
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
        }
        self.selection_range = (new_caret, new_caret);
        self.request_selection = Some((new_caret, new_caret));
        self.last_undo_snapshot = UndoEntry {
            text: self.input_text.clone(),
            cursor_char: new_caret,
        };
        self.request_focus = true;
        self.trigger_input_wall();
    }

    pub fn delete_forward(&mut self) {
        if self.input_text.is_empty() {
            return;
        }
        let prev_text = self.input_text.clone();
        let total = prev_text.chars().count();
        let prev_cursor = self.selection_range.1.min(total);
        let (lo, hi) = self.normalized_selection();
        let new_caret = if lo < hi {
            let b_lo = char_to_byte_idx(&self.input_text, lo);
            let b_hi = char_to_byte_idx(&self.input_text, hi);
            self.input_text.replace_range(b_lo..b_hi, "");
            lo
        } else if lo < total {
            let b_lo = char_to_byte_idx(&self.input_text, lo);
            let b_next = char_to_byte_idx(&self.input_text, lo + 1);
            self.input_text.replace_range(b_lo..b_next, "");
            lo
        } else {
            return;
        };
        if self.input_text != prev_text {
            self.undo_stack.push(UndoEntry {
                text: prev_text,
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
        }
        self.selection_range = (new_caret, new_caret);
        self.request_selection = Some((new_caret, new_caret));
        self.last_undo_snapshot = UndoEntry {
            text: self.input_text.clone(),
            cursor_char: new_caret,
        };
        self.request_focus = true;
        self.trigger_input_wall();
    }

    pub fn clear(&mut self) {
        if !self.input_text.is_empty() {
            let prev_cursor = self.selection_range.1.min(self.input_text.chars().count());
            self.undo_stack.push(UndoEntry {
                text: self.input_text.clone(),
                cursor_char: prev_cursor,
            });
            self.redo_stack.clear();
        }
        self.input_text.clear();
        self.last_computed_input.clear();
        self.last_effective_input.clear();
        self.is_modified = false;
        self.live_result = None;
        self.cached_display = None;
        self.live_error = None;
        self.request_focus = true;
        self.pending_calc = false;
        self.task_in_flight = false;
        self.calc_start_time = None;
        self.selection_range = (0, 0);
        self.request_selection = Some((0, 0));
        self.last_undo_snapshot = UndoEntry {
            text: String::new(),
            cursor_char: 0,
        };
        let next_epoch = self.epoch_counter.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        self.current_epoch = next_epoch;
        CANCELLATION_EPOCH.fetch_add(1, Ordering::AcqRel);
    }

    /// Processes the Input Wall pipeline:
    /// 1. Polls completed background worker responses matching the current active epoch.
    /// 2. Pre-decides calculation cost via the Hardware-Calibrated AST Cost Estimator:
    ///    - `InstantFastPath` (< 150 µs, <= 4,096 bits): executes immediately on the spot with 0 delay.
    ///    - `GuardedAsyncWorker` (> 150 µs or > 4,096 bits): dispatches to `LiveCalcWorker` behind a smart
    ///      velocity-aware Input Wall hold window so intermediate states during fast input spam are coalesced
    ///      before starting, and any running calculation is preempted in microseconds via `epoch_counter`.
    ///    - `ExceedsExactLimit` (> 524,288 bits): pre-decides immediately without wasting CPU or memory.
    pub fn step_input_wall(
        &mut self,
        vars: &FxHashMap<String, CalcValue>,
        live_ghost_answering: bool,
        adaptive_debounce_ms: u32,
    ) {
        while let Ok(resp) = self.response_receiver.try_recv() {
            if resp.epoch == self.current_epoch {
                self.task_in_flight = false;
                self.calc_start_time = None;
                match resp.result {
                    Ok((mut res, display)) => {
                        let trimmed = self.input_text.trim();
                        let effective = effective_input_slice(trimmed);
                        res.is_speculative = effective != trimmed;
                        self.last_ans = if display.dec_str.contains('…') {
                            format!("({})", effective)
                        } else {
                            display.dec_str.clone()
                        };
                        self.live_result = Some(res);
                        self.cached_display = Some(display);
                        self.live_error = None;
                    }
                    Err(e) => {
                        if !e.contains("Aborted") {
                            if !e.contains("Incomplete") && !e.contains("Expected") {
                                self.live_result = None;
                                self.cached_display = None;
                            }
                            self.live_error = Some(e);
                        }
                    }
                }
            }
        }

        // Support legacy single-field `editor.radix_mode = ...` assignments in tests seamlessly
        if radix_to_bit(self.radix_mode) != self.radix_mask && self.radix_mask == RADIX_DEC_BIT && self.radix_mode != RadixMode::Dec {
            self.radix_mask = radix_to_bit(self.radix_mode);
        }

        if let (Some(res), Some(disp)) = (&self.live_result, &mut self.cached_display) {
            if (disp.formatted_mask & self.radix_mask) != self.radix_mask
                || disp.radix_mode != first_active_radix(self.radix_mask)
            {
                disp.ensure_mask(&res.value, self.radix_mask);
            }
        }

        if self.input_text.trim().is_empty() {
            self.last_computed_input.clear();
            self.last_effective_input.clear();
            self.live_result = None;
            self.cached_display = None;
            self.live_error = None;
            self.pending_calc = false;
            self.task_in_flight = false;
            self.calc_start_time = None;
            return;
        }

        if !live_ghost_answering {
            return;
        }

        if self.pending_calc || self.input_text.trim() != self.last_computed_input {
            let trimmed = self.input_text.trim().to_string();
            let effective = effective_input_slice(&trimmed).to_string();
            self.last_computed_input = trimmed.clone();
            self.pending_calc = false;

            // If only trailing whitespace or trailing incomplete binary operators (+, -, *, /, ^)
            // changed while the effective expression is identical, keep the in-flight worker or
            // cached live_result intact without restarting the calculation!
            if !effective.is_empty() && effective == self.last_effective_input {
                if let Some(ref mut res) = self.live_result {
                    res.is_speculative = effective != trimmed;
                }
                if self.task_in_flight || self.live_result.is_some() || self.live_error.is_some() {
                    return;
                }
            }

            if effective != self.last_effective_input {
                if self.task_in_flight && self.epoch_counter.load(Ordering::Acquire) == self.current_epoch {
                    let next_epoch = self.epoch_counter.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
                    self.current_epoch = next_epoch;
                    CANCELLATION_EPOCH.fetch_add(1, Ordering::AcqRel);
                }
                self.last_effective_input = effective;
            }

            match classify_input_cost(&trimmed, vars) {
                InputClassification::SyntaxError(err) => {
                    self.task_in_flight = false;
                    self.calc_start_time = None;
                    if !err.contains("Incomplete") && !err.contains("Expected") {
                        self.live_result = None;
                        self.cached_display = None;
                    }
                    self.live_error = Some(err);
                }
                InputClassification::Classified(cost) => {
                    match cost.tier {
                        ExecTier::ExceedsExactLimit => {
                            let is_poly_limit = cost
                                .limit_reason
                                .as_deref()
                                .map_or(false, |r| r.contains("Polynomial"));
                            if is_poly_limit {
                                self.task_in_flight = false;
                                self.calc_start_time = None;
                                self.live_result = None;
                                self.cached_display = None;
                                self.live_error = Some(
                                    cost.limit_reason.unwrap_or_else(|| {
                                        "Expression exceeds exact calculation limit".to_string()
                                    }),
                                );
                            } else {
                                // Settled Deep-Compute Tier (GPU OpenCL NTT + Multi-Core CPU + BigIntSci):
                                // Hold at the input wall for a reasonable settle timeout (`hold_wall_ms`).
                                // If the user keeps typing (`input spam`), `rx_task.recv_timeout` coalesces and
                                // overwrites the task with 0 CPU/GPU wasted. Once the input stays the same,
                                // `LiveCalcWorker` activates `deep_mode: true` and calculates the big number!
                                let hold_wall_ms = (adaptive_debounce_ms as u64).clamp(40, 220);
                                self.task_in_flight = true;
                                self.calc_start_time = Some(Instant::now());
                                self.live_result = None;
                                self.cached_display = None;
                                self.live_error = None;
                                let _ = self.task_sender.send(CalcTask {
                                    input: trimmed,
                                    epoch: self.current_epoch,
                                    radix_mask: self.radix_mask,
                                    vars: vars.clone(),
                                    hold_wall_ms,
                                    deep_mode: true,
                                });
                            }
                        }
                        ExecTier::InstantFastPath => {
                            self.task_in_flight = false;
                            self.calc_start_time = None;
                            let mut local_vars = vars.clone();
                            match execute_with_telemetry_with_counter(
                                &trimmed,
                                &mut local_vars,
                                self.current_epoch,
                                &self.epoch_counter,
                            ) {
                                Ok(res) => {
                                    let _guard = EpochGuard::new(0);
                                    if let Some(display) = PreparedDisplay::from_value_mask(
                                        &res.value,
                                        self.radix_mask,
                                    ) {
                                        self.last_ans = if display.dec_str.contains('…') {
                                            format!("({})", effective_input_slice(&trimmed))
                                        } else {
                                            display.dec_str.clone()
                                        };
                                        self.live_result = Some(res);
                                        self.cached_display = Some(display);
                                        self.live_error = None;
                                    }
                                }
                                Err(e) => {
                                    if !e.contains("Aborted") {
                                        if !e.contains("Incomplete") && !e.contains("Expected") {
                                            self.live_result = None;
                                            self.cached_display = None;
                                        }
                                        self.live_error = Some(e);
                                    }
                                }
                            }
                        }
                        ExecTier::GuardedAsyncWorker => {
                            let is_rapid_burst = self.keystroke_interval_ema < 130.0;
                            let hold_wall_ms = if is_rapid_burst {
                                if cost.est_nanos >= 10_000_000 {
                                    (adaptive_debounce_ms as u64).clamp(25, 65)
                                } else if cost.est_nanos >= 1_000_000 {
                                    18
                                } else {
                                    8
                                }
                            } else if cost.est_nanos >= 10_000_000 {
                                12
                            } else {
                                0
                            };

                            self.task_in_flight = true;
                            self.calc_start_time = Some(Instant::now());
                            let _ = self.task_sender.send(CalcTask {
                                input: trimmed,
                                epoch: self.current_epoch,
                                radix_mask: self.radix_mask,
                                vars: vars.clone(),
                                hold_wall_ms,
                                deep_mode: true,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Uniform subpixel cell width at 12pt for every Unicode superscript and subscript glyph (`4.80pt`).
pub const SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT: f32 = 4.80;
/// Subpixel left pull at 12pt when transitioning from a non-whitespace normal character to a
/// superscript or subscript run (`0.60pt`, so a single superscript reduces total advance by
/// `(7.2408 - 4.80) + 0.60 = 3.04pt`).
pub const SUBPIXEL_SUP_SUB_BASE_PULL_12PT: f32 = 0.60;

#[inline(always)]
pub const fn is_super_or_sub_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{00B2}'
            | '\u{00B3}'
            | '\u{00B9}'
            | '\u{02E3}'
            | '\u{1D62}'..='\u{1D65}'
            | '\u{2070}'
            | '\u{2071}'
            | '\u{2074}'..='\u{207F}'
            | '\u{2080}'..='\u{208E}'
            | '\u{2090}'..='\u{209C}'
    )
}

/// Zero-recalculation monospace layout job builder (`break_anywhere: true`).
/// Horizontal superscript/subscript kerning is applied directly in subpixel coordinates via
/// [`apply_subpixel_super_sub_kerning`] inside [`cmd_layouter`].
pub fn build_kerned_mono_job(
    text: &str,
    wrap_width: f32,
    font_size: f32,
    color: egui::Color32,
) -> egui::text::LayoutJob {
    let format = egui::TextFormat {
        font_id: egui::FontId::monospace(font_size),
        color,
        ..Default::default()
    };
    let mut job = egui::text::LayoutJob {
        text: text.to_string(),
        sections: Vec::new(),
        wrap: egui::text::TextWrapping {
            max_width: wrap_width,
            break_anywhere: true,
            ..Default::default()
        },
        ..Default::default()
    };
    if !text.is_empty() {
        job.sections.push(egui::text::LayoutSection {
            leading_space: 0.0,
            byte_range: 0..text.len(),
            format,
        });
    }
    job
}

/// Direct subpixel `Galley` vertex + glyph kerning pass for Unicode superscripts and subscripts.
/// Gives every superscript and subscript glyph the exact same uniform subpixel cell width
/// (`w_sup = 4.80 * scale`), centers its rasterized glyph quad inside `[cursor_x, cursor_x + w_sup]`,
/// and pulls the first superscript/subscript after a non-whitespace base character left by
/// `base_pull = 0.60 * scale`.
pub fn apply_subpixel_super_sub_kerning(galley: &mut egui::Galley, font_size: f32) {
    if galley.rows.is_empty() {
        return;
    }
    let scale = font_size / 12.0;
    let w_sup = SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT * scale;
    let base_pull = SUBPIXEL_SUP_SUB_BASE_PULL_12PT * scale;

    let mut max_row_x = galley.rect.min.x;
    let mut min_mesh_x = galley.mesh_bounds.min.x;
    let mut max_mesh_x = galley.mesh_bounds.min.x;

    for row in &mut galley.rows {
        let row_has_sup = row.glyphs.iter().any(|g| is_super_or_sub_char(g.chr));
        if !row_has_sup {
            max_row_x = max_row_x.max(row.rect.max.x);
            for v in &row.visuals.mesh.vertices {
                min_mesh_x = min_mesh_x.min(v.pos.x);
                max_mesh_x = max_mesh_x.max(v.pos.x);
            }
            continue;
        }

        let mut cursor_x = row.glyphs.first().map_or(0.0, |g| g.pos.x);
        let mut in_sup_run = false;
        let mut prev_was_non_space_normal = false;
        let mut has_shifted = false;
        let mut v_idx = row.visuals.glyph_vertex_range.start;

        for g in &mut row.glyphs {
            if is_super_or_sub_char(g.chr) {
                if !in_sup_run {
                    if prev_was_non_space_normal {
                        cursor_x = (cursor_x - base_pull).max(0.0);
                    }
                    in_sup_run = true;
                }
                prev_was_non_space_normal = false;
                has_shifted = true;

                if !g.uv_rect.is_nothing() {
                    if v_idx + 3 < row.visuals.mesh.vertices.len() {
                        let cur_left = row.visuals.mesh.vertices[v_idx].pos.x;
                        let cur_right = row.visuals.mesh.vertices[v_idx + 1].pos.x;
                        let quad_w = cur_right - cur_left;
                        let dx_vert = (cursor_x + 0.5 * (w_sup - quad_w)) - cur_left;
                        for k in 0..4 {
                            row.visuals.mesh.vertices[v_idx + k].pos.x += dx_vert;
                        }
                    }
                    v_idx += 4;
                }

                g.pos.x = cursor_x;
                g.advance_width = w_sup;
                cursor_x += w_sup;
            } else {
                in_sup_run = false;
                prev_was_non_space_normal = !g.chr.is_whitespace();

                let dx = cursor_x - g.pos.x;
                if !g.uv_rect.is_nothing() {
                    if has_shifted && v_idx + 3 < row.visuals.mesh.vertices.len() {
                        for k in 0..4 {
                            row.visuals.mesh.vertices[v_idx + k].pos.x += dx;
                        }
                    }
                    v_idx += 4;
                }

                g.pos.x = cursor_x;
                cursor_x += g.advance_width;
            }
        }

        row.rect.max.x = row.rect.min.x.max(cursor_x);
        let mut row_mesh_min_x = row.visuals.mesh_bounds.min.x;
        let mut row_mesh_max_x = row.visuals.mesh_bounds.min.x;
        for v in &row.visuals.mesh.vertices {
            row_mesh_min_x = row_mesh_min_x.min(v.pos.x);
            row_mesh_max_x = row_mesh_max_x.max(v.pos.x);
        }
        row.visuals.mesh_bounds.min.x = row_mesh_min_x;
        row.visuals.mesh_bounds.max.x = row_mesh_max_x;

        max_row_x = max_row_x.max(row.rect.max.x);
        min_mesh_x = min_mesh_x.min(row_mesh_min_x);
        max_mesh_x = max_mesh_x.max(row_mesh_max_x);
    }

    galley.rect.max.x = max_row_x;
    galley.mesh_bounds.min.x = min_mesh_x;
    galley.mesh_bounds.max.x = max_mesh_x;
}

/// Shifts `'≡'` (`\u{2261}`) and leading result `'='` (`"= ..."`) glyph mesh vertices in `-y`
/// so their ink vertical center matches the exact cap/digit vertical center (`v-center`) of the
/// `Hack` monospace text line in DPI-calculated physical pixels.
pub fn align_equiv_glyph_vcenter(
    galley: &mut egui::Galley,
    pixels_per_point: f32,
    mono_vcenter_offset: f32,
    mono_row_height: f32,
) {
    let ppp = pixels_per_point.max(0.5);
    let inv_ppp = 1.0 / ppp;
    let mut only_equiv_rows = true;
    let align_leading_eq = galley.job.text.starts_with("= ");

    for (row_idx, row) in galley.rows.iter_mut().enumerate() {
        let has_target_op = row.glyphs.iter().enumerate().any(|(gi, g)| {
            !g.uv_rect.is_nothing()
                && (g.chr == '≡' || (align_leading_eq && row_idx == 0 && gi == 0 && g.chr == '='))
        });
        if !has_target_op {
            only_equiv_rows = false;
            continue;
        }

        // First pass: check if there are non-operator glyphs on the same row, and sample only
        // ASCII digits / uppercase letters ('0'..'9', 'A'..'Z') for the reference vertical center
        // so glyphs with descenders ('[', ']', ',', '(', ')', '/', '√', 'g', 'p', 'y') or
        // superscripts/subscripts never pull the operator v-center away from the cap/digit center.
        let mut has_other_glyph = false;
        let mut ref_top = f32::INFINITY;
        let mut ref_bot = f32::NEG_INFINITY;
        let mut v_cursor = row.visuals.glyph_vertex_range.start;
        for (gi, g) in row.glyphs.iter().enumerate() {
            if !g.uv_rect.is_nothing() {
                let is_op = g.chr == '≡' || (align_leading_eq && row_idx == 0 && gi == 0 && g.chr == '=');
                if !is_op {
                    has_other_glyph = true;
                }
                if v_cursor + 3 < row.visuals.mesh.vertices.len()
                    && (g.chr.is_ascii_digit() || g.chr.is_ascii_uppercase())
                {
                    let y0 = row.visuals.mesh.vertices[v_cursor].pos.y;
                    let y2 = row.visuals.mesh.vertices[v_cursor + 2].pos.y;
                    ref_top = ref_top.min(y0.min(y2));
                    ref_bot = ref_bot.max(y0.max(y2));
                }
                v_cursor += 4;
            }
        }

        if has_other_glyph {
            only_equiv_rows = false;
        }

        // Second pass: shift each target operator glyph's 4 vertices in y to match the exact line v-center
        // quantized to physical DPI pixels.
        let mut v_idx = row.visuals.glyph_vertex_range.start;
        for (gi, g) in row.glyphs.iter_mut().enumerate() {
            if !g.uv_rect.is_nothing() {
                let is_op = g.chr == '≡' || (align_leading_eq && row_idx == 0 && gi == 0 && g.chr == '=');
                if is_op && v_idx + 3 < row.visuals.mesh.vertices.len() {
                    let v_top_px = (row.visuals.mesh.vertices[v_idx].pos.y * ppp).round();
                    let v_bot_px = (row.visuals.mesh.vertices[v_idx + 2].pos.y * ppp).round();
                    let equiv_center_px = 0.5 * (v_top_px + v_bot_px);

                    let target_center_px = if ref_top < ref_bot {
                        0.5 * ((ref_top * ppp).round() + (ref_bot * ppp).round())
                    } else if has_other_glyph && mono_row_height > 0.0 {
                        (row.rect.min.y
                            + mono_vcenter_offset
                            + (row.rect.height() - mono_row_height).max(0.0))
                            * ppp
                    } else {
                        (row.rect.min.y + mono_vcenter_offset) * ppp
                    };

                    let raw_dy_px = target_center_px - equiv_center_px;
                    // Quantize to exact physical DPI pixels, breaking half-pixel (-0.5px) ties toward -y (top)
                    let dy_px = (raw_dy_px - 1e-3).round();
                    let dy = if dy_px != 0.0 {
                        dy_px * inv_ppp
                    } else if !has_other_glyph {
                        -inv_ppp
                    } else {
                        0.0
                    };

                    for k in 0..4 {
                        row.visuals.mesh.vertices[v_idx + k].pos.y += dy;
                    }
                    g.pos.y += dy;
                }
                v_idx += 4;
            }
        }

        // Normalize standalone '≡' prefix row height to match the monospace text row height
        if !has_other_glyph && mono_row_height > 0.0 && row.rect.height() > mono_row_height {
            row.rect.max.y = row.rect.min.y + mono_row_height;
        }
    }

    if only_equiv_rows && galley.rows.len() == 1 && mono_row_height > 0.0 && galley.rect.height() > mono_row_height {
        galley.rect.max.y = galley.rect.min.y + mono_row_height;
        galley.mesh_bounds.max.y = galley.mesh_bounds.max.y.min(galley.rect.max.y);
    }
}

thread_local! {
    static MONO_METRICS_CACHE: std::cell::Cell<[(u32, u32, f32, f32, f32); 4]> =
        const { std::cell::Cell::new([(0, 0, 0.0, 0.0, 0.0); 4]) };
}

#[inline]
fn cached_mono_metrics(ui: &Ui, font_size: f32, ppp: f32) -> (f32, f32, f32) {
    let fs_bits = font_size.to_bits();
    let ppp_bits = ppp.to_bits();
    let mut cached = None;
    MONO_METRICS_CACHE.with(|cell| {
        let slots = cell.get();
        for &(f_b, p_b, vc, base, rh) in &slots {
            if f_b == fs_bits && p_b == ppp_bits && rh > 0.0 {
                cached = Some((vc, base, rh));
                break;
            }
        }
    });
    if let Some(hit) = cached {
        return hit;
    }

    let computed = ui.fonts(|f| {
        let ref_g = f.layout_no_wrap(
            String::from("0"),
            egui::FontId::new(font_size, egui::FontFamily::Monospace),
            egui::Color32::WHITE,
        );
        if let Some(r0) = ref_g.rows.first() {
            let (vc, base) = if r0.visuals.mesh.vertices.len() >= 4 {
                let y0 = r0.visuals.mesh.vertices[0].pos.y;
                let y2 = r0.visuals.mesh.vertices[2].pos.y;
                (0.5 * (y0 + y2), y0.max(y2))
            } else {
                (0.5 * r0.rect.height(), 0.8 * r0.rect.height())
            };
            (vc, base, r0.rect.height())
        } else {
            (0.5 * font_size, 0.8 * font_size, font_size)
        }
    });

    MONO_METRICS_CACHE.with(|cell| {
        let mut slots = cell.get();
        slots.copy_within(0..3, 1);
        slots[0] = (fs_bits, ppp_bits, computed.0, computed.1, computed.2);
        cell.set(slots);
    });
    computed
}

#[inline]
fn compute_kerned_wrap_width(text: &str, wrap_width: f32, font_size: f32) -> f32 {
    if !wrap_width.is_finite() || wrap_width <= 0.0 {
        return wrap_width;
    }
    let scale = font_size / 12.0;
    let w_norm = 7.2408257 * scale;
    let w_consolas = 6.5976563 * scale;
    let w_sup = SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT * scale;
    let base_pull = SUBPIXEL_SUP_SUB_BASE_PULL_12PT * scale;

    let mut kerned_x = 0.0_f32;
    let mut unkerned_x = 0.0_f32;
    let mut in_sup_run = false;
    let mut prev_was_non_space_normal = false;
    let mut min_wrap_savings = f32::INFINITY;
    let mut any_row_wrapped = false;

    for ch in text.chars() {
        if ch == '\n' {
            if unkerned_x > wrap_width {
                min_wrap_savings = min_wrap_savings.min((unkerned_x - kerned_x).max(0.0));
                any_row_wrapped = true;
            }
            kerned_x = 0.0;
            unkerned_x = 0.0;
            in_sup_run = false;
            prev_was_non_space_normal = false;
            continue;
        }
        let (step_kerned, step_unkerned) = if is_super_or_sub_char(ch) {
            let pull = if !in_sup_run && prev_was_non_space_normal {
                base_pull
            } else {
                0.0
            };
            in_sup_run = true;
            prev_was_non_space_normal = false;
            let unkerned = if matches!(
                ch,
                '\u{02E3}'
                    | '\u{1D62}'..='\u{1D65}'
                    | '\u{2080}'..='\u{208E}'
                    | '\u{2090}'..='\u{209C}'
            ) {
                w_consolas
            } else {
                w_norm
            };
            (w_sup - pull, unkerned)
        } else {
            in_sup_run = false;
            prev_was_non_space_normal = !ch.is_whitespace();
            let w_ch = if ch == '\u{2009}' { 2.40 * scale } else { w_norm };
            (w_ch, w_ch)
        };
        if kerned_x + step_kerned > wrap_width && kerned_x > 0.0 {
            min_wrap_savings = min_wrap_savings.min((unkerned_x - kerned_x).max(0.0));
            any_row_wrapped = true;
            if is_super_or_sub_char(ch) {
                in_sup_run = true;
                prev_was_non_space_normal = false;
                kerned_x = w_sup;
                unkerned_x = step_unkerned;
            } else {
                in_sup_run = false;
                prev_was_non_space_normal = !ch.is_whitespace();
                kerned_x = step_kerned;
                unkerned_x = step_unkerned;
            }
        } else {
            kerned_x += step_kerned;
            unkerned_x += step_unkerned;
        }
    }

    if !any_row_wrapped || unkerned_x > wrap_width {
        min_wrap_savings = min_wrap_savings.min((unkerned_x - kerned_x).max(0.0));
    }

    if min_wrap_savings.is_finite() {
        wrap_width + min_wrap_savings
    } else {
        wrap_width
    }
}

pub fn cmd_layouter(
    ui: &Ui,
    text: &str,
    wrap_width: f32,
    font_size: f32,
    color: egui::Color32,
) -> std::sync::Arc<egui::Galley> {
    let has_super_or_sub = !text.is_ascii() && text.chars().any(is_super_or_sub_char);
    let has_equiv = (!text.is_ascii() && text.contains('≡')) || text.starts_with("= ");
    let effective_wrap_w = if has_super_or_sub {
        compute_kerned_wrap_width(text, wrap_width, font_size)
    } else {
        wrap_width
    };
    let job = build_kerned_mono_job(text, effective_wrap_w, font_size, color);
    if !has_super_or_sub && !has_equiv {
        return ui.fonts(|f| f.layout_job(job));
    }

    let mut galley = ui.fonts(|f| f.layout_job(job));
    if has_super_or_sub {
        let mut cur_wrap_w = effective_wrap_w;
        for attempt in 0..3 {
            let galley_mut = std::sync::Arc::make_mut(&mut galley);
            let mut row_unkerned = [0.0_f32; 16];
            for (i, r) in galley_mut.rows.iter().enumerate().take(16) {
                row_unkerned[i] = r.rect.width();
            }
            apply_subpixel_super_sub_kerning(galley_mut, font_size);
            if cur_wrap_w <= wrap_width || galley_mut.rect.width() <= wrap_width + 1e-2 || attempt == 2 {
                break;
            }
            let mut next_wrap_w = cur_wrap_w;
            for (i, r) in galley_mut.rows.iter().enumerate().take(16) {
                let kw = r.rect.width();
                if kw > wrap_width + 1e-2 {
                    let excess = kw - wrap_width;
                    let scale = font_size / 12.0;
                    let unkerned_drop = excess * (7.25 / 4.20) + 0.5 * scale;
                    let candidate = (row_unkerned[i] - unkerned_drop).max(wrap_width);
                    next_wrap_w = next_wrap_w.min(candidate);
                }
            }
            if attempt == 1 || next_wrap_w >= cur_wrap_w - 1e-3 {
                next_wrap_w = wrap_width;
            }
            cur_wrap_w = next_wrap_w;
            let retry_job = build_kerned_mono_job(text, cur_wrap_w, font_size, color);
            galley = ui.fonts(|f| f.layout_job(retry_job));
        }
    }
    if has_equiv {
        let galley_mut = std::sync::Arc::make_mut(&mut galley);
        let ppp = ui.ctx().pixels_per_point();
        let (mono_vcenter_offset, _, mono_row_height) = cached_mono_metrics(ui, font_size, ppp);
        align_equiv_glyph_vcenter(galley_mut, ppp, mono_vcenter_offset, mono_row_height);
    }
    galley
}

/// Renders an equivalence subline (`≡ <text>`) under a framebox with DPI-exact vertical centering
/// of the `≡` glyph and a full-row selectable hit-rect so dragging anywhere along the row selects
/// the expression cleanly without dead zones or prefix pollution.
fn render_equiv_subline(
    ui: &mut Ui,
    subline_text: &str,
    card_w: f32,
    palette: &PaperPalette,
    hover_hint: &str,
) {
    let wrap_w = (card_w - 40.0).max(100.0);
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.add_space(8.0);
        let prefix_galley = cmd_layouter(ui, "≡\u{2009}", 32.0, 12.0, palette.ink_secondary);
        let body_galley = cmd_layouter(ui, subline_text, wrap_w, 12.0, palette.ink_secondary);
        let prefix_w = (prefix_galley.size().x + 4.0).round();
        let row_w = (prefix_w + body_galley.size().x).max(card_w - 16.0);
        let row_h = body_galley.size().y.max(prefix_galley.size().y);
        let mut sense = egui::Sense::click_and_drag();
        sense.focusable = false;
        let (row_rect, response) = ui.allocate_exact_size(vec2(row_w, row_h), sense);
        let prefix_pos = row_rect.left_top();
        let body_pos = egui::pos2(row_rect.left() + prefix_w, row_rect.top());
        ui.painter()
            .galley(prefix_pos, prefix_galley, palette.ink_secondary);
        egui::text_selection::LabelSelectionState::label_text_selection(
            ui,
            &response,
            body_pos,
            body_galley,
            palette.ink_secondary,
            egui::Stroke::NONE,
        );
        let prefix_rect = egui::Rect::from_min_size(prefix_pos, vec2(prefix_w, row_h));
        if ui.rect_contains_pointer(prefix_rect) {
            response.on_hover_text(hover_hint);
        }
    });
}

struct StackDebugBuf {
    buf: [u8; 384],
    len: usize,
}

impl std::fmt::Write for StackDebugBuf {
    #[inline]
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let rem = self.buf.len().saturating_sub(self.len);
        if rem > 0 {
            let bytes = s.as_bytes();
            let n = bytes.len().min(rem);
            self.buf[self.len..self.len + n].copy_from_slice(&bytes[..n]);
            self.len += n;
        }
        Ok(())
    }
}

#[inline]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Zero-allocation inspection of `egui::text_selection::LabelSelectionState`.
/// Returns `Some(fnv1a_hash)` when a non-empty character selection (`primary != secondary`)
/// is active on any selectable `Label` in the window, or `None` otherwise.
#[inline]
pub fn label_selection_signature(ctx: &egui::Context) -> Option<u64> {
    use std::fmt::Write as _;
    let sel = egui::text_selection::LabelSelectionState::load(ctx);
    if !sel.has_selection() {
        return None;
    }
    let mut stack_buf = StackDebugBuf {
        buf: [0u8; 384],
        len: 0,
    };
    let _ = write!(&mut stack_buf, "{:?}", sel);
    let bytes = &stack_buf.buf[..stack_buf.len];
    const P_TAG: &[u8] = b"primary: WidgetTextCursor {";
    const S_TAG: &[u8] = b"secondary: WidgetTextCursor {";
    let p_idx = find_subslice(bytes, P_TAG)?;
    let s_idx = find_subslice(bytes, S_TAG)?;
    let p_rest = &bytes[p_idx + P_TAG.len()..];
    let s_rest = &bytes[s_idx + S_TAG.len()..];
    let p_end = p_rest.iter().position(|&b| b == b'}')?;
    let s_end = s_rest.iter().position(|&b| b == b'}')?;
    let p_slice = &p_rest[..p_end];
    let s_slice = &s_rest[..s_end];
    if p_slice == s_slice {
        return None;
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in p_slice.iter().chain(std::iter::once(&b'|')).chain(s_slice.iter()) {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Some(hash)
}

/// Returns `true` if `egui::text_selection::LabelSelectionState` currently holds a non-empty
/// character selection (`primary != secondary`) in any selectable `Label` in the window.
#[cfg(test)]
#[inline]
pub fn has_non_empty_label_selection(ctx: &egui::Context) -> bool {
    label_selection_signature(ctx).is_some()
}

#[inline]
fn clear_label_selection(ctx: &egui::Context, state: &mut EditorState) {
    let mut sel = egui::text_selection::LabelSelectionState::load(ctx);
    if sel.has_selection() {
        sel.clear_selection();
        sel.store(ctx);
    }
    state.last_label_selection_sig = None;
}

#[inline]
fn collapse_text_edit_selection(ctx: &egui::Context, input_id: egui::Id, state: &mut EditorState) {
    let caret = state.selection_range.1.min(state.input_text.chars().count());
    state.selection_range = (caret, caret);
    if let Some(mut edit_state) = egui::text_edit::TextEditState::load(ctx, input_id) {
        edit_state.cursor.set_char_range(Some(egui::text_selection::CCursorRange::one(
            egui::text::CCursor::new(caret),
        )));
        edit_state.store(ctx, input_id);
    }
}

enum HotkeyEvent {
    SelectAll,
    Copy,
    Cut,
    Paste(Option<String>),
    Undo,
    Redo,
    EnterCommit,
    EscapeClear,
    HistoryUp,
    HistoryDown,
}

pub fn render_editor(
    ui: &mut Ui,
    state: &mut EditorState,
    sidebar: &mut SidebarState,
    vars: &mut FxHashMap<String, CalcValue>,
    config: &mut AppConfig,
    palette: &PaperPalette,
) {
    let input_id = ui.make_persistent_id("calc_formula_input");

    if state.request_focus {
        ui.memory_mut(|m| m.request_focus(input_id));
        state.request_focus = false;
    }

    // Synchronize current selection/caret between TextEditState and LabelSelectionState
    if state.request_selection.is_some() || state.request_cursor_pos.is_some() {
        clear_label_selection(ui.ctx(), state);
    } else if let Some(edit_state) = egui::text_edit::TextEditState::load(ui.ctx(), input_id) {
        if let Some(char_range) = edit_state.cursor.char_range() {
            let total = state.input_text.chars().count();
            state.selection_range = (
                char_range.secondary.index.min(total),
                char_range.primary.index.min(total),
            );
        }
    }

    let current_label_sig = label_selection_signature(ui.ctx());
    if current_label_sig.is_some() && current_label_sig != state.last_label_selection_sig {
        // A label selection was newly created or updated (e.g. in sidebar at end of last frame)
        state.last_label_selection_sig = current_label_sig;
        collapse_text_edit_selection(ui.ctx(), input_id, state);
        ui.memory_mut(|m| m.surrender_focus(input_id));
    } else if current_label_sig.is_some() && state.selection_range.0 != state.selection_range.1 {
        // TextEdit acquired a non-empty selection while label selection was unchanged -> clear stale label selection
        clear_label_selection(ui.ctx(), state);
    } else if current_label_sig.is_none() {
        state.last_label_selection_sig = None;
    }

    let label_has_selection = state.last_label_selection_sig.is_some();

    // Intercept global window-active hotkeys (Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+A, Ctrl+Z, Ctrl+Y)
    // and command keys (Enter, Escape, ArrowUp, ArrowDown) BEFORE TextEdit so they operate
    // deterministically on `state.input_text` whether focused or not, while allowing `Event::Copy`
    // to pass through to `LabelSelectionState` whenever a non-empty label selection is active.
    let mut hotkey_queue: Vec<HotkeyEvent> = Vec::new();
    let mut seen_copy = false;
    let mut seen_cut = false;
    let mut seen_paste = false;
    let mut synthesize_label_copy_event = false;
    let mut routed_unfocused_text = String::new();
    let mut routed_unfocused_backspace = 0usize;
    let mut routed_unfocused_delete = 0usize;
    let has_focus_now = ui.memory(|m| m.has_focus(input_id));

    ui.input_mut(|i| {
        let has_copy_event = i.events.iter().any(|e| matches!(e, egui::Event::Copy));
        i.events.retain(|event| match event {
            egui::Event::Copy => {
                if label_has_selection {
                    true
                } else {
                    if !seen_copy {
                        seen_copy = true;
                        hotkey_queue.push(HotkeyEvent::Copy);
                    }
                    false
                }
            }
            egui::Event::Cut => {
                if !seen_cut {
                    seen_cut = true;
                    hotkey_queue.push(HotkeyEvent::Cut);
                }
                false
            }
            egui::Event::Paste(text) => {
                seen_paste = true;
                hotkey_queue.push(HotkeyEvent::Paste(Some(text.clone())));
                false
            }
            egui::Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } if (modifiers.ctrl || modifiers.command) && !modifiers.alt => match key {
                Key::C if !modifiers.shift => {
                    if label_has_selection {
                        if !has_copy_event {
                            synthesize_label_copy_event = true;
                        }
                    } else if !seen_copy {
                        seen_copy = true;
                        hotkey_queue.push(HotkeyEvent::Copy);
                    }
                    false
                }
                Key::X if !modifiers.shift => {
                    if !seen_cut {
                        seen_cut = true;
                        hotkey_queue.push(HotkeyEvent::Cut);
                    }
                    false
                }
                Key::V if !modifiers.shift => {
                    if !seen_paste {
                        seen_paste = true;
                        hotkey_queue.push(HotkeyEvent::Paste(None));
                    }
                    false
                }
                Key::A if !modifiers.shift => {
                    hotkey_queue.push(HotkeyEvent::SelectAll);
                    false
                }
                Key::Z => {
                    if modifiers.shift {
                        hotkey_queue.push(HotkeyEvent::Redo);
                    } else {
                        hotkey_queue.push(HotkeyEvent::Undo);
                    }
                    false
                }
                Key::Y if !modifiers.shift => {
                    hotkey_queue.push(HotkeyEvent::Redo);
                    false
                }
                _ => true,
            },
            egui::Event::Key {
                key: Key::Enter,
                pressed: true,
                modifiers,
                ..
            } if !modifiers.shift => {
                hotkey_queue.push(HotkeyEvent::EnterCommit);
                false
            }
            egui::Event::Key {
                key: Key::Escape,
                pressed: true,
                ..
            } => {
                hotkey_queue.push(HotkeyEvent::EscapeClear);
                false
            }
            egui::Event::Key {
                key: Key::ArrowUp,
                pressed: true,
                modifiers,
                ..
            } if !modifiers.shift && !modifiers.ctrl && !modifiers.command && !modifiers.alt => {
                hotkey_queue.push(HotkeyEvent::HistoryUp);
                false
            }
            egui::Event::Key {
                key: Key::ArrowDown,
                pressed: true,
                modifiers,
                ..
            } if !modifiers.shift && !modifiers.ctrl && !modifiers.command && !modifiers.alt => {
                hotkey_queue.push(HotkeyEvent::HistoryDown);
                false
            }
            egui::Event::Text(t) if t == "\n" || t == "\r" || t == "\r\n" => false,
            egui::Event::Text(t) if !has_focus_now && !i.modifiers.ctrl && !i.modifiers.command && !i.modifiers.alt => {
                routed_unfocused_text.push_str(t);
                false
            }
            egui::Event::Key {
                key: Key::Backspace,
                pressed: true,
                modifiers,
                ..
            } if !has_focus_now && !modifiers.ctrl && !modifiers.command && !modifiers.alt => {
                routed_unfocused_backspace += 1;
                false
            }
            egui::Event::Key {
                key: Key::Delete,
                pressed: true,
                modifiers,
                ..
            } if !has_focus_now && !modifiers.ctrl && !modifiers.command && !modifiers.alt => {
                routed_unfocused_delete += 1;
                false
            }
            _ => true,
        });
        if synthesize_label_copy_event {
            i.events.push(egui::Event::Copy);
        }
    });

    let mut enter_commit_requested = false;

    for ev in hotkey_queue {
        match ev {
            HotkeyEvent::SelectAll => {
                clear_label_selection(ui.ctx(), state);
                state.action_select_all();
                ui.ctx().request_repaint();
            }
            HotkeyEvent::Copy => {
                let copied = state.action_copy();
                if !copied.is_empty() {
                    set_os_clipboard_text(ui.ctx(), &copied);
                }
                ui.ctx().request_repaint();
            }
            HotkeyEvent::Cut => {
                clear_label_selection(ui.ctx(), state);
                let copied = state.action_cut();
                if !copied.is_empty() {
                    set_os_clipboard_text(ui.ctx(), &copied);
                }
                ui.ctx().request_repaint();
            }
            HotkeyEvent::Paste(maybe_text) => {
                clear_label_selection(ui.ctx(), state);
                let paste_str = match maybe_text {
                    Some(s) if !s.is_empty() => Some(s),
                    _ => get_os_clipboard_text(),
                };
                if let Some(s) = paste_str {
                    state.action_paste(&s);
                    ui.ctx().request_repaint();
                }
            }
            HotkeyEvent::Undo => {
                clear_label_selection(ui.ctx(), state);
                state.action_undo();
                ui.ctx().request_repaint();
            }
            HotkeyEvent::Redo => {
                clear_label_selection(ui.ctx(), state);
                state.action_redo();
                ui.ctx().request_repaint();
            }
            HotkeyEvent::EnterCommit => {
                enter_commit_requested = true;
            }
            HotkeyEvent::EscapeClear => {
                clear_label_selection(ui.ctx(), state);
                if !state.input_text.is_empty() {
                    state.clear();
                    state.request_focus = true;
                    ui.ctx().request_repaint();
                }
            }
            HotkeyEvent::HistoryUp => {
                clear_label_selection(ui.ctx(), state);
                if !sidebar.history.is_empty() {
                    let prev = state.input_text.clone();
                    let next_idx = match state.history_nav_idx {
                        None => {
                            sidebar.stashed_draft = Some(state.input_text.clone());
                            0
                        }
                        Some(idx) => (idx + 1).min(sidebar.history.len() - 1),
                    };
                    state.history_nav_idx = Some(next_idx);
                    state.input_text = sidebar.history[next_idx].input.clone();
                    state.record_external_change(prev);
                    state.trigger_input_wall();
                    state.request_focus = true;
                    ui.ctx().request_repaint();
                }
            }
            HotkeyEvent::HistoryDown => {
                clear_label_selection(ui.ctx(), state);
                if let Some(idx) = state.history_nav_idx {
                    let prev = state.input_text.clone();
                    if idx > 0 {
                        let next_idx = idx - 1;
                        state.history_nav_idx = Some(next_idx);
                        state.input_text = sidebar.history[next_idx].input.clone();
                    } else {
                        state.history_nav_idx = None;
                        state.input_text = sidebar.stashed_draft.take().unwrap_or_default();
                    }
                    state.record_external_change(prev);
                    state.trigger_input_wall();
                    state.request_focus = true;
                    ui.ctx().request_repaint();
                }
            }
        }
    }

    if routed_unfocused_backspace > 0 {
        clear_label_selection(ui.ctx(), state);
        for _ in 0..routed_unfocused_backspace {
            state.backspace();
        }
        ui.ctx().request_repaint();
    }
    if routed_unfocused_delete > 0 {
        clear_label_selection(ui.ctx(), state);
        for _ in 0..routed_unfocused_delete {
            state.delete_forward();
        }
        ui.ctx().request_repaint();
    }
    if !routed_unfocused_text.is_empty() {
        clear_label_selection(ui.ctx(), state);
        state.insert_text(&routed_unfocused_text);
        ui.ctx().request_repaint();
    }

    if state.request_focus {
        ui.memory_mut(|m| m.request_focus(input_id));
        state.request_focus = false;
    }

    // Apply any requested cursor/selection state to TextEditState BEFORE rendering TextEdit
    // so selection highlights and cursor positions take effect on the current frame.
    if let Some(pos) = state.request_cursor_pos.take() {
        state.request_selection = Some((pos, pos));
    }
    let pending_sel = state.request_selection.take();
    if let Some((sel_start, sel_end)) = pending_sel {
        clear_label_selection(ui.ctx(), state);
        let total = state.input_text.chars().count();
        let s0 = sel_start.min(total);
        let s1 = sel_end.min(total);
        state.selection_range = (s0, s1);
        let mut edit_state = egui::text_edit::TextEditState::load(ui.ctx(), input_id).unwrap_or_default();
        edit_state.cursor.set_char_range(Some(egui::text_selection::CCursorRange::two(
            egui::text::CCursor::new(s0),
            egui::text::CCursor::new(s1),
        )));
        edit_state.store(ui.ctx(), input_id);
    }

    let prev_input_sel = state.selection_range;
    let mut response = None;
    let outer_w = ui.available_width().max(320.0);
    state.last_card_rects.clear();
    if sidebar.visible {
        state.last_show_sidebar_btn_rect = None;
    }

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 6.0;

        // 1. Sleek Unified Header Ribbon: Brand · Multi-Toggle Radix Pills · High-Contrast Memory · Theme · Self-Test · AC · Sidebar Toggle
        let top_card = show_exact_card(ui, outer_w, palette, |ui, inner_w| {
            ui.allocate_ui_with_layout(
                vec2(inner_w, HEADER_ROW_HEIGHT),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 5.0;
                    ui.add(
                        Label::new(
                            RichText::new(env!("APP_DISPLAY_NAME"))
                                .strong()
                                .size(12.5)
                                .color(palette.accent),
                        )
                        .selectable(false),
                    )
                    .on_hover_text(concat!(
                        env!("APP_DISPLAY_NAME"),
                        " v",
                        env!("CARGO_PKG_VERSION"),
                        " — ",
                        env!("APP_PUBLISHER")
                    ));

                    ui_sym_vline(ui, 18.0, 7.0, palette.border);

                    // Independent Multi-Toggle Base / Radix Pills (`>= 1` active enforced)
                    for (r, bit, label, tip) in ALL_RADICES {
                        let is_active = (state.radix_mask & bit) != 0;
                        if ui_pill_toggle(ui, is_active, label, true, palette)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            state.toggle_radix(r);
                            state.request_focus = true;
                        }
                    }

                    ui_sym_vline(ui, 18.0, 7.0, palette.border);

                    // High-Contrast Memory Accumulator Pill (`badge_bg` rect + `accent` text when active — never text == rect!)
                    let mem_active = state.memory_is_active();
                    let mem_val_full = state.memory_display_str();
                    let mem_val_compact = truncate_pill_chars(&mem_val_full, 12);
                    let mem_pill_str = format!("Mem: {mem_val_compact}");
                    let mem_hover = format!(
                        "Memory Accumulator = {mem_val_full}\nClick to recall (MR) into expression"
                    );
                    if ui_pill_toggle(ui, mem_active, &mem_pill_str, true, palette)
                        .on_hover_text(mem_hover)
                        .clicked()
                    {
                        state.insert_text(&mem_val_full);
                        state.request_focus = true;
                    }

                    // Right-aligned controls: Symbol-only Sidebar Expand (`«`, pixel-identical to `»`), AC, Self-Test, Theme
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 5.0;
                        if !sidebar.visible {
                            let show_resp = ui
                                .add_sized(
                                    SIDEBAR_TOGGLE_BTN_SIZE,
                                    egui::Button::new(
                                        RichText::new("«")
                                            .color(palette.ink_secondary)
                                            .strong()
                                            .size(12.0),
                                    ),
                                )
                                .on_hover_text("Show Session History Sidebar");
                            state.last_show_sidebar_btn_rect = Some(show_resp.rect);
                            if show_resp.clicked() {
                                sidebar.visible = true;
                                state.request_focus = true;
                            }
                            ui_sym_vline(ui, 18.0, 7.0, palette.border);
                        }

                        if ui_pill_toggle(ui, true, "AC", false, palette)
                            .on_hover_text("All Clear — Clear expression editor (Esc)")
                            .clicked()
                        {
                            state.clear();
                        }

                        ui_sym_vline(ui, 18.0, 7.0, palette.border);

                        let diag_label = if state.show_diagnostics {
                            "Tests: ON"
                        } else {
                            "Self-Test"
                        };
                        if ui_pill_toggle(ui, state.show_diagnostics, diag_label, false, palette)
                            .on_hover_text(
                                "Built-in 119-Case Self-Test & Microsecond Latency Telemetry",
                            )
                            .clicked()
                        {
                            state.show_diagnostics = !state.show_diagnostics;
                            if state.show_diagnostics && state.diagnostics_report.is_none() {
                                state.diagnostics_report = Some(run_all_diagnostics());
                            }
                            state.request_focus = true;
                        }

                        ui_sym_vline(ui, 18.0, 7.0, palette.border);

                        egui::ComboBox::from_id_salt("theme_selector")
                            .selected_text(
                                RichText::new(
                                    super::paper_theme::PALETTES[config.theme_index].name,
                                )
                                .size(11.0),
                            )
                            .show_ui(ui, |ui| {
                                for (idx, p) in super::paper_theme::PALETTES.iter().enumerate() {
                                    if ui
                                        .selectable_value(&mut config.theme_index, idx, p.name)
                                        .clicked()
                                    {
                                        crate::config::update_config(|c| c.theme_index = idx);
                                        crate::config::save_ini(config);
                                        state.request_focus = true;
                                    }
                                }
                            })
                            .response
                            .on_hover_text("Color Palette Theme");
                    });
                },
            );
        });
        state.last_card_rects.push(top_card.response.rect);

        // Optional Diagnostics Report Drawer (Internal Self-Test telemetry view)
        if state.show_diagnostics {
            let diag_rect = render_diagnostics_drawer(ui, outer_w, state, palette);
            state.last_card_rects.push(diag_rect);
        }

        // 2. Modern Formula Input Bar (Integrated card with "fx" badge, extends in Y up to 120 px)
        let input_card = show_exact_card(ui, outer_w, palette, |ui, inner_w| {
            const HINT_FONT_SIZE: f32 = 11.5;
            let ppp = ui.ctx().pixels_per_point();
            let (_, mono_14_base, mono_14_rh) = cached_mono_metrics(ui, 14.0, ppp);
            let fx_galley = ui.fonts(|f| {
                f.layout_no_wrap(
                    String::from("fx"),
                    egui::FontId::new(14.0, egui::FontFamily::Monospace),
                    palette.accent,
                )
            });
            let fx_w = fx_galley.size().x.ceil();
            let input_wrap_w = (inner_w - fx_w - 10.0).max(100.0);

            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.add_space(2.0);
                let (fx_slot_rect, fx_resp) =
                    ui.allocate_exact_size(vec2(fx_w, mono_14_rh + 6.0), egui::Sense::hover());
                fx_resp.on_hover_text(
                    "Expression Formula Input — Supports arithmetic, matrices, complex numbers, logic, calculus & unit conversions",
                );
                ui.add_space(6.0);

                let mut layouter = |ui: &Ui, text: &str, wrap_width: f32| {
                    cmd_layouter(ui, text, wrap_width, 14.0, palette.ink_primary)
                };

                let mut fx_target_row_top_y = fx_slot_rect.top() + 3.0;
                let mut fx_target_x = fx_slot_rect.right() + 6.0;
                let mut fx_target_galley: Option<std::sync::Arc<egui::Galley>> = None;

                ScrollArea::vertical()
                    .id_salt("formula_input_scroll")
                    .max_height(120.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        let text_edit = TextEdit::multiline(&mut state.input_text)
                            .id(input_id)
                            .font(egui::TextStyle::Monospace)
                            .frame(false)
                            .desired_width(input_wrap_w)
                            .desired_rows(1)
                            .layouter(&mut layouter)
                            .margin(vec2(0.0, 3.0));
                        let text_out = text_edit.show(ui);
                        if text_out.galley_pos.y >= fx_slot_rect.top() {
                            fx_target_row_top_y = text_out.galley_pos.y;
                            fx_target_x = text_out.galley_pos.x;
                            if !state.input_text.is_empty() {
                                fx_target_galley = Some(text_out.galley.clone());
                            }
                        }
                        let resp = text_out.response;
                        if let Some((sel_start, sel_end)) = pending_sel {
                            let total = state.input_text.chars().count();
                            let s0 = sel_start.min(total);
                            let s1 = sel_end.min(total);
                            if let Some(mut edit_state) =
                                egui::text_edit::TextEditState::load(ui.ctx(), resp.id)
                            {
                                edit_state.cursor.set_char_range(Some(
                                    egui::text_selection::CCursorRange::two(
                                        egui::text::CCursor::new(s0),
                                        egui::text::CCursor::new(s1),
                                    ),
                                ));
                                edit_state.store(ui.ctx(), resp.id);
                            }
                        }
                        response = Some(resp);
                    });

                let fx_y = shared_kerning::align_prefix_galley_datum_y(
                    &fx_galley,
                    fx_target_row_top_y,
                    fx_target_galley.as_deref(),
                    mono_14_base,
                    ppp,
                    shared_kerning::PrefixDatum::AlphabeticBaseline,
                );
                if state.input_text.is_empty() {
                    let hint_galley = cmd_layouter(
                        ui,
                        "Enter expression (e.g. [1, 2; 3, 4] * 2, roots(x^2 - 5x + 6), p & !p)...",
                        input_wrap_w,
                        HINT_FONT_SIZE,
                        palette.ghost_ink,
                    );
                    let hint_y = shared_kerning::align_prefix_galley_datum_y(
                        &hint_galley,
                        fx_y,
                        Some(fx_galley.as_ref()),
                        mono_14_base,
                        ppp,
                        shared_kerning::PrefixDatum::AlphabeticBaseline,
                    );
                    ui.painter().galley(
                        egui::pos2(fx_target_x, hint_y),
                        hint_galley,
                        palette.ghost_ink,
                    );
                }
                ui.painter()
                    .galley(egui::pos2(fx_slot_rect.left(), fx_y), fx_galley, palette.accent);
            });
        });
        state.last_card_rects.push(input_card.response.rect);

        // Synchronize post-TextEdit cursor/selection and record any direct TextEdit mutations onto the undo stack
        if let Some(ref resp) = response {
            if let Some(edit_state) = egui::text_edit::TextEditState::load(ui.ctx(), resp.id) {
                if let Some(char_range) = edit_state.cursor.char_range() {
                    let total = state.input_text.chars().count();
                    state.selection_range = (
                        char_range.secondary.index.min(total),
                        char_range.primary.index.min(total),
                    );
                }
            }
            if resp.clicked()
                || resp.dragged()
                || resp.is_pointer_button_down_on()
                || resp.changed()
                || (state.selection_range != prev_input_sel
                    && state.selection_range.0 != state.selection_range.1)
            {
                clear_label_selection(ui.ctx(), state);
            }
            if state.input_text != state.last_undo_snapshot.text {
                state.undo_stack.push(state.last_undo_snapshot.clone());
                state.redo_stack.clear();
                let cur_char = state.selection_range.1.min(state.input_text.chars().count());
                state.last_undo_snapshot = UndoEntry {
                    text: state.input_text.clone(),
                    cursor_char: cur_char,
                };
            } else {
                state.last_undo_snapshot.cursor_char =
                    state.selection_range.1.min(state.input_text.chars().count());
            }
            if resp.changed() {
                state.trigger_input_wall();
            }
        }

        // Execute Pre-Calibrated Input Wall:
        // - Instantly evaluates light expressions (≤ 150 µs) on the fast path
        // - Preempts & coalesces heavy expressions (e.g. X**Y) on the background worker
        state.step_input_wall(vars, config.live_ghost_answering, config.adaptive_debounce_ms);

        if enter_commit_requested {
            if let (Some(res), Some(disp)) = (&state.live_result, &state.cached_display) {
                sidebar.push(
                    state.input_text.clone(),
                    disp.radix_str.clone(),
                    res.canonical_echo.clone(),
                    res.elapsed_nanos,
                );
                state.is_modified = false;
                sidebar.stashed_draft = None;
                state.history_nav_idx = None;
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
        }

        ScrollArea::vertical()
            .id_salt("editor_body_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.set_min_width(outer_w);
                ui.set_max_width(outer_w);
                ui.spacing_mut().item_spacing.y = 6.0;

                // Background Watchdog Indicator & Live Progress
                if state.task_in_flight {
                    let elapsed_ms = state.calc_start_time.map_or(0, |t| t.elapsed().as_millis());
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        let spinner = match (elapsed_ms / 200) % 4 {
                            0 => "◐",
                            1 => "◓",
                            2 => "◑",
                            _ => "◒",
                        };
                        ui.label(
                            RichText::new(format!(
                                "{spinner} Calculating in background ({elapsed_ms} ms elapsed)..."
                            ))
                            .color(palette.accent)
                            .monospace()
                            .strong()
                            .size(12.0),
                        );
                        ui_sym_vline(ui, 16.0, 6.0, palette.border);
                        let pred = predict_algo_complexity(&state.last_computed_input);
                        ui.label(RichText::new(pred).color(palette.ink_secondary).size(11.0));
                    });
                    ui.ctx().request_repaint();
                }

                // 3. Canonical Form Subline under Input Framebox (≡ [canonical_echo])
                if config.show_canonical_echo {
                    if let Some(ref res) = state.live_result {
                        let trimmed = state.input_text.trim();
                        if !res.canonical_echo.is_empty() && res.canonical_echo != trimmed {
                            render_equiv_subline(
                                ui,
                                &res.canonical_echo,
                                (outer_w - 8.0).max(240.0),
                                palette,
                                "Canonical Form (Parsed Expression)",
                            );
                        }
                    }
                }

                // 4. Hero Result Deck + Ordered Multi-Radix Displays (`DEC -> HEX -> OCT -> BIN`)
                let mut trigger_copy: Option<(usize, String)> = None;
                let mut add_to_mem = false;
                let (active_rads, active_len) = state.active_radices();

                if let (Some(res), Some(disp)) = (&state.live_result, &state.cached_display) {
                    let ink = if res.is_speculative || state.task_in_flight {
                        palette.ghost_ink
                    } else {
                        palette.accent
                    };
                    let primary_rad = active_rads[0];
                    let primary_slot = radix_to_slot(primary_rad);
                    let notice_live = state
                        .copy_notice_timer
                        .is_some_and(|t| t.elapsed().as_millis() < 1500);
                    if notice_live {
                        ui.ctx().request_repaint_after(Duration::from_millis(100));
                    }

                    let hero_card = show_exact_card(ui, outer_w, palette, |ui, inner_w| {
                        // Top Header of the Hero Result Card: Type badge, Latency badge, Copy and Memory actions
                        ui.horizontal(|ui| {
                            match &res.value {
                                CalcValue::Matrix { rows, cols, .. } => {
                                    ui.label(
                                        RichText::new(format!("Matrix {rows}×{cols}"))
                                            .color(palette.badge_text)
                                            .strong()
                                            .size(11.0),
                                    );
                                }
                                CalcValue::Logic {
                                    var_count,
                                    tautology,
                                    contradiction,
                                    ..
                                } => {
                                    let desc = if *tautology {
                                        "Logic · Tautology (Always TRUE)"
                                    } else if *contradiction {
                                        "Logic · Contradiction (Always FALSE)"
                                    } else {
                                        "Logic · Contingent"
                                    };
                                    ui.label(
                                        RichText::new(format!("{desc} ({var_count} vars)"))
                                            .color(palette.badge_text)
                                            .strong()
                                            .size(11.0),
                                    );
                                }
                                _ => {
                                    let type_name = radix_result_type_label(&res.value, primary_rad);
                                    ui.label(
                                        RichText::new(type_name)
                                            .color(palette.badge_text)
                                            .size(11.0),
                                    );
                                }
                            }

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.spacing_mut().item_spacing.x = 5.0;
                                let latency = format_nanos(res.elapsed_nanos);
                                ui.label(
                                    RichText::new(format!("⚡ {latency}"))
                                        .color(palette.badge_text)
                                        .monospace()
                                        .size(11.0),
                                )
                                .on_hover_text("Exact Engine Execution Time (Excludes UI Layout)");

                                ui_sym_vline(ui, 16.0, 7.0, palette.border);

                                let copy_active = notice_live && state.copy_notice_slot == primary_slot;
                                let copy_label = if copy_active { "✓ Copied" } else { "📋 Copy" };
                                if ui
                                    .push_id("copy_result_btn_primary", |ui| {
                                        ui.button(RichText::new(copy_label).size(10.5))
                                            .on_hover_text("Copy Result Value To Clipboard")
                                            .clicked()
                                    })
                                    .inner
                                {
                                    trigger_copy = Some((
                                        primary_slot,
                                        disp.multiline_for_radix(primary_rad).to_owned(),
                                    ));
                                    state.request_focus = true;
                                }

                                if ui
                                    .button(
                                        RichText::new("+ M")
                                            .color(palette.ink_primary)
                                            .strong()
                                            .size(10.5),
                                    )
                                    .on_hover_text(
                                        "Memory Add (M+) — Add current result to memory accumulator",
                                    )
                                    .clicked()
                                {
                                    add_to_mem = true;
                                    state.request_focus = true;
                                }
                            });
                        });

                        ui.add_space(4.0);

                        // Body Display: Unbounded height showing the whole answer text directly
                        render_result_body_for_radix(ui, inner_w, &res.value, disp, primary_rad, ink, palette, true);
                    });
                    state.last_card_rects.push(hero_card.response.rect);

                    // 4a. Additional Active Radices (`2nd..4th` in `DEC -> HEX -> OCT -> BIN` order, Copy-only)
                    if active_len >= 2 && !matches!(&res.value, CalcValue::Matrix { .. }) {
                        for &sec_rad in &active_rads[1..active_len] {
                            let sec_slot = radix_to_slot(sec_rad);
                            let sec_str = disp.str_for_radix(sec_rad);
                            if sec_str.is_empty() {
                                continue;
                            }
                            let sec_card = show_exact_card(ui, outer_w, palette, |ui, inner_w| {
                                ui.horizontal(|ui| {
                                    let sec_title = match sec_rad {
                                        RadixMode::Dec => "Decimal (Base 10)",
                                        RadixMode::Hex => "Hexadecimal (Base 16)",
                                        RadixMode::Oct => "Octal (Base 8)",
                                        RadixMode::Bin => "Binary (Base 2)",
                                    };
                                    ui.label(
                                        RichText::new(sec_title)
                                            .color(palette.badge_text)
                                            .strong()
                                            .size(10.5),
                                    );

                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let copy_active =
                                                notice_live && state.copy_notice_slot == sec_slot;
                                            let copy_label =
                                                if copy_active { "✓ Copied" } else { "📋 Copy" };
                                            if ui
                                                .push_id(("copy_sec_radix", sec_slot), |ui| {
                                                    ui.button(RichText::new(copy_label).size(10.0))
                                                        .on_hover_text(format!(
                                                            "Copy {sec_title} Representation To Clipboard"
                                                        ))
                                                        .clicked()
                                                })
                                                .inner
                                            {
                                                trigger_copy = Some((
                                                    sec_slot,
                                                    disp.multiline_for_radix(sec_rad).to_owned(),
                                                ));
                                                state.request_focus = true;
                                            }
                                        },
                                    );
                                });
                                ui.add_space(2.0);
                                render_result_body_for_radix(
                                    ui,
                                    inner_w,
                                    &res.value,
                                    disp,
                                    sec_rad,
                                    ink,
                                    palette,
                                    false,
                                );
                            });
                            state.last_card_rects.push(sec_card.response.rect);
                        }
                    }

                    // 4b. Single Equivalence Subline under Answer Framebox (Factors or Exact Integer Ratio)
                    if let Some(subline) = disp.answer_equiv_subline() {
                        render_equiv_subline(
                            ui,
                            &subline,
                            (outer_w - 8.0).max(240.0),
                            palette,
                            "Equivalent Exact Form (Prime Factorization / Integer Ratio)",
                        );
                    }
                } else if let Some(ref err) = state.live_error {
                    if !state.input_text.trim().is_empty()
                        && !err.contains("Incomplete")
                        && !err.contains("Expected")
                        && !err.contains("Aborted")
                    {
                        ui.label(
                            RichText::new(format!("Notice: {err}"))
                                .color(palette.ghost_ink)
                                .size(11.0),
                        );
                    }
                }

                // Apply deferred actions once immutable borrow of state.live_result ends
                if let Some((slot, text)) = trigger_copy {
                    set_os_clipboard_text(ui.ctx(), &text);
                    state.copy_notice_slot = slot;
                    state.copy_notice_timer = Some(Instant::now());
                }
                if add_to_mem {
                    state.memory_add_current();
                }

                // 5. Categorized Possibilities Deck with Universal Action Row
                let deck_rect = render_keypad_grid(ui, outer_w, state, palette);
                state.last_card_rects.push(deck_rect);
            });
    });

    // If a selectable Label inside `render_editor` just acquired or updated a non-empty selection
    // during this frame, immediately collapse any TextEdit selection and surrender focus so the
    // two selection systems never compete.
    let post_label_sig = label_selection_signature(ui.ctx());
    if post_label_sig.is_some() && post_label_sig != state.last_label_selection_sig {
        state.last_label_selection_sig = post_label_sig;
        collapse_text_edit_selection(ui.ctx(), input_id, state);
        ui.memory_mut(|m| m.surrender_focus(input_id));
    } else if post_label_sig.is_none() {
        state.last_label_selection_sig = None;
    }

    if state.request_focus {
        clear_label_selection(ui.ctx(), state);
        ui.memory_mut(|m| m.request_focus(input_id));
        state.request_focus = false;
    } else if response
        .as_ref()
        .is_some_and(|r| r.lost_focus() || r.clicked_elsewhere())
    {
        collapse_text_edit_selection(ui.ctx(), input_id, state);
    }
}

#[inline]
fn truncate_pill_chars(s: &str, max_chars: usize) -> String {
    let mut it = s.char_indices();
    if let Some((byte_idx, _)) = it.nth(max_chars) {
        format!("{}…", &s[..byte_idx])
    } else {
        s.to_owned()
    }
}

fn radix_result_type_label(val: &CalcValue, r: RadixMode) -> &'static str {
    match r {
        RadixMode::Dec => match val {
            CalcValue::ExactDecimal { .. } => "Exact Decimal (Banking Tier)",
            CalcValue::FastInt(_) => "Integer Result (DEC)",
            CalcValue::BigIntSci { .. } => "Astronomical Integer (GPU/NTT Scientific)",
            CalcValue::HeapBigInt { limbs, .. } if limbs.len() >= 512 => {
                "Arbitrary-Precision Integer (GPU/NTT Accelerated)"
            }
            CalcValue::InlineBigInt { .. } | CalcValue::HeapBigInt { .. } => {
                "Arbitrary-Precision Integer"
            }
            CalcValue::Factored { .. } => "Prime Factorization",
            CalcValue::Complex { .. } => "Complex Number (a + bi)",
            CalcValue::Polynomial { .. } => "Symbolic Polynomial",
            _ => "Decimal Result",
        },
        RadixMode::Hex => "Hexadecimal (Base 16)",
        RadixMode::Oct => "Octal (Base 8)",
        RadixMode::Bin => "Binary (Base 2)",
    }
}

fn render_result_body_for_radix(
    ui: &mut Ui,
    inner_w: f32,
    val: &CalcValue,
    disp: &PreparedDisplay,
    radix: RadixMode,
    ink: egui::Color32,
    palette: &PaperPalette,
    is_hero: bool,
) {
    match val {
        CalcValue::Matrix { .. } | CalcValue::Logic { .. }
            if is_hero || !matches!(radix, RadixMode::Bin) =>
        {
            let multiline_text = disp.multiline_for_radix(radix);
            let res_wrap_w = (inner_w - 8.0).max(100.0);
            let galley = cmd_layouter(ui, multiline_text, res_wrap_w, 14.0, ink);
            let row_w = galley.size().x.max(res_wrap_w);
            let row_h = galley.size().y;
            let mut sense = egui::Sense::click_and_drag();
            sense.focusable = false;
            let (row_rect, resp_ans) = ui.allocate_exact_size(vec2(row_w, row_h), sense);
            egui::text_selection::LabelSelectionState::label_text_selection(
                ui,
                &resp_ans,
                row_rect.left_top(),
                galley,
                ink,
                egui::Stroke::NONE,
            );
            resp_ans.on_hover_text("Drag to select and press Ctrl+C to copy");
        }
        _ => {
            let formatted_full = disp.str_for_radix(radix);
            let font_size = if !is_hero {
                if formatted_full.len() <= 48 { 16.0 } else { 13.5 }
            } else if formatted_full.len() <= 24 {
                26.0
            } else if formatted_full.len() <= 80 {
                18.0
            } else {
                14.0
            };

            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let ppp = ui.ctx().pixels_per_point();
                let (mono_vc, _, mono_rh) = cached_mono_metrics(ui, font_size, ppp);
                let eq_font_size = (font_size * 0.92_f32).round();
                let eq_galley = ui.fonts(|f| {
                    f.layout_no_wrap(
                        String::from("="),
                        egui::FontId::new(eq_font_size, egui::FontFamily::Proportional),
                        palette.ink_secondary,
                    )
                });
                let eq_w = (eq_galley.size().x + 8.0).round();
                let res_wrap_w = (inner_w - eq_w - 4.0).max(100.0);
                let galley = cmd_layouter(ui, formatted_full, res_wrap_w, font_size, ink);
                let row_w = (eq_w + galley.size().x).max(inner_w);
                let row_h = galley.size().y.max(mono_rh);
                let mut sense = egui::Sense::click_and_drag();
                sense.focusable = false;
                let (row_rect, resp_ans) = ui.allocate_exact_size(vec2(row_w, row_h), sense);
                let eq_y = shared_kerning::align_prefix_galley_datum_y(
                    &eq_galley,
                    row_rect.top(),
                    Some(galley.as_ref()),
                    mono_vc,
                    ppp,
                    shared_kerning::PrefixDatum::CapDigitMidline,
                );
                let eq_pos = egui::pos2(row_rect.left(), eq_y);
                let body_pos = egui::pos2(row_rect.left() + eq_w, row_rect.top());
                ui.painter().galley(eq_pos, eq_galley, palette.ink_secondary);
                egui::text_selection::LabelSelectionState::label_text_selection(
                    ui,
                    &resp_ans,
                    body_pos,
                    galley,
                    ink,
                    egui::Stroke::NONE,
                );
                resp_ans.on_hover_text("Drag to select and press Ctrl+C to copy");
            });
        }
    }
}

/// Renders the interactive diagnostics drawer for ground-truth performance testing within the GUI
fn render_diagnostics_drawer(
    ui: &mut Ui,
    outer_w: f32,
    state: &mut EditorState,
    palette: &PaperPalette,
) -> egui::Rect {
    show_exact_card(ui, outer_w, palette, |ui, _inner_w| {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Continuum Self-Test & Diagnostic Telemetry")
                    .color(palette.accent)
                    .strong()
                    .size(12.0),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(RichText::new("Rerun All Tests").size(10.5))
                    .on_hover_text("Re-execute all 119 verification cases & refresh latency percentiles")
                    .clicked()
                {
                    state.diagnostics_report = Some(run_all_diagnostics());
                }
            });
        });

        ui.add_space(2.0);

        let mut clicked_expr: Option<String> = None;

        if let Some(ref rep) = state.diagnostics_report {
            let pct = if rep.results.is_empty() {
                100.0
            } else {
                (rep.total_passed as f64 / rep.results.len() as f64) * 100.0
            };
            let pass_color = if rep.total_failed == 0 {
                palette.accent
            } else {
                egui::Color32::from_rgb(220, 50, 50)
            };

            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{}/{} Passed ({pct:.1}%)",
                        rep.total_passed,
                        rep.results.len()
                    ))
                    .color(pass_color)
                    .strong()
                    .size(11.0),
                );
                ui_sym_vline(ui, 14.0, 5.0, palette.border);
                ui.label(
                    RichText::new(format!("Total: {:.2} ms", rep.total_duration_ms))
                        .color(palette.ink_secondary)
                        .size(10.5),
                );
                ui_sym_vline(ui, 14.0, 5.0, palette.border);
                ui.label(
                    RichText::new(format!("p50: {}", format_nanos(rep.p50_nanos)))
                        .color(palette.badge_text)
                        .monospace()
                        .size(10.5),
                );
                ui_sym_vline(ui, 14.0, 5.0, palette.border);
                ui.label(
                    RichText::new(format!("p99: {}", format_nanos(rep.p99_nanos)))
                        .color(palette.badge_text)
                        .monospace()
                        .size(10.5),
                );
                ui_sym_vline(ui, 14.0, 5.0, palette.border);
                ui.label(
                    RichText::new(format!(
                        "{}-bit SIMD · {}×{} L1 Tile",
                        crate::engine::wisdom::cpu_simd_width_bits(),
                        crate::engine::wisdom::l1_matrix_block_dim(),
                        crate::engine::wisdom::l1_matrix_block_dim()
                    ))
                    .color(palette.ink_secondary)
                    .monospace()
                    .size(10.0),
                );
            });

            ui.add_space(4.0);

            ScrollArea::vertical()
                .id_salt("diag_results_scroll")
                .max_height(140.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    for r in &rep.results {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 8.0;
                            let status_badge = if r.passed { "PASS" } else { "FAIL" };
                            let status_color = if r.passed {
                                palette.accent
                            } else {
                                egui::Color32::from_rgb(220, 50, 50)
                            };
                            ui.allocate_ui_with_layout(
                                vec2(36.0, 18.0),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(status_badge)
                                            .color(status_color)
                                            .strong()
                                            .monospace()
                                            .size(10.0),
                                    );
                                },
                            );
                            ui.allocate_ui_with_layout(
                                vec2(114.0, 18.0),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(format!("[{}]", r.category))
                                            .color(palette.ink_secondary)
                                            .monospace()
                                            .size(10.0),
                                    );
                                },
                            );

                            let expr_btn = ui.button(
                                RichText::new(r.expression)
                                    .color(palette.ink_primary)
                                    .monospace()
                                    .size(10.0),
                            );
                            if expr_btn
                                .on_hover_text("Load Diagnostic Expression Into Editor")
                                .clicked()
                            {
                                clicked_expr = Some(r.expression.to_string());
                            }

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                // Leave a symmetric right margin (`14.0 pt`) before the floating scrollbar when expanded/opened
                                ui.add_space(14.0);
                                ui.label(
                                    RichText::new(format_nanos(r.elapsed_nanos))
                                        .color(palette.badge_text)
                                        .monospace()
                                        .size(10.0),
                                );
                            });
                        });
                    }
                });
        }

        if let Some(expr) = clicked_expr {
            let prev = state.input_text.clone();
            state.input_text = expr;
            state.record_external_change(prev);
            state.trigger_input_wall();
            state.request_focus = true;
            ui.ctx().request_repaint();
        }
    })
    .response
    .rect
}

const UNIFORM_GRID_BTN_GAP: f32 = 5.0;

/// Renders the categorized possibilities deck with progressive disclosure and universal utility row
fn render_keypad_grid(
    ui: &mut Ui,
    outer_w: f32,
    state: &mut EditorState,
    palette: &PaperPalette,
) -> egui::Rect {
    show_exact_card(ui, outer_w, palette, |ui, inner_w| {
        ui.spacing_mut().item_spacing = vec2(UNIFORM_GRID_BTN_GAP, UNIFORM_GRID_BTN_GAP);

        // Deck Header with Category Tabs and Collapse/Expand Toggle:
        // `"▾ Deck"` / `"▸ Deck"` uses the exact same `ui_pill_toggle` 22pt rounded-rectangle
        // dimensions and alignment as its neighbor pills, locking header height identically in folded & unfolded states!
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(UNIFORM_GRID_BTN_GAP, UNIFORM_GRID_BTN_GAP);
            let toggle_label = if state.show_function_deck {
                "▾ Deck"
            } else {
                "▸ Deck"
            };
            let toggle_tip = if state.show_function_deck {
                "Collapse Possibilities Deck (Unlocks Compact Window Height)"
            } else {
                "Expand Possibilities Deck (Show Categorized Function Keypad)"
            };
            if ui_pill_toggle(ui, state.show_function_deck, toggle_label, false, palette)
                .on_hover_text(toggle_tip)
                .clicked()
            {
                state.show_function_deck = !state.show_function_deck;
                state.request_focus = true;
            }

            if state.show_function_deck {
                ui_sym_vline(ui, 18.0, 7.0, palette.border);
                let categories = [
                    DeckCategory::MathTrig,
                    DeckCategory::Logic,
                    DeckCategory::LinearAlgebra,
                    DeckCategory::NumberTheory,
                    DeckCategory::Programmer,
                    DeckCategory::StatsUnits,
                ];
                for cat in categories {
                    let is_active = state.deck_category == cat;
                    if ui_pill_toggle(ui, is_active, cat.label(), false, palette)
                        .on_hover_text(cat.tooltip())
                        .clicked()
                    {
                        state.deck_category = cat;
                        state.request_focus = true;
                    }
                }
            } else {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new("Direct Keyboard Typing Active")
                            .color(palette.ghost_ink)
                            .size(10.5),
                    );
                });
            }
        });

        let sp_x = UNIFORM_GRID_BTN_GAP;

        if state.show_function_deck {
            ui_sym_hline(ui, inner_w, 3.0, palette.border);
            let btn_h = 32.0;
            let col_w = ((inner_w - 3.0 * sp_x) * 0.25).max(40.0);

            match state.deck_category {
                DeckCategory::MathTrig => {
                    let rows: [[(&str, &str); 4]; 5] = [
                        [("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("sqrt", "sqrt(")],
                        [
                            ("asin", "asin("),
                            ("acos", "acos("),
                            ("atan", "atan("),
                            ("cbrt", "cbrt("),
                        ],
                        [
                            ("sinh", "sinh("),
                            ("cosh", "cosh("),
                            ("tanh", "tanh("),
                            ("hypot", "hypot("),
                        ],
                        [("ln", "ln("), ("log10", "log10("), ("abs", "abs("), ("n!", "!")],
                        [("pi", "pi"), ("tau", "tau"), ("phi", "phi"), ("e", "e")],
                    ];
                    for row in &rows {
                        render_4col_row(ui, row, col_w, btn_h, palette.ink_primary, state);
                    }
                }
                DeckCategory::Logic => {
                    let rows: [[(&str, &str); 4]; 3] = [
                        [
                            ("AND (&)", " & "),
                            ("OR (|)", " | "),
                            ("NOT (!)", "!"),
                            ("XOR (^)", " ^ "),
                        ],
                        [
                            ("-> (Implies)", " -> "),
                            ("<-> (Iff)", " <-> "),
                            ("p", "p"),
                            ("q", "q"),
                        ],
                        [("r", "r"), ("s", "s"), ("TRUE", "true"), ("FALSE", "false")],
                    ];
                    for row in &rows {
                        render_4col_row(ui, row, col_w, btn_h, palette.ink_primary, state);
                    }
                }
                DeckCategory::LinearAlgebra => {
                    let rows: [[(&str, &str); 4]; 3] = [
                        [
                            ("det(M)", "det("),
                            ("inv(M)", "inv("),
                            ("transpose(M)", "transpose("),
                            ("trace(M)", "trace("),
                        ],
                        [
                            ("rank(M)", "rank("),
                            ("eig(M)", "eig("),
                            ("solve(A,b)", "solve("),
                            ("norm(M)", "norm("),
                        ],
                        [
                            ("eye(N)", "eye("),
                            ("[ (Open)", "["),
                            ("] (Close)", "]"),
                            ("; (Row Sep)", "; "),
                        ],
                    ];
                    for row in &rows {
                        render_4col_row(ui, row, col_w, btn_h, palette.ink_primary, state);
                    }
                }
                DeckCategory::NumberTheory => {
                    let rows: [[(&str, &str); 4]; 4] = [
                        [
                            ("primes(n)", "primes("),
                            ("fib(n)", "fib("),
                            ("fac(n)", "fac("),
                            ("gcd(a, b)", "gcd("),
                        ],
                        [
                            ("lcm(a, b)", "lcm("),
                            ("nCr(n, r)", "nCr("),
                            ("nPr(n, r)", "nPr("),
                            ("% (Mod)", " mod "),
                        ],
                        [
                            ("diff(f)", "diff("),
                            ("int(f,a,b)", "integrate("),
                            ("roots(p)", "roots("),
                            ("i (Imag)", "i"),
                        ],
                        [
                            ("root(n, x)", "root("),
                            ("log(b, x)", "log("),
                            ("sgn(x)", "sgn("),
                            ("deg(rad)", "deg("),
                        ],
                    ];
                    for row in &rows {
                        render_4col_row(ui, row, col_w, btn_h, palette.ink_primary, state);
                    }
                }
                DeckCategory::Programmer => {
                    let rows: [[(&str, &str); 4]; 3] = [
                        [
                            ("band(a, b)", "band("),
                            ("bor(a, b)", "bor("),
                            ("bxor(a, b)", "bxor("),
                            ("bnot(a)", "bnot("),
                        ],
                        [
                            ("shl(a, n)", "shl("),
                            ("shr(a, n)", "shr("),
                            ("popcnt(a)", "popcnt("),
                            ("clz(a)", "clz("),
                        ],
                        [
                            ("rol(a, n)", "rol("),
                            ("ror(a, n)", "ror("),
                            ("0x (Hex)", "0x"),
                            ("0b (Bin)", "0b"),
                        ],
                    ];
                    for row in &rows {
                        render_4col_row(ui, row, col_w, btn_h, palette.ink_primary, state);
                    }
                }
                DeckCategory::StatsUnits => {
                    let rows: [[(&str, &str); 4]; 4] = [
                        [
                            ("min(..)", "min("),
                            ("max(..)", "max("),
                            ("sum(..)", "sum("),
                            ("mean(..)", "mean("),
                        ],
                        [
                            ("median(..)", "median("),
                            ("std(..)", "std("),
                            ("var(..)", "var("),
                            ("convert(..)", "convert("),
                        ],
                        [
                            ("km to m", "10 km to m"),
                            ("c to f", "100 c to f"),
                            ("lb to kg", "10 lb to kg"),
                            ("gb to mb", "1 gb to mb"),
                        ],
                        [
                            ("kwh to j", "1 kwh to j"),
                            ("hp to w", "1 hp to w"),
                            ("psi to bar", "100 psi to bar"),
                            ("mps to kph", "10 mps to kph"),
                        ],
                    ];
                    for row in &rows {
                        render_4col_row(ui, row, col_w, btn_h, palette.ink_primary, state);
                    }
                }
            }
        }

        // Identical pixel-snapped horizontal separator & vertical margin in BOTH folded & unfolded states!
        ui_sym_hline(ui, inner_w, 3.0, palette.border);

        // Dedicated Tactile Universal Quick Utility Action Bar (`MC .. AC`, pixel-flush to `inner_w`)
        let col_w = ((inner_w - 6.0 * sp_x) / 7.0).max(28.0);
        let btn_h = 28.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = UNIFORM_GRID_BTN_GAP;
            if ui
                .add_sized([col_w, btn_h], egui::Button::new(RichText::new("MC").size(11.0)))
                .on_hover_text("Memory Clear (MC) — Reset memory accumulator to 0")
                .clicked()
            {
                state.memory_clear();
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
            if ui
                .add_sized([col_w, btn_h], egui::Button::new(RichText::new("MR").size(11.0)))
                .on_hover_text("Memory Recall (MR) — Insert stored memory value into expression")
                .clicked()
            {
                let s = state.memory_display_str();
                state.insert_text(&s);
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
            if ui
                .add_sized(
                    [col_w, btn_h],
                    egui::Button::new(RichText::new("M+").color(palette.accent).size(11.0).strong()),
                )
                .on_hover_text("Memory Add (M+) — Add current value to memory accumulator")
                .clicked()
            {
                state.memory_add_current();
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
            if ui
                .add_sized(
                    [col_w, btn_h],
                    egui::Button::new(RichText::new("M-").color(palette.accent).size(11.0).strong()),
                )
                .on_hover_text("Memory Subtract (M-) — Subtract current value from memory accumulator")
                .clicked()
            {
                state.memory_sub_current();
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
            if ui
                .add_sized(
                    [col_w, btn_h],
                    egui::Button::new(RichText::new("ANS").color(palette.ink_primary).size(11.0)),
                )
                .on_hover_text("Previous Answer (ANS) — Insert latest result into expression")
                .clicked()
            {
                let ans = state.last_ans.clone();
                state.insert_text(&ans);
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
            if ui
                .add_sized(
                    [col_w, btn_h],
                    egui::Button::new(RichText::new("DEL").color(palette.ink_primary).size(11.0)),
                )
                .on_hover_text("Delete / Backspace (DEL) — Delete selected characters or previous character")
                .clicked()
            {
                state.backspace();
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
            if ui
                .add_sized(
                    [col_w, btn_h],
                    egui::Button::new(RichText::new("AC").color(palette.accent).size(11.0).strong()),
                )
                .on_hover_text("All Clear (AC) — Clear expression editor (Esc)")
                .clicked()
            {
                state.clear();
                state.request_focus = true;
                ui.ctx().request_repaint();
            }
        });
    })
    .response
    .rect
}

fn deck_button_tooltip(label: &str) -> &'static str {
    match label {
        // Math & Trig
        "sin" => "Sine — sin(x) in radians (or sin(30 deg))",
        "cos" => "Cosine — cos(x) in radians (or cos(60 deg))",
        "tan" => "Tangent — tan(x) in radians (or tan(45 deg))",
        "sqrt" => "Principal Square Root — sqrt(x) (exact decimal/integer & complex)",
        "asin" => "Inverse Sine (Arcsine) — asin(x), domain [-1, 1]",
        "acos" => "Inverse Cosine (Arccosine) — acos(x), domain [-1, 1]",
        "atan" => "Inverse Tangent (Arctangent) — atan(x) (also supports atan2(y, x))",
        "cbrt" => "Cube Root — cbrt(x) (exact decimal/integer & real root)",
        "sinh" => "Hyperbolic Sine — sinh(x)",
        "cosh" => "Hyperbolic Cosine — cosh(x)",
        "tanh" => "Hyperbolic Tangent — tanh(x)",
        "hypot" => "Euclidean Hypotenuse — hypot(x, y) = √(x² + y²)",
        "ln" => "Natural Logarithm (Base e) — ln(x)",
        "log10" => "Common Logarithm (Base 10) — log10(x)",
        "abs" => "Absolute Value / Magnitude — abs(x)",
        "n!" => "Exact Arbitrary-Precision Factorial — n! (up to 10000!)",
        "pi" => "Archimedes' Constant π ≈ 3.141592653589793",
        "tau" => "Circle Constant τ = 2π ≈ 6.283185307179586",
        "phi" => "Golden Ratio φ = (1 + √5) / 2 ≈ 1.618033988749895",
        "e" => "Euler's Number e ≈ 2.718281828459045",
        // Logic
        "AND (&)" => "Logical Conjunction (AND) — True when both propositions hold",
        "OR (|)" => "Logical Disjunction (OR) — True when at least one proposition holds",
        "NOT (!)" => "Logical Negation (NOT) — Inverts proposition truth value",
        "XOR (^)" => "Logical Exclusive OR (XOR) — True when propositions differ",
        "-> (Implies)" => "Material Implication (p -> q) — Equivalent to !p | q",
        "<-> (Iff)" => "Biconditional Equivalence (p <-> q) — True when p equals q",
        "p" => "Boolean Proposition Variable p",
        "q" => "Boolean Proposition Variable q",
        "r" => "Boolean Proposition Variable r",
        "s" => "Boolean Proposition Variable s",
        "TRUE" => "Boolean Constant TRUE (1)",
        "FALSE" => "Boolean Constant FALSE (0)",
        // Linear Algebra
        "det(M)" => "Matrix Determinant — det(A) via LU Decomposition",
        "inv(M)" => "Matrix Inverse — inv(A) via AVX2/NEON Gauss-Jordan Elimination",
        "transpose(M)" => "Matrix Transpose — transpose(A) swaps rows and columns",
        "trace(M)" => "Matrix Trace — trace(A) sums main diagonal elements",
        "rank(M)" => "Matrix Rank — rank(A) counts linearly independent rows",
        "eig(M)" => "Eigenvalues — eig(A) (closed-form 2×2 & symmetric N×N Jacobi)",
        "solve(A,b)" => "Linear System Solver — solve(A, b) solves A · x = b",
        "norm(M)" => "Frobenius Matrix / Euclidean Vector Norm — norm(M)",
        "eye(N)" => "Identity Matrix — eye(N) constructs an N×N unit matrix",
        "[ (Open)" => "Open Matrix / Vector Literal (e.g. [1, 2; 3, 4])",
        "] (Close)" => "Close Matrix / Vector Literal",
        "; (Row Sep)" => "Matrix Row Separator ';'",
        // Number Theory & Calculus
        "primes(n)" => "Prime Factorization — primes(n) via Miller-Rabin & Brent-Pollard Rho",
        "fib(n)" => "Exact N-th Fibonacci Number — fib(n) via Fast Doubling (n ≤ 20000)",
        "fac(n)" => "Exact Factorial Function — fac(n)",
        "gcd(a, b)" => "Greatest Common Divisor — gcd(a, b) via 128-bit Stein Binary GCD",
        "lcm(a, b)" => "Least Common Multiple — lcm(a, b) in exact 128-bit arithmetic",
        "nCr(n, r)" => "Binomial Combinations — nCr(n, r) = n! / (r! · (n - r)!)",
        "nPr(n, r)" => "Permutations — nPr(n, r) = n! / (n - r)!",
        "% (Mod)" => "Euclidean Modulo Remainder — a mod b",
        "diff(f)" => "Symbolic / Richardson O(h⁶) Derivative — diff(x^3 + 2x) or diff(sin(x), 0)",
        "int(f,a,b)" => "Symbolic / 128-Node Gauss-Legendre Integral — integrate(f) or integrate(f, a, b)",
        "roots(p)" => "All Real & Complex Polynomial Roots — roots(x^2 - 5x + 6)",
        "i (Imag)" => "Imaginary Unit i (i² = -1)",
        "root(n, x)" => "Real N-th Root — root(n, x) = x^(1/n)",
        "log(b, x)" => "Arbitrary-Base Logarithm — log(base, value)",
        "sgn(x)" => "Signum Function — sgn(x) ∈ {-1, 0, 1}",
        "deg(rad)" => "Radians to Degrees — deg(pi) = 180",
        // Programmer Bitwise
        "band(a, b)" => "64-Bit Bitwise AND — band(a, b)",
        "bor(a, b)" => "64-Bit Bitwise OR — bor(a, b)",
        "bxor(a, b)" => "64-Bit Bitwise XOR — bxor(a, b)",
        "bnot(a)" => "64-Bit Bitwise NOT (One's Complement) — bnot(a)",
        "shl(a, n)" => "64-Bit Logical Shift Left — shl(val, bits)",
        "shr(a, n)" => "64-Bit Logical Shift Right — shr(val, bits) (also ashr for arithmetic)",
        "popcnt(a)" => "Population Count — popcnt(a) counts set '1' bits in 64-bit word",
        "clz(a)" => "Count Leading Zeros — clz(a) in 64-bit word (also ctz)",
        "rol(a, n)" => "64-Bit Rotate Left — rol(val, bits)",
        "ror(a, n)" => "64-Bit Rotate Right — ror(val, bits)",
        "0x (Hex)" => "Hexadecimal Literal Prefix (e.g. 0xFF00)",
        "0b (Bin)" => "Binary Literal Prefix (e.g. 0b101101)",
        // Stats & Units
        "min(..)" => "Minimum Value — min(a, b, c, ...)",
        "max(..)" => "Maximum Value — max(a, b, c, ...)",
        "sum(..)" => "Exact Summation — sum(a, b, c, ...)",
        "mean(..)" => "Arithmetic Mean (Average) — mean(a, b, c, ...)",
        "median(..)" => "Exact Statistical Median — median(a, b, c, ...)",
        "std(..)" => "Sample Standard Deviation — std(a, b, c, ...)",
        "var(..)" => "Sample Variance — var(a, b, c, ...)",
        "convert(..)" => "Unit Conversion Function — convert(value, \"km\", \"mi\")",
        "km to m" => "Length Conversion Example — Kilometers to Meters (10 km to m)",
        "c to f" => "Temperature Conversion Example — Celsius to Fahrenheit (100 c to f)",
        "lb to kg" => "Mass Conversion Example — Pounds to Kilograms (10 lb to kg)",
        "gb to mb" => "Data Size Conversion Example — Gigabytes to Megabytes (1 gb to mb)",
        "kwh to j" => "Energy Conversion Example — Kilowatt-Hours to Joules (1 kwh to j)",
        "hp to w" => "Power Conversion Example — Mechanical Horsepower to Watts (1 hp to w)",
        "psi to bar" => "Pressure Conversion Example — PSI to Bar (100 psi to bar)",
        "mps to kph" => "Speed Conversion Example — Meters/Second to km/h (10 mps to kph)",
        _ => "Insert Into Expression Editor",
    }
}

fn render_4col_row(
    ui: &mut Ui,
    items: &[(&str, &str); 4],
    col_w: f32,
    btn_h: f32,
    color: egui::Color32,
    state: &mut EditorState,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = UNIFORM_GRID_BTN_GAP;
        for &(label, insert) in items {
            let btn = egui::Button::new(RichText::new(label).color(color).size(11.0));
            if ui
                .add_sized([col_w, btn_h], btn)
                .on_hover_text(deck_button_tooltip(label))
                .clicked()
            {
                if insert.ends_with('(') {
                    state.insert_func_template(insert);
                } else {
                    state.insert_text(insert);
                }
                ui.ctx().request_repaint();
            }
        }
    });
}

#[allow(dead_code)]
pub fn format_num_clean(val: f64) -> String { // wiring-ok
    format_clean_float(val)
}

fn format_nanos(nanos: u128) -> String {
    if nanos < 1_000 {
        format!("{} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.1} µs", nanos as f64 / 1_000.0)
    } else {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    }
}

fn predict_algo_complexity(expr: &str) -> &'static str {
    if expr.contains('!') {
        "Complexity Class: O(N log N) Arbitrary-Precision Factorial"
    } else if expr.contains("**") || expr.contains('^') {
        "Complexity Class: O(log E · M(D)) Binary Exponentiation"
    } else if expr.contains(';') || expr.contains("inv(") || expr.contains("det(") {
        "Complexity Class: O(N³) Gaussian Elimination"
    } else if expr.contains("primes(") || expr.contains("factors(") {
        "Complexity Class: O(N¼) Pollard's Rho + Miller-Rabin"
    } else {
        "Complexity Class: High-Throughput Rayon Worker Thread"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_editor_state_end_to_end_input_spam_and_coalescing() {
        let mut editor = EditorState::new();
        let vars = FxHashMap::default();

        // 1. Simulate rapid character-by-character typing of `9999**9999 + 12345 * 2`
        let spam_sequence = "9999**9999 + 12345 * 2";
        for ch in spam_sequence.chars() {
            let t0 = Instant::now();
            let mut buf = [0u8; 4];
            editor.insert_text(ch.encode_utf8(&mut buf));
            editor.step_input_wall(&vars, true, 35);
            let step_elapsed = t0.elapsed();
            assert!(
                step_elapsed < Duration::from_millis(5),
                "EditorState::step_input_wall blocked UI for {:?} while typing '{}' (state: '{}')",
                step_elapsed,
                ch,
                editor.input_text
            );
        }

        // After typing `9999**9999 + 12345 * 2`, it must be dispatched to the background worker
        assert!(editor.task_in_flight);

        // 2. Immediately clear and type a fast banking-tier sum (`38,28+60,23+9,17 +2,1`).
        //    The Input Wall must preempt the in-flight `9999**9999 + 12345 * 2` worker and
        //    evaluate the fast sum synchronously in 0 frames!
        editor.clear();
        editor.insert_text("38,28+60,23+9,17 +2,1");
        let fast_t0 = Instant::now();
        editor.step_input_wall(&vars, true, 35);
        assert!(fast_t0.elapsed() < Duration::from_millis(5));
        assert!(!editor.task_in_flight);
        assert_eq!(
            editor.cached_display.as_ref().map(|d| d.dec_str.as_str()),
            Some("109.78")
        );

        // 3. Simulate spamming heavy power expressions (`9999**8000`, `9999**9500`) and
        //    settling on `2**5000 + 1`. Verify coalescing + worker completion.
        editor.clear();
        editor.insert_text("9999**8000 + 1");
        editor.step_input_wall(&vars, true, 25);
        editor.clear();
        editor.insert_text("9999**9500 + 2");
        editor.step_input_wall(&vars, true, 25);
        editor.clear();
        editor.insert_text("2**5000 + 1");
        editor.step_input_wall(&vars, true, 25);
        assert!(editor.task_in_flight);

        let wait_start = Instant::now();
        while editor.task_in_flight && wait_start.elapsed() < Duration::from_millis(500) {
            std::thread::sleep(Duration::from_millis(5));
            editor.step_input_wall(&vars, true, 25);
        }

        assert!(
            !editor.task_in_flight,
            "Worker should finish `2**5000 + 1` after preempting earlier spam"
        );
        let disp = editor
            .cached_display
            .as_ref()
            .expect("Expected cached_display for 2**5000 + 1");
        assert_eq!(disp.dec_str.len(), 1506);
        assert!(disp.dec_str.ends_with("377"));
        assert!(!editor.live_result.as_ref().unwrap().is_speculative);

        // 4. Verify trailing spaces and trailing binary operators (`" "`, `"+"`, `" "`) do NOT
        //    abort or re-dispatch the already-computed `2**5000 + 1`, and mark it speculative in O(1).
        let settled_epoch = editor.current_epoch;
        for ch in [" ", "+", " "] {
            editor.insert_text(ch);
            editor.step_input_wall(&vars, true, 25);
            assert_eq!(
                editor.current_epoch, settled_epoch,
                "Trailing whitespace/operator '{}' should not increment epoch or restart worker",
                ch
            );
            assert!(!editor.task_in_flight);
            assert!(editor.live_result.is_some());
        }
        assert!(editor.live_result.as_ref().unwrap().is_speculative);

        // 5. Verify that transitioning from a valid result (`9999`) to a massive power (`9999**999999`)
        //    clears stale `live_result` while waiting at the settle wall, and then calculates the big number
        //    once the input stays the same!
        editor.clear();
        editor.insert_text("9999");
        editor.step_input_wall(&vars, true, 25);
        assert_eq!(
            editor.cached_display.as_ref().map(|d| d.dec_str.as_str()),
            Some("9999")
        );
        editor.insert_text("**999999");
        editor.step_input_wall(&vars, true, 25);
        assert!(
            editor.live_result.is_none() && editor.cached_display.is_none(),
            "Settled deep-compute dispatch must clear stale 9999 while waiting at the settle wall"
        );
        assert!(editor.task_in_flight);

        let settle_wait = Instant::now();
        while editor.task_in_flight && settle_wait.elapsed() < Duration::from_millis(500) {
            std::thread::sleep(Duration::from_millis(5));
            editor.step_input_wall(&vars, true, 25);
        }
        assert!(
            !editor.task_in_flight && editor.live_result.is_some(),
            "After settle timeout when input stays the same, 9999**999999 must be calculated"
        );
        assert!(
            editor.cached_display.as_ref().unwrap().dec_str.contains("3999953 digits"),
            "Expected 3999953-digit scientific summary for 9999**999999, got: {}",
            editor.cached_display.as_ref().unwrap().dec_str
        );

        editor.clear();
        editor.insert_text("9999");
        editor.step_input_wall(&vars, true, 25);
        assert!(editor.live_result.is_some());
        editor.insert_text("/0");
        editor.step_input_wall(&vars, true, 25);
        assert!(
            editor.live_result.is_none() && editor.cached_display.is_none(),
            "Division by zero must clear stale live_result"
        );
        assert!(editor.live_error.is_some());
    }

    #[test]
    fn test_hotkeys_copy_paste_select_all_and_limitless_undo_redo() {
        let mut editor = EditorState::new();
        let vars = FxHashMap::default();

        // 1. Type initial expression
        editor.insert_text("[1, 2; 3, 4] * 2");
        assert_eq!(editor.input_text, "[1, 2; 3, 4] * 2");
        assert!(!editor.has_selection());

        // 2. When nothing is selected (caret_start == caret_end at end, middle, or 0),
        //    Ctrl+C (`action_copy`) must return the whole input text!
        assert_eq!(editor.action_copy(), "[1, 2; 3, 4] * 2");
        editor.selection_range = (5, 5); // caret in the middle, still caret_start == caret_end
        assert!(!editor.has_selection());
        assert_eq!(editor.action_copy(), "[1, 2; 3, 4] * 2");
        editor.selection_range = (0, 0); // caret at start, still caret_start == caret_end
        assert_eq!(editor.action_copy(), "[1, 2; 3, 4] * 2");

        // 3. When a substring is selected (`caret_start != caret_end`), Ctrl+C copies only the selection
        editor.selection_range = (0, 12);
        assert!(editor.has_selection());
        assert_eq!(editor.action_copy(), "[1, 2; 3, 4]");

        // 4. Ctrl+A (`action_select_all`) selects all text; subsequent Ctrl+V (`action_paste`) replaces all
        let (sel_lo, sel_hi) = editor.action_select_all();
        assert_eq!((sel_lo, sel_hi), (0, 16));
        assert!(editor.has_selection());
        assert_eq!(editor.action_copy(), "[1, 2; 3, 4] * 2");

        editor.action_paste("roots(x^2 - 5x + 6)");
        assert_eq!(editor.input_text, "roots(x^2 - 5x + 6)");
        assert!(!editor.has_selection());
        editor.step_input_wall(&vars, true, 25);
        assert_eq!(
            editor.cached_display.as_ref().map(|d| d.dec_str.as_str()),
            Some("[2, 3]")
        );

        // 5. Pasting when nothing is selected and caret is in the middle inserts at caret
        editor.selection_range = (5, 5); // right after "roots"
        editor.action_paste("_poly");
        assert_eq!(editor.input_text, "roots_poly(x^2 - 5x + 6)");

        // 6. Undo (`Ctrl+Z`) and Redo (`Ctrl+Y`) restore exact states and cursor positions
        assert!(editor.action_undo());
        assert_eq!(editor.input_text, "roots(x^2 - 5x + 6)");
        assert!(editor.action_undo());
        assert_eq!(editor.input_text, "[1, 2; 3, 4] * 2");
        assert!(editor.action_undo());
        assert_eq!(editor.input_text, "");
        assert!(!editor.action_undo()); // stack now at bottom

        assert!(editor.action_redo());
        assert_eq!(editor.input_text, "[1, 2; 3, 4] * 2");
        assert!(editor.action_redo());
        assert_eq!(editor.input_text, "roots(x^2 - 5x + 6)");
        assert!(editor.action_redo());
        assert_eq!(editor.input_text, "roots_poly(x^2 - 5x + 6)");
        assert!(!editor.action_redo()); // stack now at top

        // 7. Limitless Undo/Redo stress test across 300 sequential edits
        editor.clear();
        assert_eq!(editor.input_text, "");
        let mut snapshots = Vec::with_capacity(300);
        for i in 0..300 {
            snapshots.push(editor.input_text.clone());
            editor.insert_text(&format!("+{}", i));
        }
        assert_eq!(snapshots.len(), 300);
        let final_expr = editor.input_text.clone();

        for i in (0..300).rev() {
            assert!(editor.action_undo(), "Undo failed at step {}", i);
            assert_eq!(editor.input_text, snapshots[i]);
        }
        assert_eq!(editor.input_text, "");

        for i in 0..300 {
            assert!(editor.action_redo(), "Redo failed at step {}", i);
            if i + 1 < 300 {
                assert_eq!(editor.input_text, snapshots[i + 1]);
            }
        }
        assert_eq!(editor.input_text, final_expr);

        // 8. Win32 native clipboard roundtrip verification
        #[cfg(windows)]
        {
            let probe = "3² × 5 + roots(x^2 - 5x + 6)";
            if win_clipboard::set_text(probe) {
                assert_eq!(win_clipboard::get_text().as_deref(), Some(probe));
            }
        }
    }

    #[test]
    fn test_kerned_mono_job_super_sub_kerning() {
        let color = egui::Color32::WHITE;

        // 1. Single-section LayoutJob invariant (0.0 leading_space, full byte range, break_anywhere)
        for sample_text in [
            "[1, 2; 3, 4] * 2",
            "2¹⁰ × x₁ + 5",
            "2⁴\u{2009}+\u{2009}x₁₂",
        ] {
            let job = build_kerned_mono_job(sample_text, 400.0, 12.0, color);
            assert_eq!(job.sections.len(), 1);
            assert_eq!(job.sections[0].leading_space, 0.0);
            assert_eq!(job.sections[0].byte_range, 0..sample_text.len());
            assert!(job.wrap.break_anywhere);
        }

        // 2. Verify subpixel Galley kerning pass: exact uniform cell step (4.80 * scale),
        // quad centering, base_pull (0.60 * scale), and multi-line wrapping across all DPI scales
        let ctx = egui::Context::default();
        super::super::paper_theme::setup_fonts(&ctx);
        for &ppp in &[1.0_f32, 1.25, 1.5, 1.75, 2.0] {
            ctx.set_pixels_per_point(ppp);
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    for &font_size in &[12.0_f32, 14.0] {
                        let scale = font_size / 12.0;
                        let expected_w_sup = SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT * scale;
                        let expected_base_pull = SUBPIXEL_SUP_SUB_BASE_PULL_12PT * scale;

                        for sample in [
                            "1⁰¹²³⁴⁵⁶⁷⁸⁹",
                            "2⁴⁴⁸⁸⁰⁰⁺⁻⁼",
                            "x₀₁₂₃₄₅₆₇₈₉",
                            "1¹²³⁴⁵⁶⁷⁸⁹⁰\u{2009}+\u{2009}2¹⁰\u{2009}×\u{2009}3¹²\u{2009}+\u{2009}5¹⁴",
                            "2ⁿ⁺¹\u{2009}+\u{2009}xᵢ₊₁\u{2009}+\u{2009}x₁₂³⁴\u{2009}+\u{2009}x²₁",
                        ] {
                            let g = cmd_layouter(ui, sample, 600.0, font_size, egui::Color32::WHITE);
                            let row = &g.rows[0];
                            let verts = &row.visuals.mesh.vertices;
                            let mut vi = 0;
                            let mut prev_sup: Option<(char, f32, f32)> = None;
                            let mut prev_base_end: Option<(char, f32, f32)> = None;
                            let mut sup_run_idx = 0usize;

                            for gl in &row.glyphs {
                                if !gl.uv_rect.is_nothing() {
                                    let vx0 = verts[vi].pos.x;
                                    let vx1 = verts[vi + 1].pos.x;
                                    let quad_center_x = 0.5 * (vx0 + vx1);
                                    if is_super_or_sub_char(gl.chr) {
                                        assert!(
                                            (gl.advance_width - expected_w_sup).abs() < 1e-3,
                                            "Expected advance_width {} for '{}', got {}",
                                            expected_w_sup,
                                            gl.chr,
                                            gl.advance_width
                                        );
                                        assert!(
                                            (quad_center_x - (gl.pos.x + 0.5 * expected_w_sup)).abs() < 1e-3,
                                            "Expected quad center of '{}' at cell midpoint {}, got {}",
                                            gl.chr,
                                            gl.pos.x + 0.5 * expected_w_sup,
                                            quad_center_x
                                        );
                                        if sup_run_idx == 0 {
                                            if let Some((bch, b_end_x, b_vx1)) = prev_base_end {
                                                let pull = b_end_x - gl.pos.x;
                                                assert!(
                                                    (pull - expected_base_pull).abs() < 1e-3,
                                                    "Expected base_pull {} after '{}' before '{}' in '{}', got {}",
                                                    expected_base_pull,
                                                    bch,
                                                    gl.chr,
                                                    sample,
                                                    pull
                                                );
                                                assert!(
                                                    quad_center_x > b_vx1,
                                                    "First super/sub '{}' (center={}) overlapped base '{}' (vx1={}) in '{}' at ppp={}",
                                                    gl.chr,
                                                    quad_center_x,
                                                    bch,
                                                    b_vx1,
                                                    sample,
                                                    ppp
                                                );
                                            }
                                        }
                                        if let Some((pch, prev_pos_x, prev_qc_x)) = prev_sup {
                                            let pos_step = gl.pos.x - prev_pos_x;
                                            let qc_step = quad_center_x - prev_qc_x;
                                            assert!(
                                                (pos_step - expected_w_sup).abs() < 1e-3,
                                                "Expected uniform subpixel pos step {} between '{}' and '{}' in '{}' at ppp={}, got {}",
                                                expected_w_sup,
                                                pch,
                                                gl.chr,
                                                sample,
                                                ppp,
                                                pos_step
                                            );
                                            assert!(
                                                (qc_step - expected_w_sup).abs() < 1e-3,
                                                "Expected uniform subpixel quad-center step {} between '{}' and '{}' in '{}' at ppp={}, got {}",
                                                expected_w_sup,
                                                pch,
                                                gl.chr,
                                                sample,
                                                ppp,
                                                qc_step
                                            );
                                        }
                                        prev_sup = Some((gl.chr, gl.pos.x, quad_center_x));
                                        sup_run_idx += 1;
                                    } else {
                                        prev_sup = None;
                                        prev_base_end = if !gl.chr.is_whitespace() {
                                            Some((gl.chr, gl.pos.x + gl.advance_width, vx1))
                                        } else {
                                            None
                                        };
                                        sup_run_idx = 0;
                                    }
                                    vi += 4;
                                } else {
                                    prev_sup = None;
                                    prev_base_end = None;
                                    sup_run_idx = 0;
                                }
                            }
                        }
                    }

                    // 3. Verify narrow-window multi-line wrapping of long superscripts, subscripts, and mixed rows
                    for &(sample_wrap, wrap_w) in &[
                        ("1⁰¹²³⁴⁵⁶⁷⁸⁹⁰¹²³⁴⁵⁶⁷⁸⁹", 50.0_f32),
                        ("x₀₁₂₃₄₅₆₇₈₉₀₁₂₃₄₅₆₇₈₉", 54.5_f32),
                        ("1⁰¹²³⁴⁵⁶⁷⁸⁹⁰¹²³⁴⁵⁶+12345678901234567", 100.0_f32),
                    ] {
                        let g_wrap = cmd_layouter(ui, sample_wrap, wrap_w, 12.0, egui::Color32::WHITE);
                        assert!(
                            g_wrap.rows.len() >= 2,
                            "Expected '{}' at wrap_w={} to wrap across >= 2 rows",
                            sample_wrap,
                            wrap_w
                        );
                        assert!(
                            g_wrap.rect.width() <= wrap_w + 1e-2,
                            "Galley width {} exceeded wrap_w {} for '{}' at ppp={}",
                            g_wrap.rect.width(),
                            wrap_w,
                            sample_wrap,
                            ppp
                        );
                        for (r_idx, row) in g_wrap.rows.iter().enumerate() {
                            assert!(
                                row.rect.max.x <= wrap_w + 1e-2,
                                "Row {} max.x {} exceeded wrap_w {} for '{}' at ppp={}",
                                r_idx,
                                row.rect.max.x,
                                wrap_w,
                                sample_wrap,
                                ppp
                            );
                            let mut prev_sup_pos: Option<(char, f32)> = None;
                            for gl in &row.glyphs {
                                if is_super_or_sub_char(gl.chr) {
                                    if let Some((pch, p_x)) = prev_sup_pos {
                                        let step = gl.pos.x - p_x;
                                        assert!(
                                            (step - SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT).abs() < 1e-3,
                                            "Wrapped row {} step between '{}' and '{}' at ppp={} expected {}, got {}",
                                            r_idx,
                                            pch,
                                            gl.chr,
                                            ppp,
                                            SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT,
                                            step
                                        );
                                    }
                                    prev_sup_pos = Some((gl.chr, gl.pos.x));
                                } else {
                                    prev_sup_pos = None;
                                }
                            }
                        }
                    }

                    // 4. Verify multi-row (x + 2)^12 selection & cursor hit-testing across DPI scales
                    let poly12 = "x¹² + 24x¹¹ + 264x¹⁰ + 1760x⁹ + 7920x⁸ + 25344x⁷ + 59136x⁶ + 101376x⁵ + 126720x⁴ + 112640x³ + 67584x² + 24576x + 4096";
                    let g_poly12 = cmd_layouter(ui, poly12, 568.0, 18.0, egui::Color32::WHITE);
                    assert!(g_poly12.rows.len() >= 2);
                    assert!(g_poly12.rect.width() <= 568.0 + 1e-2);
                    for (r_idx, row) in g_poly12.rows.iter().enumerate() {
                        let row_mid_y = row.rect.center().y;
                        for (c_idx, gl) in row.glyphs.iter().enumerate() {
                            let mid_x = gl.pos.x + 0.5 * gl.advance_width;
                            let hit_cursor = g_poly12.cursor_from_pos(egui::vec2(mid_x, row_mid_y));
                            assert_eq!(hit_cursor.rcursor.row, r_idx);
                            assert!(
                                hit_cursor.rcursor.column == c_idx || hit_cursor.rcursor.column == c_idx + 1,
                                "Cursor hit at glyph ({}, {}) '{}' mid_x={} resolved to col {}",
                                r_idx,
                                c_idx,
                                gl.chr,
                                mid_x,
                                hit_cursor.rcursor.column
                            );
                        }
                    }
                });
            });
        }
    }

    #[test]
    fn test_egui_context_render_editor_hotkeys_and_galley_kerning() {
        let ctx = egui::Context::default();
        super::super::paper_theme::setup_fonts(&ctx);
        let palette = super::super::paper_theme::get_palette(0);
        let mut editor = EditorState::new();
        let mut sidebar = SidebarState::new();
        let mut vars = FxHashMap::default();
        let mut config = AppConfig::default();

        let ctrl_mods = egui::Modifiers {
            alt: false,
            ctrl: true,
            shift: false,
            mac_cmd: false,
            command: true,
        };

        // Frame 1: Initial render & verify Galley glyph X-positions for superscript & subscript kerning
        let mut galley_gap_before_sup = 0.0_f32;
        let mut galley_gap_after_sup = 0.0_f32;
        let mut sup_adv = 0.0_f32;
        let mut sub_step = 0.0_f32;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let g = cmd_layouter(ui, "3² + x₁₂", 400.0, 12.0, egui::Color32::BLACK);
                let glyphs = &g.rows[0].glyphs;
                assert_eq!(glyphs[0].chr, '3');
                assert_eq!(glyphs[1].chr, '²');
                assert_eq!(glyphs[2].chr, ' ');
                sup_adv = glyphs[1].advance_width;
                galley_gap_before_sup = glyphs[1].pos.x - (glyphs[0].pos.x + glyphs[0].advance_width);
                galley_gap_after_sup = glyphs[2].pos.x - (glyphs[1].pos.x + glyphs[1].advance_width);

                // Verify subscripts '₁' and '₂' step by exact uniform SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT
                let g_sub1 = glyphs.iter().find(|gl| gl.chr == '₁').unwrap();
                let g_sub2 = glyphs.iter().find(|gl| gl.chr == '₂').unwrap();
                sub_step = g_sub2.pos.x - g_sub1.pos.x;

                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert!(
            (sup_adv - SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT).abs() < 1e-3,
            "Expected superscript advance_width {}, got {}",
            SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT,
            sup_adv
        );
        assert!(
            (galley_gap_before_sup - (-SUBPIXEL_SUP_SUB_BASE_PULL_12PT)).abs() < 1e-3,
            "Expected superscript base_pull -{}, got {}",
            SUBPIXEL_SUP_SUB_BASE_PULL_12PT,
            galley_gap_before_sup
        );
        assert!(
            galley_gap_after_sup.abs() < 1e-3,
            "Expected contiguous cell boundary after superscript (0.0 gap), got {}",
            galley_gap_after_sup
        );
        assert!(
            (sub_step - SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT).abs() < 1e-3,
            "Subscripts '₁' and '₂' must have exact uniform step {}, got {}",
            SUBPIXEL_SUP_SUB_CELL_WIDTH_12PT,
            sub_step
        );

        // Frame 2: Inject Paste event -> should populate input_text and evaluate
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Paste("3^2 + 4^2".to_string()));
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(editor.input_text, "3^2 + 4^2");
        assert_eq!(editor.cached_display.as_ref().map(|d| d.dec_str.as_str()), Some("25"));

        // Verify backspace at caret index 0 (0, 0) is a no-op and does not pop the end of input_text
        editor.selection_range = (0, 0);
        editor.backspace();
        assert_eq!(editor.input_text, "3^2 + 4^2");
        editor.selection_range = (9, 9);

        // Frame 3: Inject Event::Copy when nothing is selected (caret_start == caret_end) -> copies whole input
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Copy);
        let out = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out.platform_output.copied_text, "3^2 + 4^2");

        // Frame 4: Inject Ctrl+A -> selects all (0..9)
        let mut raw = egui::RawInput::default();
        raw.modifiers = ctrl_mods;
        raw.events.push(egui::Event::Key {
            key: Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(editor.normalized_selection(), (0, 9));

        // Frame 5: Inject Paste while all is selected -> replaces selection cleanly
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Paste("roots(x^2 - 5x + 6)".to_string()));
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(editor.input_text, "roots(x^2 - 5x + 6)");
        assert_eq!(editor.cached_display.as_ref().map(|d| d.dec_str.as_str()), Some("[2, 3]"));

        // Frame 5b: Place caret in the middle (index 5) and press Enter -> commits to history
        // WITHOUT inserting '\n' into input_text or polluting undo_stack
        editor.selection_range = (5, 5);
        let undo_len_before_enter = editor.undo_stack.len();
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Key {
            key: Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        raw.events.push(egui::Event::Text("\n".to_string()));
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(editor.input_text, "roots(x^2 - 5x + 6)");
        assert_eq!(editor.undo_stack.len(), undo_len_before_enter);
        assert_eq!(sidebar.history.len(), 1);
        assert_eq!(sidebar.history[0].input, "roots(x^2 - 5x + 6)");

        // Frame 6: Inject single Ctrl+Z after Enter -> immediately undoes back to "3^2 + 4^2"
        let mut raw = egui::RawInput::default();
        raw.modifiers = ctrl_mods;
        raw.events.push(egui::Event::Key {
            key: Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(editor.input_text, "3^2 + 4^2");
        assert_eq!(editor.cached_display.as_ref().map(|d| d.dec_str.as_str()), Some("25"));

        // Frame 7: Inject Ctrl+Y -> redoes back to "roots(x^2 - 5x + 6)"
        let mut raw = egui::RawInput::default();
        raw.modifiers = ctrl_mods;
        raw.events.push(egui::Event::Key {
            key: Key::Y,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(editor.input_text, "roots(x^2 - 5x + 6)");
    }

    #[test]
    fn test_equiv_glyph_dpi_vcenter_alignment_and_single_answer_subline() {
        let ctx = egui::Context::default();
        super::super::paper_theme::setup_fonts(&ctx);

        // 1. Verify DPI-calculated pixel v-center alignment of '≡' across multiple DPI scales
        for &ppp in &[1.0_f32, 1.25, 1.5, 1.75, 2.0] {
            ctx.set_pixels_per_point(ppp);
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    // Measure unadjusted raw '≡' vs adjusted cmd_layouter '≡\u{2009}' and '2² × 7'
                    let raw_job = build_kerned_mono_job("≡\u{2009}", 64.0, 12.0, egui::Color32::BLACK);
                    let raw_galley = ui.fonts(|f| f.layout_job(raw_job));
                    let raw_v = &raw_galley.rows[0].visuals.mesh.vertices;
                    let raw_equiv_vcenter = 0.5 * (raw_v[0].pos.y + raw_v[2].pos.y);

                    let adj_prefix = cmd_layouter(ui, "≡\u{2009}", 64.0, 12.0, egui::Color32::BLACK);
                    let adj_v = &adj_prefix.rows[0].visuals.mesh.vertices;
                    let adj_equiv_vcenter = 0.5 * (adj_v[0].pos.y + adj_v[2].pos.y);

                    let body_galley = cmd_layouter(ui, "2² × 7", 200.0, 12.0, egui::Color32::BLACK);
                    let body_v = &body_galley.rows[0].visuals.mesh.vertices;
                    // First glyph '2' has 4 vertices [0..4]
                    let digit_vcenter = 0.5 * (body_v[0].pos.y + body_v[2].pos.y);

                    // Confirm '≡' was shifted upwards in -y
                    assert!(
                        adj_equiv_vcenter < raw_equiv_vcenter,
                        "Expected '≡' vcenter ({}) to be shifted in -y relative to unadjusted ({}) at ppp={}",
                        adj_equiv_vcenter,
                        raw_equiv_vcenter,
                        ppp
                    );

                    // Confirm physical pixel quantization of shifted vertices
                    let top_px = adj_v[0].pos.y * ppp;
                    let bot_px = adj_v[2].pos.y * ppp;
                    assert!(
                        (top_px - top_px.round()).abs() < 1e-3 && (bot_px - bot_px.round()).abs() < 1e-3,
                        "Expected DPI-snapped integer pixel coordinates at ppp={}, got top_px={}, bot_px={}",
                        ppp,
                        top_px,
                        bot_px
                    );

                    // Confirm '≡' ink vertical center matches digit '2' ink vertical center within <= 0.55 physical px
                    let diff_px = (adj_equiv_vcenter - digit_vcenter).abs() * ppp;
                    assert!(
                        diff_px <= 0.55,
                        "Expected '≡' vcenter ({}) to match text vcenter ({}) within <= 0.55 physical px at ppp={}, got diff_px={}",
                        adj_equiv_vcenter,
                        digit_vcenter,
                        ppp,
                        diff_px
                    );

                    // Confirm standalone prefix galley height matches single-row body galley height
                    assert!(
                        (adj_prefix.rect.height() - body_galley.rows[0].rect.height()).abs() < 1e-3,
                        "Prefix galley height ({}) must match body row height ({})",
                        adj_prefix.rect.height(),
                        body_galley.rows[0].rect.height()
                    );

                    // Confirm combined single-galley "≡ 3 × 5" also aligns '≡' to the digit '3' v-center
                    let combined_g = cmd_layouter(ui, "≡ 3 × 5", 200.0, 12.0, egui::Color32::BLACK);
                    let comb_v = &combined_g.rows[0].visuals.mesh.vertices;
                    // Vertex [0..4] is '≡', Vertex [4..8] is '3' (space has no vertices)
                    let comb_equiv_vc = 0.5 * (comb_v[0].pos.y + comb_v[2].pos.y);
                    let comb_digit_vc = 0.5 * (comb_v[4].pos.y + comb_v[6].pos.y);
                    assert!(
                        (comb_equiv_vc - comb_digit_vc).abs() * ppp <= 0.55,
                        "Combined galley '≡' vcenter ({}) must match '3' vcenter ({}) at ppp={}",
                        comb_equiv_vc,
                        comb_digit_vc,
                        ppp
                    );

                    // Confirm descenders ('[', ']', ',', '√') in combined galleys never pull '≡' down (+y)
                    // and keep '≡' aligned within <= 0.55 physical px of the digits on that row.
                    for (desc_sample, digit_v_start) in [
                        ("≡ [[1, 2], [3, 4]]", 12usize), // '≡'(0), '['(4), '['(8), '1'(12)
                        ("≡ √144 × 2", 8usize),        // '≡'(0), '√'(4), '1'(8)
                    ] {
                        let raw_desc_job = build_kerned_mono_job(desc_sample, 240.0, 12.0, egui::Color32::BLACK);
                        let raw_desc_g = ui.fonts(|f| f.layout_job(raw_desc_job));
                        let raw_dv = &raw_desc_g.rows[0].visuals.mesh.vertices;
                        let raw_desc_equiv_vc = 0.5 * (raw_dv[0].pos.y + raw_dv[2].pos.y);

                        let desc_g = cmd_layouter(ui, desc_sample, 240.0, 12.0, egui::Color32::BLACK);
                        let dv = &desc_g.rows[0].visuals.mesh.vertices;
                        let desc_equiv_vc = 0.5 * (dv[0].pos.y + dv[2].pos.y);
                        let row_digit_vc = 0.5 * (dv[digit_v_start].pos.y + dv[digit_v_start + 2].pos.y);
                        assert!(
                            desc_equiv_vc <= raw_desc_equiv_vc + 1e-3,
                            "Descender sample '{}' must never shift '≡' down (+y): got {} vs raw {} at ppp={}",
                            desc_sample,
                            desc_equiv_vc,
                            raw_desc_equiv_vc,
                            ppp
                        );
                        assert!(
                            (desc_equiv_vc - row_digit_vc).abs() * ppp <= 0.55,
                            "Descender sample '{}' '≡' vcenter ({}) must match row digit vcenter ({}) at ppp={}",
                            desc_sample,
                            desc_equiv_vc,
                            row_digit_vc,
                            ppp
                        );
                    }
                });
            });
        }

        // 2. Verify PreparedDisplay::answer_equiv_subline deduplication and single-subline rules
        ctx.set_pixels_per_point(1.0);
        let disp_28 = PreparedDisplay::from_value(&CalcValue::FastInt(28), RadixMode::Dec).unwrap();
        assert_eq!(disp_28.answer_equiv_subline().as_deref(), Some("2² × 7"));

        let disp_frac = PreparedDisplay::from_value(
            &CalcValue::ExactDecimal {
                mantissa: 25,
                scale: 2,
            },
            RadixMode::Dec,
        )
        .unwrap();
        assert_eq!(disp_frac.answer_equiv_subline().as_deref(), Some("1/2²"));

        let disp_prime = PreparedDisplay::from_value(&CalcValue::FastInt(17), RadixMode::Dec).unwrap();
        assert_eq!(disp_prime.answer_equiv_subline().as_deref(), Some("17 (Prime)"));

        let disp_one = PreparedDisplay::from_value(&CalcValue::FastInt(1), RadixMode::Dec).unwrap();
        assert_eq!(disp_one.answer_equiv_subline(), None);

        let mut disp_dup = disp_28.clone();
        disp_dup.radix_str = "2² × 7".to_string();
        assert_eq!(disp_dup.answer_equiv_subline(), None);
    }

    #[test]
    fn test_single_subline_under_answer_framebox_and_universal_selection_copy() {
        let ctx = egui::Context::default();
        super::super::paper_theme::setup_fonts(&ctx);
        let palette = super::super::paper_theme::get_palette(0);
        let mut editor = EditorState::new();
        let mut sidebar = SidebarState::new();
        let mut vars = FxHashMap::default();
        let mut config = AppConfig::default();

        // Set expression to "4 * 7" -> canonical echo "4 × 7", answer "28", subline "2² × 7"
        editor.input_text = "4 * 7".to_string();
        editor.selection_range = (5, 5);
        editor.trigger_input_wall();

        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });

        // Inspect all painted Text shapes to verify:
        // 1. "2² × 7" is painted ONCE (never twice) and no "[ Factors:" badge exists.
        // 2. "2² × 7" is positioned strictly below the "= 28" answer text in Y.
        let mut answer_y = None;
        let mut factors_y_positions = Vec::new();
        for clipped in &out.shapes {
            if let egui::epaint::Shape::Text(t) = &clipped.shape {
                let txt = t.galley.text();
                assert!(
                    !txt.contains("[ Factors:"),
                    "Legacy duplicate '[ Factors: ... ]' badge must not be rendered, found: {}",
                    txt
                );
                if txt == "28" {
                    answer_y = Some(t.pos.y + t.galley.rect.max.y);
                } else if txt == "2² × 7" {
                    factors_y_positions.push(t.pos.y);
                }
            }
        }
        assert_eq!(
            factors_y_positions.len(),
            1,
            "Expected prime factors '2² × 7' to be painted exactly once, found {:?}",
            factors_y_positions
        );
        let ans_bottom = answer_y.expect("Expected answer '28' text shape");
        assert!(
            factors_y_positions[0] > ans_bottom,
            "Expected factors subline y ({}) to be strictly under the answer text bottom ({})",
            factors_y_positions[0],
            ans_bottom
        );

        // Now verify universal copy behavior:
        // (A) Nothing selected anywhere -> Ctrl+C copies whole input ("4 * 7")
        let ctrl_mods = egui::Modifiers {
            alt: false,
            ctrl: true,
            shift: false,
            mac_cmd: false,
            command: true,
        };
        let mut raw = egui::RawInput::default();
        raw.modifiers = ctrl_mods;
        raw.events.push(egui::Event::Key {
            key: Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let out_copy_all = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out_copy_all.platform_output.copied_text, "4 * 7");

        // (B) Substring selected in input ("4 *") -> Ctrl+C copies "4 *"
        editor.request_selection = Some((0, 3));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Copy);
        let out_copy_sub = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out_copy_sub.platform_output.copied_text, "4 *");

        // (C) Simulate drag-selecting the "2² × 7" label under the answer framebox, then pressing Ctrl+C
        let mut factors_rect = None;
        for clipped in &out_copy_sub.shapes {
            if let egui::epaint::Shape::Text(t) = &clipped.shape {
                if t.galley.text() == "2² × 7" {
                    factors_rect = Some(egui::Rect::from_min_size(t.pos, t.galley.size()));
                }
            }
        }
        let rect = factors_rect.expect("Expected '2² × 7' text shape rect");
        let start_pos = egui::pos2(rect.left() + 1.0, rect.center().y);
        let end_pos = egui::pos2(rect.right() - 1.0, rect.center().y);

        // Pointer press at start of "2² × 7"
        let mut raw_press = egui::RawInput::default();
        raw_press.events.push(egui::Event::PointerMoved(start_pos));
        raw_press.events.push(egui::Event::PointerButton {
            pos: start_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_press, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });

        // Pointer drag to end of "2² × 7" and release
        let mut raw_drag = egui::RawInput::default();
        raw_drag.events.push(egui::Event::PointerMoved(end_pos));
        raw_drag.events.push(egui::Event::PointerButton {
            pos: end_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_drag, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });

        assert!(
            has_non_empty_label_selection(&ctx),
            "Expected non-empty label selection after dragging across '2² × 7'"
        );

        // Press Ctrl+C while "2² × 7" label is selected -> must copy "2² × 7", NOT "4 * 7"
        let mut raw_label_copy = egui::RawInput::default();
        raw_label_copy.modifiers = ctrl_mods;
        raw_label_copy.events.push(egui::Event::Key {
            key: Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let out_label_copy = ctx.run(raw_label_copy, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out_label_copy.platform_output.copied_text, "2² × 7");

        // (C2) Without clicking anywhere with the mouse, select a substring in the input ("4 *")
        // while the label selection was active -> stale label selection must be cleared and Ctrl+C copies "4 *"
        editor.request_selection = Some((0, 3));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert!(
            !has_non_empty_label_selection(&ctx),
            "Selecting text in TextEdit must clear stale LabelSelectionState"
        );
        let mut raw_sub_after_label = egui::RawInput::default();
        raw_sub_after_label.modifiers = ctrl_mods;
        raw_sub_after_label.events.push(egui::Event::Key {
            key: Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let out_sub_after_label = ctx.run(raw_sub_after_label, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out_sub_after_label.platform_output.copied_text, "4 *");

        // (D) Single-click without drag on the label (0-length selection primary == secondary)
        // -> has_non_empty_label_selection is false -> Ctrl+C copies whole input ("4 * 7")
        let mut raw_single_click = egui::RawInput::default();
        raw_single_click.events.push(egui::Event::PointerMoved(start_pos));
        raw_single_click.events.push(egui::Event::PointerButton {
            pos: start_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        raw_single_click.events.push(egui::Event::PointerButton {
            pos: start_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_single_click, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert!(
            !has_non_empty_label_selection(&ctx),
            "0-length single click on label must not be treated as a non-empty selection"
        );
        let mut raw_fallback_copy = egui::RawInput::default();
        raw_fallback_copy.modifiers = ctrl_mods;
        raw_fallback_copy.events.push(egui::Event::Key {
            key: Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let out_fallback = ctx.run(raw_fallback_copy, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out_fallback.platform_output.copied_text, "4 * 7");

        // (E) Verify exact rational fraction ("1 / 4" -> "= 0.25", subline "1/4" rendered once under answer)
        // and drag-selecting the main answer "0.25" starting in the empty space to the right of "0.25"
        // and overshooting left onto "=" copies "0.25" cleanly without "="
        editor.input_text = "1 / 4".to_string();
        editor.selection_range = (5, 5);
        editor.trigger_input_wall();
        let out_frac = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });

        let mut ans_025_rect = None;
        let mut ratio_y_positions = Vec::new();
        for clipped in &out_frac.shapes {
            if let egui::epaint::Shape::Text(t) = &clipped.shape {
                let txt = t.galley.text();
                assert!(
                    !txt.contains("[ 1/4 ]"),
                    "Legacy duplicate '[ 1/4 ]' badge must not be rendered, found: {}",
                    txt
                );
                if txt == "0.25" {
                    ans_025_rect = Some(egui::Rect::from_min_size(t.pos, t.galley.size()));
                } else if txt == "1/2²" {
                    ratio_y_positions.push(t.pos.y);
                }
            }
        }
        let ans_rect = ans_025_rect.expect("Expected '0.25' answer text shape");
        assert_eq!(ratio_y_positions.len(), 1, "Expected '1/2²' ratio subline exactly once");
        assert!(
            ratio_y_positions[0] > ans_rect.bottom(),
            "Expected '1/2²' subline y ({}) strictly under '0.25' bottom ({})",
            ratio_y_positions[0],
            ans_rect.bottom()
        );

        // Drag right-to-left starting 80px to the right of "0.25" (empty card area) and overshooting 15px left of "0.25" (over "=")
        let wide_start = egui::pos2(ans_rect.right() + 80.0, ans_rect.center().y);
        let wide_end = egui::pos2(ans_rect.left() - 15.0, ans_rect.center().y);
        let mut raw_wide_press = egui::RawInput::default();
        raw_wide_press.events.push(egui::Event::PointerMoved(wide_start));
        raw_wide_press.events.push(egui::Event::PointerButton {
            pos: wide_start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_wide_press, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        let mut raw_wide_drag = egui::RawInput::default();
        raw_wide_drag.events.push(egui::Event::PointerMoved(wide_end));
        raw_wide_drag.events.push(egui::Event::PointerButton {
            pos: wide_end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_wide_drag, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        let mut raw_wide_copy = egui::RawInput::default();
        raw_wide_copy.modifiers = ctrl_mods;
        raw_wide_copy.events.push(egui::Event::Key {
            key: Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let out_wide_copy = ctx.run(raw_wide_copy, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
            });
        });
        assert_eq!(out_wide_copy.platform_output.copied_text, "0.25");

        // (F) Verify Session History sidebar label selection + Ctrl+C copies the sidebar text
        sidebar.push(
            "3x5 + 2^4".to_string(),
            "31".to_string(),
            "3 × 5 + 2⁴".to_string(),
            420,
        );
        let out_with_sidebar = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.allocate_ui(egui::vec2(500.0, 600.0), |ui| {
                        render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                    });
                    ui.allocate_ui(egui::vec2(320.0, 600.0), |ui| {
                        super::super::sidebar::render_sidebar(
                            ui,
                            &mut sidebar,
                            &mut editor.input_text,
                            &mut editor.is_modified,
                            &palette,
                        );
                    });
                });
            });
        });
        let mut sb_echo_rect = None;
        for clipped in &out_with_sidebar.shapes {
            if let egui::epaint::Shape::Text(t) = &clipped.shape {
                if t.galley.text() == "≡ 3 × 5 + 2⁴" {
                    sb_echo_rect = Some(egui::Rect::from_min_size(t.pos, t.galley.size()));
                }
            }
        }
        let sb_rect = sb_echo_rect.expect("Expected sidebar '≡ 3 × 5 + 2⁴' text shape");
        let sb_start = egui::pos2(sb_rect.left() + 1.0, sb_rect.center().y);
        let sb_end = egui::pos2(sb_rect.right() - 1.0, sb_rect.center().y);

        let mut raw_sb_press = egui::RawInput::default();
        raw_sb_press.events.push(egui::Event::PointerMoved(sb_start));
        raw_sb_press.events.push(egui::Event::PointerButton {
            pos: sb_start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_sb_press, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.allocate_ui(egui::vec2(500.0, 600.0), |ui| {
                        render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                    });
                    ui.allocate_ui(egui::vec2(320.0, 600.0), |ui| {
                        super::super::sidebar::render_sidebar(
                            ui,
                            &mut sidebar,
                            &mut editor.input_text,
                            &mut editor.is_modified,
                            &palette,
                        );
                    });
                });
            });
        });
        let mut raw_sb_drag = egui::RawInput::default();
        raw_sb_drag.events.push(egui::Event::PointerMoved(sb_end));
        raw_sb_drag.events.push(egui::Event::PointerButton {
            pos: sb_end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw_sb_drag, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.allocate_ui(egui::vec2(500.0, 600.0), |ui| {
                        render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                    });
                    ui.allocate_ui(egui::vec2(320.0, 600.0), |ui| {
                        super::super::sidebar::render_sidebar(
                            ui,
                            &mut sidebar,
                            &mut editor.input_text,
                            &mut editor.is_modified,
                            &palette,
                        );
                    });
                });
            });
        });
        let mut raw_sb_copy = egui::RawInput::default();
        raw_sb_copy.modifiers = ctrl_mods;
        raw_sb_copy.events.push(egui::Event::Key {
            key: Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl_mods,
        });
        let out_sb_copy = ctx.run(raw_sb_copy, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.allocate_ui(egui::vec2(500.0, 600.0), |ui| {
                        render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                    });
                    ui.allocate_ui(egui::vec2(320.0, 600.0), |ui| {
                        super::super::sidebar::render_sidebar(
                            ui,
                            &mut sidebar,
                            &mut editor.input_text,
                            &mut editor.is_modified,
                            &palette,
                        );
                    });
                });
            });
        });
        assert_eq!(out_sb_copy.platform_output.copied_text, "≡ 3 × 5 + 2⁴");
    }

    #[test]
    fn test_fx_and_equals_dpi_baseline_midline_alignment_and_zero_italics() {
        let mut vars = FxHashMap::default();
        let mut config = AppConfig::default();
        let palette = crate::ui::get_palette(0);

        for ppp in [1.0_f32, 1.25_f32, 1.5_f32, 1.75_f32, 2.0_f32] {
            let ctx = egui::Context::default();
            crate::ui::setup_fonts(&ctx);
            crate::ui::apply_theme(&ctx, &palette);
            ctx.set_pixels_per_point(ppp);

            let mut editor = EditorState::new();
            let mut sidebar = SidebarState::new();
            sidebar.stashed_draft = Some("draft_expr + 1".to_string());

            for expr in [
                "123232",
                "12345678901234567890123456789012345",
                "123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
            ] {
                editor.input_text = expr.to_string();
                editor.sync_initial_input();
                editor.trigger_input_wall();

                let mut last_out = None;
                for _ in 0..3 {
                    let out = ctx.run(egui::RawInput::default(), |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            ui.horizontal_top(|ui| {
                                ui.allocate_ui(egui::vec2(560.0, 640.0), |ui| {
                                    render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                                });
                                ui.allocate_ui(egui::vec2(320.0, 640.0), |ui| {
                                    super::super::sidebar::render_sidebar(
                                        ui,
                                        &mut sidebar,
                                        &mut editor.input_text,
                                        &mut editor.is_modified,
                                        &palette,
                                    );
                                });
                            });
                        });
                    });
                    last_out = Some(out);
                }
                let out = last_out.unwrap();

                let mut fx_bot_px = None;
                let mut input_bot_px = None;
                let mut eq_mid_px = None;
                let mut ans_mid_px = None;

                fn collect_text_shapes<'a>(
                    shapes: &'a [egui::epaint::ClippedShape],
                    acc: &mut Vec<&'a egui::epaint::TextShape>,
                ) {
                    for cs in shapes {
                        match &cs.shape {
                            egui::epaint::Shape::Text(t) => acc.push(t),
                            egui::epaint::Shape::Vec(v) => {
                                for s in v {
                                    if let egui::epaint::Shape::Text(t) = s {
                                        acc.push(t);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }

                let expected_ans = editor
                    .cached_display
                    .as_ref()
                    .map(|d| d.radix_str.clone())
                    .unwrap_or_default();

                let mut texts = Vec::new();
                collect_text_shapes(&out.shapes, &mut texts);

                for t in texts {
                    for sec in &t.galley.job.sections {
                        assert!(
                            !sec.format.italics,
                            "Zero-italics invariant violated on text '{}' at ppp={ppp}",
                            t.galley.text()
                        );
                    }
                    let txt = t.galley.text();
                    if txt == "fx" {
                        let r0 = &t.galley.rows[0];
                        let bot = r0
                            .visuals
                            .mesh
                            .vertices
                            .iter()
                            .fold(f32::NEG_INFINITY, |a, v| a.max(v.pos.y));
                        fx_bot_px = Some((t.pos.y + bot) * ppp);
                    } else if txt == "=" {
                        let r0 = &t.galley.rows[0];
                        let top = r0
                            .visuals
                            .mesh
                            .vertices
                            .iter()
                            .fold(f32::INFINITY, |a, v| a.min(v.pos.y));
                        let bot = r0
                            .visuals
                            .mesh
                            .vertices
                            .iter()
                            .fold(f32::NEG_INFINITY, |a, v| a.max(v.pos.y));
                        eq_mid_px = Some((t.pos.y + 0.5 * (top + bot)) * ppp);
                    } else if txt == expr && input_bot_px.is_none() {
                        let r0 = &t.galley.rows[0];
                        let mut bot = f32::NEG_INFINITY;
                        let mut flat_bot = f32::NEG_INFINITY;
                        let mut vi = r0.visuals.glyph_vertex_range.start;
                        for g in &r0.glyphs {
                            if !g.uv_rect.is_nothing() {
                                if g.chr.is_ascii_digit() && vi + 3 < r0.visuals.mesh.vertices.len() {
                                    let gb = r0.visuals.mesh.vertices[vi + 2].pos.y;
                                    bot = bot.max(gb);
                                    if matches!(g.chr, '1' | '2' | '4' | '7') {
                                        flat_bot = flat_bot.max(gb);
                                    }
                                }
                                vi += 4;
                            }
                        }
                        let b_use = if flat_bot > f32::NEG_INFINITY { flat_bot } else { bot };
                        input_bot_px = Some((t.pos.y + b_use) * ppp);
                    } else if !expected_ans.is_empty() && txt == expected_ans {
                        let r0 = &t.galley.rows[0];
                        let mut top = f32::INFINITY;
                        let mut bot = f32::NEG_INFINITY;
                        let mut vi = r0.visuals.glyph_vertex_range.start;
                        for g in &r0.glyphs {
                            if !g.uv_rect.is_nothing() {
                                if (g.chr.is_ascii_digit() || g.chr.is_ascii_uppercase())
                                    && vi + 3 < r0.visuals.mesh.vertices.len()
                                {
                                    top = top.min(r0.visuals.mesh.vertices[vi].pos.y);
                                    bot = bot.max(r0.visuals.mesh.vertices[vi + 2].pos.y);
                                }
                                vi += 4;
                            }
                        }
                        ans_mid_px = Some((t.pos.y + 0.5 * (top + bot)) * ppp);
                    }
                }

                let fx_b = fx_bot_px.expect("Missing 'fx' text shape");
                let in_b = input_bot_px.expect("Missing input TextEdit text shape");
                assert!(
                    (fx_b - in_b).abs() <= 0.55,
                    "'fx' baseline ({fx_b:.2} px) misaligned from input '{expr}' baseline ({in_b:.2} px) at ppp={ppp}"
                );

                let eq_m = eq_mid_px.expect("Missing '=' text shape");
                let ans_m = ans_mid_px.expect("Missing answer text shape");
                assert!(
                    (eq_m - ans_m).abs() <= 0.65,
                    "'=' v-center ({eq_m:.2} px) misaligned from answer '{expr}' digit v-center ({ans_m:.2} px) at ppp={ppp}"
                );
            }

            // Also verify empty-input `"fx"` & `"Enter expression..."` hint text alphabetic baseline alignment!
            editor.clear();
            let mut empty_out = None;
            for _ in 0..2 {
                let out = ctx.run(egui::RawInput::default(), |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.allocate_ui(egui::vec2(560.0, 640.0), |ui| {
                            render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                        });
                    });
                });
                empty_out = Some(out);
            }
            let out_e = empty_out.unwrap();
            let mut fx_empty_bot_px = None;
            let mut hint_bot_px = None;
            let mut empty_texts = Vec::new();
            for cs in &out_e.shapes {
                match &cs.shape {
                    egui::epaint::Shape::Text(t) => empty_texts.push(t),
                    egui::epaint::Shape::Vec(v) => {
                        for s in v {
                            if let egui::epaint::Shape::Text(t) = s {
                                empty_texts.push(t);
                            }
                        }
                    }
                    _ => {}
                }
            }
            for t in empty_texts {
                let txt = t.galley.text();
                if txt == "fx" {
                    let r0 = &t.galley.rows[0];
                    let bot = r0
                        .visuals
                        .mesh
                        .vertices
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, v| a.max(v.pos.y));
                    fx_empty_bot_px = Some((t.pos.y + bot) * ppp);
                } else if txt.starts_with("Enter expression") {
                    let r0 = &t.galley.rows[0];
                    let mut vi = r0.visuals.glyph_vertex_range.start;
                    for g in &r0.glyphs {
                        if !g.uv_rect.is_nothing() {
                            if g.chr == 'E' && vi + 3 < r0.visuals.mesh.vertices.len() {
                                let gb = r0.visuals.mesh.vertices[vi + 2].pos.y;
                                hint_bot_px = Some((t.pos.y + gb) * ppp);
                                break;
                            }
                            vi += 4;
                        }
                    }
                }
            }
            let fx_eb = fx_empty_bot_px.expect("Missing 'fx' text shape when input is empty");
            let hint_eb = hint_bot_px.expect("Missing 'Enter expression...' hint text shape");
            assert!(
                (fx_eb - hint_eb).abs() <= 0.55,
                "'fx' baseline ({fx_eb:.2} px) misaligned from 'Enter expression...' hint baseline ({hint_eb:.2} px) at ppp={ppp}"
            );
        }
    }

    #[test]
    fn test_continuum_card_alignment_sidebar_toggle_coords_and_multiradix() {
        let ctx = egui::Context::default();
        super::super::paper_theme::setup_fonts(&ctx);
        let mut config = AppConfig::default();
        let palette = super::super::paper_theme::get_palette(0);
        super::super::paper_theme::apply_theme(&ctx, &palette);

        let mut editor = EditorState::new();
        let mut sidebar = SidebarState::new();
        let mut vars = FxHashMap::default();
        editor.input_text = "255".to_string();
        editor.sync_initial_input();
        editor.trigger_input_wall();

        // 1. Multi-toggle radix invariants: independent toggle, >= 1 active, 0 recalculation epoch bump
        editor.step_input_wall(&vars, true, 180);
        let epoch_before = editor.current_epoch;
        editor.toggle_radix(RadixMode::Dec); // Cannot turn off sole active radix!
        assert_eq!(editor.radix_mask, RADIX_DEC_BIT);

        editor.toggle_radix(RadixMode::Hex);
        editor.toggle_radix(RadixMode::Oct);
        editor.toggle_radix(RadixMode::Bin);
        assert_eq!(editor.radix_mask, 0b1111);
        editor.step_input_wall(&vars, true, 180);
        assert_eq!(
            editor.current_epoch, epoch_before,
            "Multi-radix conversion must NOT advance epoch or recalculate expression!"
        );
        let disp = editor.cached_display.as_ref().unwrap();
        assert_eq!(disp.str_for_radix(RadixMode::Dec), "255");
        assert_eq!(disp.str_for_radix(RadixMode::Hex), "0xFF");
        assert_eq!(disp.str_for_radix(RadixMode::Oct), "0o377");
        assert!(disp.str_for_radix(RadixMode::Bin).contains("1111"));

        // Turning off DEC shifts primary hero radix to HEX while keeping 3 active
        editor.toggle_radix(RadixMode::Dec);
        assert_eq!(editor.radix_mode, RadixMode::Hex);
        editor.set_radix_mask(RADIX_DEC_BIT | RADIX_HEX_BIT | RADIX_BIN_BIT);
        editor.show_diagnostics = true;
        editor.diagnostics_report = Some(run_all_diagnostics());

        for &ppp in &[1.0_f32, 1.25, 1.5, 2.0] {
            ctx.set_pixels_per_point(ppp);
            let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(1060.0, 720.0));
            let mut raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                ..Default::default()
            };
            raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point =
                Some(ppp);

            // Warm-up frame after `set_pixels_per_point` so egui's screen_rect zoom rescale settles
            let _ = ctx.run(raw.clone(), |_| {});

            // Pass A: Session History VISIBLE (`»` button) + Deck Unfolded
            sidebar.visible = true;
            editor.show_function_deck = true;
            let mut hide_screen = egui::Rect::ZERO;
            let _ = ctx.run(raw.clone(), |ctx| {
                hide_screen = ctx.screen_rect();
                let panel_frame = egui::Frame::central_panel(&ctx.style())
                    .fill(palette.background)
                    .inner_margin(egui::Margin::same(10.0));
                egui::CentralPanel::default().frame(panel_frame).show(ctx, |ui| {
                    let pr = ui.max_rect();
                    let pw = pr.width();
                    let ph = pr.height();
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        let sb_w = (pw * 0.33).clamp(300.0, 410.0);
                        let ed_w = pw - sb_w - 17.0;
                        ui.allocate_ui(vec2(ed_w, ph), |ui| {
                            ui.set_min_width(ed_w);
                            ui.set_max_width(ed_w);
                            render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                        });
                        super::super::paper_theme::ui_sym_vline(ui, ph, 8.0, palette.border);
                        ui.allocate_ui(vec2(sb_w, ph), |ui| {
                            ui.set_min_width(sb_w);
                            ui.set_max_width(sb_w);
                            super::super::sidebar::render_sidebar(
                                ui,
                                &mut sidebar,
                                &mut editor.input_text,
                                &mut editor.is_modified,
                                &palette,
                            );
                        });
                    });
                });
            });

            let hide_btn = sidebar
                .last_toggle_btn_rect
                .expect("Missing Hide Session History '»' button rect");
            assert!(editor.last_card_rects.len() >= 6);
            let ref_left = editor.last_card_rects[0].left();
            let ref_right = editor.last_card_rects[0].right();
            for (idx, r) in editor.last_card_rects.iter().enumerate() {
                assert!(
                    (r.left() - ref_left).abs() <= 0.05 && (r.right() - ref_right).abs() <= 0.05,
                    "Card #{idx} edges ({:.2}..{:.2}) misaligned from Top Ribbon ({ref_left:.2}..{ref_right:.2}) at ppp={ppp}",
                    r.left(),
                    r.right()
                );
            }

            // Pass B: Session History HIDDEN (`«` button) + Deck Folded
            sidebar.visible = false;
            editor.show_function_deck = false;
            let mut show_screen = egui::Rect::ZERO;
            let _ = ctx.run(raw, |ctx| {
                show_screen = ctx.screen_rect();
                let panel_frame = egui::Frame::central_panel(&ctx.style())
                    .fill(palette.background)
                    .inner_margin(egui::Margin::same(10.0));
                egui::CentralPanel::default().frame(panel_frame).show(ctx, |ui| {
                    let pw = ui.max_rect().width();
                    let ph = ui.max_rect().height();
                    ui.allocate_ui(vec2(pw, ph), |ui| {
                        ui.set_min_width(pw);
                        ui.set_max_width(pw);
                        render_editor(ui, &mut editor, &mut sidebar, &mut vars, &mut config, &palette);
                    });
                });
            });

            let show_btn = editor
                .last_show_sidebar_btn_rect
                .expect("Missing Show Session History '«' button rect");
            assert_eq!(show_screen, hide_screen);
            assert!(
                (show_btn.right() - hide_btn.right()).abs() <= 0.05
                    && (show_btn.top() - hide_btn.top()).abs() <= 0.05
                    && (show_btn.left() - hide_btn.left()).abs() <= 0.05
                    && (show_btn.bottom() - hide_btn.bottom()).abs() <= 0.05,
                "Show ('«' {:?}) and Hide ('»' {:?}) Session History button coordinates differ at ppp={ppp}!",
                show_btn,
                hide_btn
            );
        }
    }
}

