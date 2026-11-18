#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(linker_messages)]

// Hybrid Graphics dGPU Hints (exported via `build.rs` `/EXPORT:`): bind OpenGL/OpenCL contexts
// to discrete NVIDIA Optimus / AMD PowerXpress GPUs rather than integrated UMA adapters.
#[cfg(target_os = "windows")]
#[unsafe(no_mangle)]
pub static NvOptimusEnablement: u32 = 0x0000_0001;

#[cfg(target_os = "windows")]
#[unsafe(no_mangle)]
pub static AmdPowerXpressRequestHighPerformance: i32 = 1;

use eframe::egui;
#[cfg(not(target_os = "android"))]
use mimalloc::MiMalloc;
use rustc_hash::FxHashMap;

#[cfg(not(target_os = "android"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod config;
mod engine;
mod ui;

use config::{load_or_create_ini, save_ini, AppConfig};
use engine::{CalcValue, trigger_afk_calibration};
use ui::{
    apply_theme, get_palette, render_editor, render_sidebar, setup_fonts, ui_sym_vline,
    EditorState, SidebarState,
};

/// Loads the application window/taskbar icon with case-flexible `<projectname>_icon.ico` -> `*.ico`
/// runtime discovery (beside executable, CWD, or `manual release\`), falling back to the compile-time
/// embedded `.ico` resource so single-file portable `Continuum.exe` always renders its crisp icon.
fn load_runtime_window_icon() -> Option<egui::IconData> {
    let slug = env!("APP_SLUG");
    let mut candidate_ico: Option<std::path::PathBuf> = None;

    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        candidate_ico = shared_icon_master::resolve_project_ico_path(exe_dir, slug, None);
    }
    if candidate_ico.is_none() {
        candidate_ico = shared_icon_master::resolve_project_ico_path(
            std::path::Path::new("."),
            slug,
            Some(std::path::Path::new("manual release")),
        );
    }

    if let Some(ico_path) = candidate_ico
        && let Ok(bytes) = std::fs::read(&ico_path)
        && let Ok(img) = shared_icon_master::decode_largest_ico_frame_rgba8(&bytes)
    {
        return Some(egui::IconData {
            rgba: img.rgba,
            width: img.width,
            height: img.height,
        });
    }

    const EMBEDDED_ICO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embedded_app_icon.ico"));
    if !EMBEDDED_ICO.is_empty()
        && let Ok(img) = shared_icon_master::decode_largest_ico_frame_rgba8(EMBEDDED_ICO)
    {
        return Some(egui::IconData {
            rgba: img.rgba,
            width: img.width,
            height: img.height,
        });
    }
    None
}

struct CalculatorApp {
    config: AppConfig,
    editor: EditorState,
    sidebar: SidebarState,
    vars: FxHashMap<String, CalcValue>,
    was_focused: bool,
    frame_count: u32,
    screenshot_path: Option<String>,
    simulate_typing: Option<Vec<char>>,
    typed_index: usize,
    test_memory_add: bool,
    last_min_inner_size: (f32, f32),
}

impl CalculatorApp {
    fn new(cc: &eframe::CreationContext<'_>, mut cfg: AppConfig) -> Self {
        // Configure egui custom fonts & styling (INI already loaded once in `main()`)
        setup_fonts(&cc.egui_ctx);
        let palette = get_palette(cfg.theme_index);
        apply_theme(&cc.egui_ctx, &palette);

        // Check for CLI testing arguments
        let args: Vec<String> = std::env::args().collect();
        let mut screenshot_path = None;
        let mut editor = EditorState::new();
        let mut sidebar = SidebarState::new();

        if let Some(pos) = args.iter().position(|a| a == "--screenshot") {
            let path = args.get(pos + 1).cloned().unwrap_or_else(|| "capture.png".to_string());
            screenshot_path = Some(path);
            editor.input_text = "1232133×2131231231231313322".to_string();
            editor.is_modified = true;
        }

        if let Some(pos) = args.iter().position(|a| a == "--expr") {
            if let Some(expr) = args.get(pos + 1) {
                editor.input_text = expr.clone();
                editor.is_modified = true;
            }
        }

        if args.iter().any(|a| a == "--empty-expr") {
            editor.input_text.clear();
            editor.is_modified = false;
        }

        if let Some(pos) = args.iter().position(|a| a == "--radix") {
            if let Some(r) = args.get(pos + 1) {
                let mut mask = 0u8;
                for tok in r.split([',', '+', '|']) {
                    match tok.trim().to_ascii_lowercase().as_str() {
                        "dec" | "10" => mask |= crate::ui::editor::RADIX_DEC_BIT,
                        "hex" | "16" => mask |= crate::ui::editor::RADIX_HEX_BIT,
                        "oct" | "8" => mask |= crate::ui::editor::RADIX_OCT_BIT,
                        "bin" | "2" => mask |= crate::ui::editor::RADIX_BIN_BIT,
                        "all" => mask = 0b1111,
                        _ => {}
                    }
                }
                editor.set_radix_mask(if mask == 0 {
                    crate::ui::editor::RADIX_DEC_BIT
                } else {
                    mask
                });
            }
        }

        if args.iter().any(|a| a == "--stream") {
            editor.show_cmd_stream = true;
        }

        if args.iter().any(|a| a == "--collapse-deck") {
            editor.show_function_deck = false;
        }

        if args.iter().any(|a| a == "--hide-sidebar") {
            sidebar.visible = false;
        }

        if args.iter().any(|a| a == "--show-diagnostics") {
            editor.show_diagnostics = true;
            editor.diagnostics_report = Some(crate::engine::run_all_diagnostics());
        }

        if let Some(pos) = args.iter().position(|a| a == "--theme") {
            if let Some(t_str) = args.get(pos + 1) {
                if let Ok(idx) = t_str.parse::<usize>() {
                    cfg.theme_index = idx;
                }
            }
        }

        if let Some(pos) = args.iter().position(|a| a == "--mem") {
            if let Some(m_str) = args.get(pos + 1) {
                if let Some((m, s)) = crate::engine::bignum::parse_exact_decimal(m_str) {
                    let cv = CalcValue::normalize_decimal(m, s);
                    editor.memory_sum = cv.to_f64();
                    editor.memory_val = Some(cv);
                } else if let Ok(val) = m_str.parse::<f64>() {
                    editor.memory_sum = val;
                    editor.memory_val = Some(CalcValue::from_f64(val));
                }
            }
        }

        if args.iter().any(|a| a == "--seed-history") {
            sidebar.push(
                "3(5) + 2^4".to_string(),
                "31".to_string(),
                "3\u{2009}×\u{2009}5\u{2009}+\u{2009}2⁴".to_string(),
                1250,
            );
            sidebar.push(
                "\\sqrt{144} * 2".to_string(),
                "24".to_string(),
                "√144\u{2009}×\u{2009}2".to_string(),
                1890,
            );
            sidebar.push(
                "[[1, 2], [3, 4]] * [[2, 0], [1, 2]]".to_string(),
                "[[4.0, 4.0], [10.0, 8.0]]".to_string(),
                "[[1, 2], [3, 4]]\u{2009}×\u{2009}[[2, 0], [1, 2]]".to_string(),
                3400,
            );
        }

        let mut simulate_typing = None;
        if let Some(pos) = args.iter().position(|a| a == "--simulate-typing") {
            if let Some(seq) = args.get(pos + 1) {
                simulate_typing = Some(seq.chars().collect::<Vec<char>>());
                editor.input_text.clear();
                editor.last_computed_input.clear();
            }
        }

        let test_memory_add = args.iter().any(|a| a == "--test-memory-add");

        if !editor.input_text.trim().is_empty() {
            editor.sync_initial_input();
            editor.trigger_input_wall();
        }

        Self {
            config: cfg,
            editor,
            sidebar,
            vars: FxHashMap::default(),
            was_focused: true,
            frame_count: 0,
            screenshot_path,
            simulate_typing,
            typed_index: 0,
            test_memory_add,
            last_min_inner_size: (0.0, 0.0),
        }
    }
}

impl eframe::App for CalculatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame_count += 1;
        let palette = get_palette(self.config.theme_index);

        // Frame-by-frame live typing simulator harness
        if let Some(ref seq) = self.simulate_typing {
            ctx.request_repaint();
            if self.typed_index < seq.len() {
                let ch = seq[self.typed_index];
                self.editor.insert_text(&ch.to_string());
                self.typed_index += 1;
            } else if self.typed_index == seq.len() && self.test_memory_add {
                if self.editor.live_result.is_some() {
                    self.editor.memory_add_current();
                    self.test_memory_add = false;
                }
            } else if (!self.editor.task_in_flight && self.frame_count >= (seq.len() as u32 + 8)) || self.frame_count >= (seq.len() as u32 + 120) {
                if self.screenshot_path.is_some() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                }
            }
        } else {
            // Request quick repaint if waiting for background worker results or pending calculation
            if self.editor.task_in_flight
                || self.editor.pending_calc
                || (self.editor.live_result.is_none()
                    && self.editor.live_error.is_none()
                    && !self.editor.input_text.trim().is_empty())
            {
                ctx.request_repaint_after(std::time::Duration::from_millis(8));
            }

            // Handle automated screenshot request
            if self.screenshot_path.is_some() {
                ctx.request_repaint();
                if self.frame_count >= 5 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                }
            }
        }

        // Interactive F12 snapshot hotkey
        if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            ctx.request_repaint();
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
            self.screenshot_path = Some("capture.png".to_string());
        }

        // Process screenshot events
        for event in &ctx.input(|i| i.raw.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event {
                if let Some(ref path) = self.screenshot_path {
                    let (w, h) = (image.size[0] as u32, image.size[1] as u32);
                    let mut raw_bytes = Vec::with_capacity((w * h * 4) as usize);
                    for p in &image.pixels {
                        raw_bytes.push(p.r());
                        raw_bytes.push(p.g());
                        raw_bytes.push(p.b());
                        raw_bytes.push(p.a());
                    }
                    if let Some(buf) = image::RgbaImage::from_raw(w, h, raw_bytes) {
                        let _ = buf.save(path);
                    }
                    std::process::exit(0);
                }
            }
        }

        // Dynamic responsive minimum window size:
        // - Wider left pane (`>= 600 pt`) so `Continuum | DEC HEX OCT BIN | Mem: <val> | Theme | Self-Test | AC`
        //   always stays fully visible with generous horizontal breathing room.
        // - Smaller minimum window height (`300 pt` vs `540 pt`) when Possibilities Deck is folded!
        let target_min_w = if self.sidebar.visible { 950.0_f32 } else { 620.0_f32 };
        let target_min_h = if self.editor.show_function_deck {
            540.0_f32
        } else {
            300.0_f32
        };
        if (target_min_w, target_min_h) != self.last_min_inner_size {
            self.last_min_inner_size = (target_min_w, target_min_h);
            ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                target_min_w,
                target_min_h,
            )));
        }

        // AFK Window Focus Loss & Live Viewport Geometry Sync (`[EGUI-REENTRANT-LOCK]` safe)
        let (is_focused, inner_rect, outer_rect, is_minimized) = ctx.input(|i| {
            let vp = i.viewport();
            (
                i.focused,
                vp.inner_rect,
                vp.outer_rect,
                vp.minimized.unwrap_or(false),
            )
        });
        if !is_minimized && self.screenshot_path.is_none() {
            if let Some(ir) = inner_rect {
                let w = ir.width();
                let h = ir.height();
                if w >= 580.0 && h >= 280.0 {
                    self.config.window_width = w;
                    self.config.window_height = h;
                }
            }
            if let Some(or) = outer_rect {
                self.config.window_x = or.min.x as i32;
                self.config.window_y = or.min.y as i32;
            }
        }
        if self.was_focused && !is_focused {
            // Window lost focus -> trigger background idle benchmarking
            trigger_afk_calibration();
        }
        self.was_focused = is_focused;

        // Apply theme colors
        apply_theme(ctx, &palette);

        // Quiescent Standby: Zero repaint requests during idle; uniform 10pt outer frame so
        // left/right/top margins and Show (`«`) / Hide (`»`) coordinates match to `0.00 px`!
        let panel_frame = egui::Frame::central_panel(&ctx.style())
            .fill(palette.background)
            .inner_margin(egui::Margin::same(10.0));
        egui::CentralPanel::default().frame(panel_frame).show(ctx, |ui| {
            let panel_rect = ui.max_rect();
            let panel_w = panel_rect.width();
            let panel_h = panel_rect.height();

            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let divider_total_w = 17.0_f32; // 8pt left + 1pt stroke + 8pt right
                let (editor_w, sidebar_w) = if self.sidebar.visible {
                    let sb_w = (panel_w * 0.33)
                        .clamp(300.0, 410.0)
                        .min((panel_w - 580.0 - divider_total_w).max(260.0));
                    let ed_w = (panel_w - sb_w - divider_total_w).max(320.0);
                    (ed_w, sb_w)
                } else {
                    (panel_w, 0.0)
                };

                ui.allocate_ui(egui::vec2(editor_w, panel_h), |ui| {
                    ui.set_min_width(editor_w);
                    ui.set_max_width(editor_w);
                    render_editor(
                        ui,
                        &mut self.editor,
                        &mut self.sidebar,
                        &mut self.vars,
                        &mut self.config,
                        &palette,
                    );
                });

                if self.sidebar.visible {
                    ui_sym_vline(ui, panel_h, 8.0, palette.border);

                    ui.allocate_ui(egui::vec2(sidebar_w, panel_h), |ui| {
                        ui.set_min_width(sidebar_w);
                        ui.set_max_width(sidebar_w);
                        let prev_input = self.editor.input_text.clone();
                        render_sidebar(
                            ui,
                            &mut self.sidebar,
                            &mut self.editor.input_text,
                            &mut self.editor.is_modified,
                            &palette,
                        );
                        if self.editor.input_text != prev_input {
                            let was_mod = self.editor.is_modified;
                            self.editor.record_external_change(prev_input);
                            self.editor.trigger_input_wall();
                            self.editor.is_modified = was_mod;
                            ctx.request_repaint();
                        }
                    });
                }
            });
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        save_ini(&self.config);
        // 120ms anti-zombie exit watchdog: guarantees background OpenCL/WGL/worker threads never linger
        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(120));
            std::process::exit(0);
        });
    }
}

#[cfg(windows)]
pub fn attach_to_interactive_default_desktop() {
    static ATTACHED_DESK: std::sync::atomic::AtomicPtr<core::ffi::c_void> =
        std::sync::atomic::AtomicPtr::new(core::ptr::null_mut());
    #[repr(C)]
    struct StartupInfoW {
        cb: u32,
        lp_reserved: *mut u16,
        lp_desktop: *mut u16,
        lp_title: *mut u16,
        dw_x: u32,
        dw_y: u32,
        dw_x_size: u32,
        dw_y_size: u32,
        dw_x_count_chars: u32,
        dw_y_count_chars: u32,
        dw_fill_attribute: u32,
        dw_flags: u32,
        w_show_window: u16,
        cb_reserved2: u16,
        lp_reserved2: *mut u8,
        h_std_input: *mut core::ffi::c_void,
        h_std_output: *mut core::ffi::c_void,
        h_std_error: *mut core::ffi::c_void,
    }
    #[repr(C)]
    struct ProcessInformation {
        h_process: *mut core::ffi::c_void,
        h_thread: *mut core::ffi::c_void,
        dw_process_id: u32,
        dw_thread_id: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
        fn GetCommandLineW() -> *mut u16;
        fn SetEnvironmentVariableW(lp_name: *const u16, lp_value: *const u16) -> i32;
        fn CreateProcessW(
            lp_application_name: *const u16,
            lp_command_line: *mut u16,
            lp_process_attributes: *const core::ffi::c_void,
            lp_thread_attributes: *const core::ffi::c_void,
            b_inherit_handles: i32,
            dw_creation_flags: u32,
            lp_environment: *const core::ffi::c_void,
            lp_current_directory: *const u16,
            lp_startup_info: *const StartupInfoW,
            lp_process_information: *mut ProcessInformation,
        ) -> i32;
        fn WaitForSingleObject(h_handle: *mut core::ffi::c_void, dw_milliseconds: u32) -> u32;
        fn GetExitCodeProcess(h_process: *mut core::ffi::c_void, lp_exit_code: *mut u32) -> i32;
        fn CloseHandle(h_object: *mut core::ffi::c_void) -> i32;
    }
    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetThreadDesktop(dwThreadId: u32) -> *mut core::ffi::c_void;
        fn GetUserObjectInformationW(
            hObj: *mut core::ffi::c_void,
            nIdx: i32,
            pv: *mut core::ffi::c_void,
            nLen: u32,
            lpn: *mut u32,
        ) -> i32;
        fn OpenDesktopW(
            lpszDesktop: *const u16,
            dwFlags: u32,
            fInherit: i32,
            dwDesiredAccess: u32,
        ) -> *mut core::ffi::c_void;
        fn SetThreadDesktop(hDesktop: *mut core::ffi::c_void) -> i32;
    }
    unsafe {
        let cur = GetThreadDesktop(GetCurrentThreadId());
        let mut buf = [0u16; 32];
        let mut need = 0u32;
        if !cur.is_null()
            && GetUserObjectInformationW(cur, 2, buf.as_mut_ptr().cast(), 64, &mut need) != 0
        {
            const DEF: [u8; 7] = *b"default";
            if buf[7] == 0
                && (0..7).all(|k| (buf[k] as u8).to_ascii_lowercase() == DEF[k] && buf[k] <= 0x7F)
            {
                return; // Pristine-Identity: already on WinSta0\Default -> 0 WGL/ICD disturbance!
            }
        }

        // Pre-WGL CreateProcessW self-relaunch onto `winsta0\default` when spawned on `exebox-*`
        // (bypasses `SetThreadDesktop` ERROR_BUSY=170 and NVIDIA/AMD WGL creation-desktop binding).
        let is_cli_only = std::env::args().any(|a| {
            matches!(
                a.as_str(),
                "--test-suite" | "--diagnostics" | "--bench" | "--benchmark" | "--test" | "-t"
            )
        });
        let wait_child = std::env::args().any(|a| a == "--screenshot");
        if !is_cli_only && std::env::var_os("APP_DESK_RELAUNCHED").is_none() {
            const ENV_KEY: [u16; 20] = [
                b'A' as u16, b'P' as u16, b'P' as u16, b'_' as u16, b'D' as u16, b'E' as u16,
                b'S' as u16, b'K' as u16, b'_' as u16, b'R' as u16, b'E' as u16, b'L' as u16,
                b'A' as u16, b'U' as u16, b'N' as u16, b'C' as u16, b'H' as u16, b'E' as u16,
                b'D' as u16, 0,
            ];
            const ENV_VAL: [u16; 2] = [b'1' as u16, 0];
            let mut desk_w: [u16; 16] = [
                b'w' as u16, b'i' as u16, b'n' as u16, b's' as u16, b't' as u16, b'a' as u16,
                b'0' as u16, b'\\' as u16, b'd' as u16, b'e' as u16, b'f' as u16, b'a' as u16,
                b'u' as u16, b'l' as u16, b't' as u16, 0,
            ];
            let _ = SetEnvironmentVariableW(ENV_KEY.as_ptr(), ENV_VAL.as_ptr());
            let cmd_ptr = GetCommandLineW();
            if !cmd_ptr.is_null() {
                let mut cmd_len = 0usize;
                while *cmd_ptr.add(cmd_len) != 0 && cmd_len < 32_767 {
                    cmd_len += 1;
                }
                let mut cmd_buf = Vec::with_capacity(cmd_len + 1);
                cmd_buf.extend_from_slice(core::slice::from_raw_parts(cmd_ptr, cmd_len));
                cmd_buf.push(0);

                let mut si: StartupInfoW = core::mem::zeroed();
                si.cb = core::mem::size_of::<StartupInfoW>() as u32;
                si.lp_desktop = desk_w.as_mut_ptr();
                let mut pi: ProcessInformation = core::mem::zeroed();
                if CreateProcessW(
                    core::ptr::null(),
                    cmd_buf.as_mut_ptr(),
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    0,
                    core::ptr::null(),
                    core::ptr::null(),
                    &si,
                    &mut pi,
                ) != 0
                {
                    let mut exit_code = 0u32;
                    if wait_child {
                        let _ = WaitForSingleObject(pi.h_process, 30_000);
                        let _ = GetExitCodeProcess(pi.h_process, &mut exit_code);
                    }
                    let _ = CloseHandle(pi.h_thread);
                    let _ = CloseHandle(pi.h_process);
                    std::process::exit(exit_code as i32);
                }
            }
        }

        const NAME: [u16; 8] = [
            b'D' as u16, b'e' as u16, b'f' as u16, b'a' as u16, b'u' as u16, b'l' as u16,
            b't' as u16, 0,
        ];
        let mut h = OpenDesktopW(NAME.as_ptr(), 0, 0, 0x01FF);
        if h.is_null() {
            h = OpenDesktopW(NAME.as_ptr(), 0, 0, 0x1000_0000);
        }
        if !h.is_null() && SetThreadDesktop(h) != 0 {
            ATTACHED_DESK.store(h, std::sync::atomic::Ordering::Release); // Retain; never call CloseDesktop
        }
    }
}

fn main() -> eframe::Result {
    #[cfg(windows)]
    attach_to_interactive_default_desktop();

    #[cfg(all(windows, target_arch = "x86_64"))]
    unsafe {
        // Hardware FTZ (bit 15) + DAZ (bit 6) subnormal flush (`0` ~120-cycle denormal microcode assists)
        #[allow(deprecated)]
        {
            let mxcsr = core::arch::x86_64::_mm_getcsr();
            core::arch::x86_64::_mm_setcsr(mxcsr | 0x8040);
        }
    }

    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn AttachConsole(dwProcessId: u32) -> i32;
        }
        AttachConsole(0xFFFFFFFF);
    }

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| {
        a == "--test-suite"
            || a == "--diagnostics"
            || a == "--bench"
            || a == "--benchmark"
            || a == "--test"
            || a == "-t"
    }) {
        let _ = load_or_create_ini();
        let report = engine::run_all_diagnostics();
        engine::print_diagnostics_report(&report);
        if report.total_failed > 0 {
            std::process::exit(1);
        }
        std::process::exit(0);
    }

    let initial_cfg = load_or_create_ini();
    let is_screenshot = args.iter().any(|a| a == "--screenshot");
    let collapse_deck = args.iter().any(|a| a == "--collapse-deck");
    let hide_sidebar = args.iter().any(|a| a == "--hide-sidebar");
    let (win_w, win_h) = if is_screenshot {
        let mut sw = 1060.0_f32;
        let mut sh = if collapse_deck { 340.0_f32 } else { 720.0_f32 };
        if let Some(pos) = args.iter().position(|a| a == "--win-size")
            && let Some(spec) = args.get(pos + 1)
            && let Some((ws, hs)) = spec.split_once('x')
        {
            if let Ok(w_parsed) = ws.parse::<f32>() {
                sw = w_parsed;
            }
            if let Ok(h_parsed) = hs.parse::<f32>() {
                sh = h_parsed;
            }
        }
        (sw, sh)
    } else {
        (initial_cfg.window_width, initial_cfg.window_height)
    };

    let app_title = env!("APP_DISPLAY_NAME");
    let init_min_w = if hide_sidebar { 620.0 } else { 950.0 };
    let init_min_h = if collapse_deck { 300.0 } else { 540.0 };
    let mut viewport = egui::ViewportBuilder::default()
        .with_title(app_title)
        .with_inner_size([win_w, win_h])
        .with_min_inner_size([init_min_w, init_min_h])
        .with_decorations(true)
        .with_transparent(false);
    if let Some(icon_data) = load_runtime_window_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon_data));
    }
    if !is_screenshot
        && initial_cfg.window_x != -1
        && initial_cfg.window_y != -1
        && initial_cfg.window_x != i32::MIN
        && initial_cfg.window_y != i32::MIN
    {
        viewport = viewport.with_position([
            initial_cfg.window_x as f32,
            initial_cfg.window_y as f32,
        ]);
    }

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        app_title,
        native_options,
        Box::new(move |cc| Ok(Box::new(CalculatorApp::new(cc, initial_cfg)))),
    )
}
