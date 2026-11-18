use egui::{Color32, Stroke};

#[derive(Clone, Copy, Debug)]
pub struct PaperPalette {
    pub name: &'static str,
    pub background: Color32,
    pub surface: Color32,
    pub border: Color32,
    pub ink_primary: Color32,
    pub ink_secondary: Color32,
    pub ghost_ink: Color32,
    pub accent: Color32,
    pub badge_bg: Color32,
    pub badge_text: Color32,
}

pub const PALETTES: [PaperPalette; 4] = [
    // 0: Classic Academic Manuscript Cream (Warm Old-Paper & Soft Non-Flashing Cream)
    PaperPalette {
        name: "Academic Cream",
        background: Color32::from_rgb(0xE8, 0xE0, 0xCE),
        surface: Color32::from_rgb(0xF4, 0xEF, 0xE3),
        border: Color32::from_rgb(0xCE, 0xC4, 0xAE),
        ink_primary: Color32::from_rgb(0x22, 0x1F, 0x1B),
        ink_secondary: Color32::from_rgb(0x5A, 0x53, 0x47),
        ghost_ink: Color32::from_rgb(0x82, 0x79, 0x6A),
        accent: Color32::from_rgb(0x8B, 0x26, 0x35),
        badge_bg: Color32::from_rgb(0xDD, 0xD3, 0xBE),
        badge_text: Color32::from_rgb(0x4A, 0x3F, 0x31),
    },
    // 1: Blueprint Navy
    PaperPalette {
        name: "Blueprint Navy",
        background: Color32::from_rgb(0x16, 0x1D, 0x2C),
        surface: Color32::from_rgb(0x1F, 0x28, 0x3D),
        border: Color32::from_rgb(0x2D, 0x3A, 0x54),
        ink_primary: Color32::from_rgb(0xEB, 0xF1, 0xFA),
        ink_secondary: Color32::from_rgb(0x94, 0xA3, 0xB8),
        ghost_ink: Color32::from_rgb(0x78, 0x88, 0xA0),
        accent: Color32::from_rgb(0x38, 0xBD, 0xF8),
        badge_bg: Color32::from_rgb(0x26, 0x33, 0x4D),
        badge_text: Color32::from_rgb(0x7D, 0xD3, 0xFC),
    },
    // 2: Sepia Manuscript
    PaperPalette {
        name: "Sepia Library",
        background: Color32::from_rgb(0x2B, 0x22, 0x1C),
        surface: Color32::from_rgb(0x38, 0x2D, 0x26),
        border: Color32::from_rgb(0x4E, 0x3E, 0x35),
        ink_primary: Color32::from_rgb(0xF3, 0xE7, 0xD9),
        ink_secondary: Color32::from_rgb(0xB8, 0xA4, 0x94),
        ghost_ink: Color32::from_rgb(0x9E, 0x8C, 0x7E),
        accent: Color32::from_rgb(0xD9, 0x82, 0x3B),
        badge_bg: Color32::from_rgb(0x44, 0x36, 0x2D),
        badge_text: Color32::from_rgb(0xE8, 0xB8, 0x8D),
    },
    // 3: Obsidian Drafting (OLED Dark)
    PaperPalette {
        name: "Obsidian OLED",
        background: Color32::from_rgb(0x10, 0x11, 0x14),
        surface: Color32::from_rgb(0x18, 0x1A, 0x1F),
        border: Color32::from_rgb(0x26, 0x2A, 0x33),
        ink_primary: Color32::from_rgb(0xF0, 0xF2, 0xF5),
        ink_secondary: Color32::from_rgb(0x8B, 0x94, 0x9E),
        ghost_ink: Color32::from_rgb(0x76, 0x82, 0x94),
        accent: Color32::from_rgb(0x00, 0xE6, 0x76),
        badge_bg: Color32::from_rgb(0x20, 0x24, 0x2C),
        badge_text: Color32::from_rgb(0x69, 0xF0, 0xAE),
    },
];

pub fn get_palette(index: usize) -> PaperPalette {
    PALETTES[index % PALETTES.len()]
}

pub fn apply_theme(ctx: &egui::Context, palette: &PaperPalette) {
    let mut visuals = if palette.name == "Academic Cream" {
        egui::Visuals::light()
    } else {
        egui::Visuals::dark()
    };
    visuals.panel_fill = palette.background;
    visuals.window_fill = palette.surface;
    visuals.window_stroke = Stroke::new(1.0_f32, palette.border);
    visuals.extreme_bg_color = palette.surface; // TextEdit background matches paper surface!
    
    visuals.widgets.noninteractive.bg_fill = palette.surface;
    visuals.widgets.noninteractive.weak_bg_fill = palette.surface;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, palette.border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, palette.ink_primary);
    visuals.widgets.noninteractive.rounding = egui::Rounding::same(6.0);
    
    visuals.widgets.inactive.bg_fill = palette.background;
    visuals.widgets.inactive.weak_bg_fill = palette.background;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, palette.border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, palette.ink_secondary);
    visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    
    visuals.widgets.hovered.bg_fill = palette.badge_bg;
    visuals.widgets.hovered.weak_bg_fill = palette.badge_bg;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, palette.ink_secondary);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, palette.ink_primary);
    visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    
    visuals.widgets.active.bg_fill = palette.badge_bg;
    visuals.widgets.active.weak_bg_fill = palette.badge_bg;
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, palette.accent);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, palette.accent);
    visuals.widgets.active.rounding = egui::Rounding::same(6.0);

    visuals.widgets.open.bg_fill = palette.badge_bg;
    visuals.widgets.open.weak_bg_fill = palette.badge_bg;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, palette.ink_secondary);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0_f32, palette.ink_primary);
    visuals.widgets.open.rounding = egui::Rounding::same(6.0);
    
    visuals.selection.bg_fill = palette.accent;
    visuals.selection.stroke = Stroke::new(1.0_f32, palette.background);
    
    ctx.set_visuals(visuals);
    ctx.style_mut(|style| {
        style.interaction.selectable_labels = true;
        style.interaction.multi_widget_text_select = true;
        style.spacing.scroll = egui::style::ScrollStyle::floating();
    });
}

pub const CARD_INNER_MARGIN: f32 = 8.0;
pub const CARD_STROKE_WIDTH: f32 = 1.0;
pub const HEADER_ROW_HEIGHT: f32 = 24.0;
pub const SIDEBAR_TOGGLE_BTN_SIZE: [f32; 2] = [24.0, 22.0];

/// Renders a rounded Paper card locked to exact `outer_w` (`0.00 px` right-edge drift across cards).
#[inline]
pub fn show_exact_card<R>(
    ui: &mut egui::Ui,
    outer_w: f32,
    palette: &PaperPalette,
    add_contents: impl FnOnce(&mut egui::Ui, f32) -> R,
) -> egui::InnerResponse<R> {
    let inset = 2.0 * CARD_INNER_MARGIN;
    let inner_w = (outer_w - inset).max(80.0);
    egui::Frame::none()
        .fill(palette.surface)
        .stroke(Stroke::new(CARD_STROKE_WIDTH, palette.border))
        .rounding(egui::Rounding::same(6.0))
        .inner_margin(egui::Margin::same(CARD_INNER_MARGIN))
        .show(ui, |ui| {
            ui.set_max_width(inner_w);
            ui.set_width(inner_w);
            let ret = add_contents(ui, inner_w);
            // Guarantee exact `inner_w` even if an inner widget was narrower
            let target_right = ui.min_rect().left() + inner_w;
            if ui.min_rect().right() < target_right {
                ui.expand_to_include_x(target_right);
            }
            ret
        })
}

/// Renders a pixel-snapped vertical rule with identical optical spacing (`half_gap`) on both sides,
/// compensating for `ui.spacing().item_spacing.x` so `"BIN | Mem"` has symmetric left/right gutters.
#[inline]
pub fn ui_sym_vline(ui: &mut egui::Ui, height: f32, half_gap: f32, color: Color32) {
    let sp_x = ui.spacing().item_spacing.x;
    let alloc_w = (2.0 * half_gap + 1.0 - 2.0 * sp_x).max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(alloc_w, height), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let ppp = ui.ctx().pixels_per_point();
        let inv_ppp = 1.0 / ppp;
        let x = (rect.center().x * ppp).round() * inv_ppp;
        ui.painter().line_segment(
            [
                egui::pos2(x, rect.top() + 2.0),
                egui::pos2(x, (rect.bottom() - 2.0).max(rect.top() + 2.0)),
            ],
            Stroke::new(inv_ppp.max(1.0), color),
        );
    }
}

/// Renders a symmetric horizontal rule inside a card (`v_gap` above and `v_gap` below a pixel-snapped stroke).
#[inline]
pub fn ui_sym_hline(ui: &mut egui::Ui, width: f32, v_gap: f32, color: Color32) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(width, 2.0 * v_gap + 1.0), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let ppp = ui.ctx().pixels_per_point();
        let inv_ppp = 1.0 / ppp;
        let y = (rect.center().y * ppp).round() * inv_ppp;
        ui.painter().line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            Stroke::new(inv_ppp.max(1.0), color),
        );
    }
}

/// High-contrast symmetric pill button (`badge_bg` fill + `accent` text/border when active,
/// never painting `accent` text on `accent` rect fill, with equal `7.0 pt` left/right padding).
pub fn ui_pill_toggle(
    ui: &mut egui::Ui,
    active: bool,
    text: &str,
    monospace: bool,
    palette: &PaperPalette,
) -> egui::Response {
    let font_id = egui::FontId::new(
        11.0,
        if monospace {
            egui::FontFamily::Monospace
        } else {
            egui::FontFamily::Proportional
        },
    );
    let text_color = if active {
        palette.accent
    } else {
        palette.ink_secondary
    };
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_owned(), font_id, text_color));
    let pad_x = 7.0_f32;
    let desired = egui::vec2((galley.size().x + 2.0 * pad_x).ceil(), 22.0);
    let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let hovered = resp.hovered();
        let bg = if active {
            palette.badge_bg
        } else if hovered {
            palette.badge_bg.gamma_multiply(0.55)
        } else {
            Color32::TRANSPARENT
        };
        let stroke = if active {
            Stroke::new(1.2_f32, palette.accent)
        } else if hovered {
            Stroke::new(1.0_f32, palette.ink_secondary)
        } else {
            Stroke::new(1.0_f32, palette.border)
        };
        ui.painter()
            .rect(rect, egui::Rounding::same(5.0), bg, stroke);
        let draw_ink = if active {
            palette.accent
        } else if hovered {
            palette.ink_primary
        } else {
            palette.ink_secondary
        };
        let pos = egui::pos2(
            rect.center().x - 0.5 * galley.size().x,
            rect.center().y - 0.5 * galley.size().y,
        );
        ui.painter().galley(pos, galley, draw_ink);
    }
    resp
}

pub fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    #[cfg(windows)]
    {
        if let Ok(consolas_bytes) = std::fs::read(r"C:\Windows\Fonts\consola.ttf") {
            fonts.font_data.insert(
                "consolas".to_owned(),
                egui::FontData::from_owned(consolas_bytes),
            );
            if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                let insert_idx = mono.len().min(1);
                mono.insert(insert_idx, "consolas".to_owned());
            }
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push("consolas".to_owned());
        }
        if let Ok(font_bytes) = std::fs::read(r"C:\Windows\Fonts\seguisym.ttf") {
            fonts.font_data.insert(
                "segoe_ui_symbol".to_owned(),
                egui::FontData::from_owned(font_bytes),
            );
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push("segoe_ui_symbol".to_owned());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push("segoe_ui_symbol".to_owned());
        }
    }
    #[cfg(unix)]
    {
        const UNIX_MONO_CANDIDATES: &[&str] = &[
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ];
        if let Some(mono_bytes) = UNIX_MONO_CANDIDATES
            .iter()
            .copied()
            .find(|p| std::path::Path::new(p).is_file())
            .and_then(|p| std::fs::read(p).ok())
        {
            fonts.font_data.insert(
                "unix_mono_sub".to_owned(),
                egui::FontData::from_owned(mono_bytes),
            );
            if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                let insert_idx = mono.len().min(1);
                mono.insert(insert_idx, "unix_mono_sub".to_owned());
            }
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push("unix_mono_sub".to_owned());
        }
        const UNIX_SYM_CANDIDATES: &[&str] = &[
            "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
            "/usr/share/fonts/TTF/NotoSansSymbols2-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/System/Library/Fonts/Apple Symbols.ttf",
        ];
        if let Some(sym_bytes) = UNIX_SYM_CANDIDATES
            .iter()
            .copied()
            .find(|p| std::path::Path::new(p).is_file())
            .and_then(|p| std::fs::read(p).ok())
        {
            fonts.font_data.insert(
                "unix_symbols".to_owned(),
                egui::FontData::from_owned(sym_bytes),
            );
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push("unix_symbols".to_owned());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push("unix_symbols".to_owned());
        }
    }
    ctx.set_fonts(fonts);
    ctx.set_zoom_factor(1.0);
    ctx.options_mut(|o| {
        o.zoom_with_keyboard = false;
    });
    ctx.style_mut(|style| {
        style.interaction.selectable_labels = true;
        style.interaction.multi_widget_text_select = true;
    });
    ctx.on_end_pass(
        "os_clipboard_sync",
        std::sync::Arc::new(|ctx| {
            let copied = ctx.output(|o| o.copied_text.clone());
            if !copied.is_empty() {
                super::editor::mirror_clipboard_text(&copied);
                #[cfg(windows)]
                {
                    let _ = super::editor::win_clipboard::set_text(&copied);
                }
            }
        }),
    );
}

