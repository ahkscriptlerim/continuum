use egui::{RichText, ScrollArea, Ui};
use serde::{Deserialize, Serialize};
use super::paper_theme::{
    show_exact_card, ui_pill_toggle, ui_sym_vline, PaperPalette, HEADER_ROW_HEIGHT,
    SIDEBAR_TOGGLE_BTN_SIZE,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub input: String,
    pub result: String,
    pub canonical_echo: String,
    pub elapsed_nanos: u128,
}

pub struct SidebarState {
    pub history: Vec<HistoryEntry>,
    pub stashed_draft: Option<String>,
    pub visible: bool,
    pub last_toggle_btn_rect: Option<egui::Rect>,
}

impl SidebarState {
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(64),
            stashed_draft: None,
            visible: true,
            last_toggle_btn_rect: None,
        }
    }

    pub fn push(&mut self, input: String, result: String, canonical_echo: String, elapsed_nanos: u128) {
        if input.trim().is_empty() {
            return;
        }
        self.history.insert(
            0,
            HistoryEntry {
                input,
                result,
                canonical_echo,
                elapsed_nanos,
            },
        );
        if self.history.len() > 100 {
            self.history.truncate(100);
        }
    }
}

#[inline]
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut it = s.char_indices();
    if let Some((byte_idx, _)) = it.nth(max_chars) {
        format!("{}…", &s[..byte_idx])
    } else {
        s.to_string()
    }
}

pub fn render_sidebar(
    ui: &mut Ui,
    sidebar: &mut SidebarState,
    current_input: &mut String,
    is_modified: &mut bool,
    palette: &PaperPalette,
) {
    let outer_w = ui.available_width().max(220.0);
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 6.0;

        // Top Header Card — pixel-aligned in height, top inset, and right inset with the Left Header Ribbon
        // so `"»"` (Hide Session History) sits at the exact same `(right, top)` window coordinates as `"«"`!
        show_exact_card(ui, outer_w, palette, |ui, inner_w| {
            ui.allocate_ui_with_layout(
                egui::vec2(inner_w, HEADER_ROW_HEIGHT),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new("Session History")
                                .color(palette.ink_primary)
                                .strong()
                                .size(13.0),
                        )
                        .selectable(false),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 5.0;
                        let hide_resp = ui
                            .add_sized(
                                SIDEBAR_TOGGLE_BTN_SIZE,
                                egui::Button::new(
                                    RichText::new("»")
                                        .color(palette.ink_secondary)
                                        .strong()
                                        .size(12.0),
                                ),
                            )
                            .on_hover_text("Hide Session History Sidebar");
                        sidebar.last_toggle_btn_rect = Some(hide_resp.rect);
                        if hide_resp.clicked() {
                            sidebar.visible = false;
                        }

                        if !sidebar.history.is_empty() {
                            ui_sym_vline(ui, 18.0, 7.0, palette.border);
                            if ui_pill_toggle(ui, false, "Clear", false, palette)
                                .on_hover_text("Clear All Session History Entries")
                                .clicked()
                            {
                                sidebar.history.clear();
                            }
                        }
                    });
                },
            );
        });

        // Law of No Lost Info: Active Draft Banner
        let mut restore_draft = false;
        let mut discard_draft = false;
        if let Some(ref draft) = sidebar.stashed_draft {
            show_exact_card(ui, outer_w, palette, |ui, _inner_w| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 5.0;
                    ui.label(RichText::new("Saved Draft").color(palette.accent).size(11.0).strong());
                    ui_sym_vline(ui, 16.0, 7.0, palette.border);
                    if ui_pill_toggle(ui, true, "Restore", false, palette)
                        .on_hover_text("Restore Saved Draft Into Expression Editor")
                        .clicked()
                    {
                        restore_draft = true;
                    }
                    if ui_pill_toggle(ui, false, "Discard", false, palette)
                        .on_hover_text("Discard Saved Draft")
                        .clicked()
                    {
                        discard_draft = true;
                    }
                });
                let preview = truncate_chars(draft, 54);
                ui.label(RichText::new(preview).color(palette.ghost_ink).size(10.5));
            });
        }
        if restore_draft {
            if let Some(draft) = sidebar.stashed_draft.take() {
                *current_input = draft;
                *is_modified = false;
            }
        } else if discard_draft {
            sidebar.stashed_draft = None;
        }

        if sidebar.history.is_empty() {
            show_exact_card(ui, outer_w, palette, |ui, _inner_w| {
                ui.add_space(18.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("No calculations yet.").color(palette.ghost_ink).size(12.0));
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Press Enter on any expression to pin it here.")
                            .color(palette.ghost_ink)
                            .size(10.5),
                    );
                });
                ui.add_space(18.0);
            });
            return;
        }

        ScrollArea::vertical()
            .id_salt("sidebar_history_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 6.0;
                let mut to_load = None;

                for (idx, entry) in sidebar.history.iter().enumerate() {
                    show_exact_card(ui, outer_w, palette, |ui, inner_w| {
                        // Expression header
                        ui.horizontal(|ui| {
                            let input_label = truncate_chars(&entry.input, 50);
                            let btn = ui.button(
                                RichText::new(input_label)
                                    .color(palette.ink_primary)
                                    .monospace()
                                    .size(12.0),
                            );
                            if btn
                                .on_hover_text("Load Expression Into Editor (Preserves Unsaved Draft)")
                                .clicked()
                            {
                                to_load = Some(idx);
                            }
                        });

                        // Canonical Echo with monospace superscript/subscript kerning
                        if !entry.canonical_echo.is_empty() && entry.canonical_echo != entry.input {
                            let echo_text = format!("≡ {}", truncate_chars(&entry.canonical_echo, 52));
                            let wrap_w = (inner_w - 8.0).max(80.0);
                            let galley = super::editor::cmd_layouter(
                                ui,
                                &echo_text,
                                wrap_w,
                                10.5,
                                palette.ink_secondary,
                            );
                            ui.add(egui::Label::new(galley).selectable(true));
                        }

                        // Result & Latency badge
                        ui.horizontal(|ui| {
                            let res_text = format!("= {}", truncate_chars(&entry.result, 38));
                            let wrap_w = (inner_w - 76.0).max(60.0);
                            let galley = super::editor::cmd_layouter(
                                ui,
                                &res_text,
                                wrap_w,
                                13.0,
                                palette.accent,
                            );
                            ui.add(egui::Label::new(galley).selectable(true));

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let badge_text = format_nanos(entry.elapsed_nanos);
                                ui.label(
                                    RichText::new(format!("⚡ {badge_text}"))
                                        .color(palette.badge_text)
                                        .size(9.5)
                                        .monospace(),
                                )
                                .on_hover_text("Measured Bare-Metal Execution Latency");
                            });
                        });
                    });
                }

                if let Some(idx) = to_load {
                    if *is_modified && !current_input.trim().is_empty() {
                        sidebar.stashed_draft = Some(current_input.clone());
                    }
                    *current_input = sidebar.history[idx].input.clone();
                    *is_modified = false;
                }
            });
    });
}

fn format_nanos(nanos: u128) -> String {
    if nanos < 1_000 {
        format!("{nanos} ns")
    } else if nanos < 1_000_000 {
        format!("{:.1} µs", nanos as f64 / 1_000.0)
    } else {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    }
}
