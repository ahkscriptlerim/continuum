pub struct LogicEngine;

impl LogicEngine {
    /// Returns the canonical bitboard for variable index 0..=5
    pub fn var_bitboard(idx: usize) -> u64 {
        match idx {
            0 => 0xAAAAAAAAAAAAAAAA, // Alternating 1-bit
            1 => 0xCCCCCCCCCCCCCCCC, // Alternating 2-bit
            2 => 0xF0F0F0F0F0F0F0F0, // Alternating 4-bit
            3 => 0xFF00FF00FF00FF00, // Alternating 8-bit
            4 => 0xFFFF0000FFFF0000, // Alternating 16-bit
            5 => 0xFFFFFFFF00000000, // Alternating 32-bit
            _ => 0,
        }
    }

    #[inline(always)]
    pub fn and(a: u64, b: u64) -> u64 { a & b }

    #[inline(always)]
    pub fn or(a: u64, b: u64) -> u64 { a | b }

    #[inline(always)]
    pub fn not(a: u64) -> u64 { !a }

    #[inline(always)]
    pub fn xor(a: u64, b: u64) -> u64 { a ^ b }

    #[inline(always)]
    pub fn implies(a: u64, b: u64) -> u64 { (!a) | b }

    #[inline(always)]
    pub fn iff(a: u64, b: u64) -> u64 { !(a ^ b) }

    pub fn is_tautology(table: u64, var_count: u8) -> bool {
        let mask = Self::active_mask(var_count);
        (table & mask) == mask
    }

    pub fn is_contradiction(table: u64, var_count: u8) -> bool {
        let mask = Self::active_mask(var_count);
        (table & mask) == 0
    }

    pub fn active_mask(var_count: u8) -> u64 {
        let rows = 1u64 << var_count.min(6);
        if rows == 64 {
            !0u64
        } else {
            (1u64 << rows) - 1
        }
    }

    /// Formats a complete truth table for up to 4 variables
    pub fn format_truth_table(table: u64, var_names: &[String]) -> String { // wiring-ok
        use std::fmt::Write;
        let count = var_names.len().min(4);
        let rows = 1 << count;
        let mut out = String::with_capacity(512);

        // Header
        for name in var_names.iter().take(count) {
            let _ = write!(out, "{:^5}|", name);
        }
        out.push_str(" Result\n");
        let dash_len = (count * 6) + 8;
        for _ in 0..dash_len {
            out.push('-');
        }
        out.push('\n');

        // Rows (zero-allocation static LUT slices)
        const CELL_LUT: [&str; 2] = ["  F  |", "  T  |"];
        const RES_LUT: [&str; 2] = ["   F\n", "   T\n"];
        for r in 0..rows {
            for v in 0..count {
                let bit = ((r >> v) & 1) as usize;
                out.push_str(CELL_LUT[bit]);
            }
            let res_bit = ((table >> r) & 1) as usize;
            out.push_str(RES_LUT[res_bit]);
        }

        out
    }

    /// Rephrases an expression into De Morgan equivalent description
    pub fn rephrase_de_morgan(expr_desc: &str) -> String { // wiring-ok
        if expr_desc.contains('v') || expr_desc.contains('∨') || expr_desc.contains('|') {
            format!("De Morgan Dual: ¬(¬A ∧ ¬B)")
        } else if expr_desc.contains('∧') || expr_desc.contains('&') {
            format!("De Morgan Dual: ¬(¬A ∨ ¬B)")
        } else {
            format!("Normalized: {}", expr_desc)
        }
    }
}
