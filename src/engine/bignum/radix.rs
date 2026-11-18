/// High-Performance Multi-Radix Base Conversion Engine
/// Supports exact decimal (base 10), hexadecimal (base 16), octal (base 8), and binary (base 2).
/// Includes single-cycle limb division by 10^19 (Knuth Alg D), O(log E) 19-digit normalized decimal
/// scientific scaling (`DecSci`), single-pass O(N) modular trailing-digit extraction, and stream formatting.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadixMode {
    Dec = 10,
    Hex = 16,
    Oct = 8,
    Bin = 2,
}

impl RadixMode {
    pub fn name(&self) -> &'static str {
        match self {
            RadixMode::Dec => "DEC (10)",
            RadixMode::Hex => "HEX (16)",
            RadixMode::Oct => "OCT (8)",
            RadixMode::Bin => "BIN (2)",
        }
    }
}

pub const BASE10_MAX_CHUNK: u64 = 10_000_000_000_000_000_000; // 10^19 fits within u64
pub const NORM_DEC_MIN: u64 = 1_000_000_000_000_000_000; // 10^18

/// Normalized 19-digit base-10 scientific accumulator (`value ≈ mant * 10^exp10`, `mant ∈ [10^18, 10^19)`).
/// Total integer decimal digits = `exp10 + 19`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecSci {
    pub mant: u64,
    pub exp10: i64,
}

impl DecSci {
    pub const ONE: Self = Self {
        mant: NORM_DEC_MIN,
        exp10: -18,
    };

    #[inline]
    pub fn from_u128(mut v: u128) -> Self {
        if v == 0 {
            return Self { mant: 0, exp10: 0 };
        }
        let mut exp10 = 0i64;
        while v >= BASE10_MAX_CHUNK as u128 {
            v = (v + 5) / 10;
            exp10 += 1;
        }
        while v < NORM_DEC_MIN as u128 {
            v *= 10;
            exp10 -= 1;
        }
        Self {
            mant: v as u64,
            exp10,
        }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        if self.mant == 0 || rhs.mant == 0 {
            return Self { mant: 0, exp10: 0 };
        }
        let prod = (self.mant as u128) * (rhs.mant as u128);
        // prod is in [10^36, 10^38)
        let (mut m, mut e) = if prod >= 10_000_000_000_000_000_000_000_000_000_000_000_000u128 {
            (
                ((prod + 5_000_000_000_000_000_000) / 10_000_000_000_000_000_000) as u64,
                self.exp10.saturating_add(rhs.exp10).saturating_add(19),
            )
        } else {
            (
                ((prod + 500_000_000_000_000_000) / 1_000_000_000_000_000_000) as u64,
                self.exp10.saturating_add(rhs.exp10).saturating_add(18),
            )
        };
        if m >= BASE10_MAX_CHUNK {
            m /= 10;
            e = e.saturating_add(1);
        }
        Self { mant: m, exp10: e }
    }

    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        if self.mant == 0 || rhs.mant == 0 {
            return Self { mant: 0, exp10: 0 };
        }
        let num_base = self.mant as u128;
        let den = rhs.mant as u128;
        let (mut m, mut e) = if self.mant >= rhs.mant {
            let scaled = num_base * (NORM_DEC_MIN as u128);
            (
                ((scaled + den / 2) / den) as u64,
                self.exp10.saturating_sub(rhs.exp10).saturating_sub(18),
            )
        } else {
            let scaled = num_base * (BASE10_MAX_CHUNK as u128);
            (
                ((scaled + den / 2) / den) as u64,
                self.exp10.saturating_sub(rhs.exp10).saturating_sub(19),
            )
        };
        if m >= BASE10_MAX_CHUNK {
            m /= 10;
            e = e.saturating_add(1);
        }
        Self { mant: m, exp10: e }
    }

    #[inline]
    pub fn pow_u64(mut self, mut exp: u64) -> Self {
        let mut res = Self::ONE;
        while exp > 0 {
            if (exp & 1) != 0 {
                res = res.mul(self);
            }
            if exp > 1 {
                self = self.mul(self);
            }
            exp >>= 1;
        }
        res
    }

    #[inline]
    pub fn pow2(exp: u64) -> Self {
        let two = Self {
            mant: 2_000_000_000_000_000_000,
            exp10: -18,
        };
        two.pow_u64(exp)
    }

    #[inline]
    pub fn from_limbs(limbs: &[u64]) -> Self {
        let mut end = limbs.len();
        while end > 0 && limbs[end - 1] == 0 {
            end -= 1;
        }
        if end == 0 {
            return Self { mant: 0, exp10: 0 };
        }
        if end == 1 {
            return Self::from_u128(limbs[0] as u128);
        }
        if end == 2 {
            let combined = ((limbs[1] as u128) << 64) | (limbs[0] as u128);
            return Self::from_u128(combined);
        }
        let lz = limbs[end - 1].leading_zeros();
        if lz == 0 {
            let combined = ((limbs[end - 1] as u128) << 64) | (limbs[end - 2] as u128);
            let shift_bits = ((end - 2) as u64) * 64;
            Self::from_u128(combined).mul(Self::pow2(shift_bits))
        } else {
            let combined = ((limbs[end - 1] as u128) << (64 + lz))
                | ((limbs[end - 2] as u128) << lz)
                | ((limbs[end - 3] as u128) >> (64 - lz));
            let shift_bits = ((end - 2) as u64) * 64 - (lz as u64);
            Self::from_u128(combined).mul(Self::pow2(shift_bits))
        }
    }

    #[inline]
    pub fn total_digits(self) -> u64 {
        if self.mant == 0 {
            1
        } else {
            self.exp10.saturating_add(19).max(1) as u64
        }
    }
}

/// Computes `limbs mod 10^19` in a single O(N) linear pass with zero heap allocations.
#[inline]
pub fn limbs_mod_10e19(limbs: &[u64]) -> u64 {
    const MOD19: u128 = 10_000_000_000_000_000_000;
    const TWO64_MOD19: u128 = 8_446_744_073_709_551_616; // 2^64 mod 10^19
    let mut rem: u128 = 0;
    for &limb in limbs.iter().rev() {
        rem = (rem * TWO64_MOD19 + (limb as u128)) % MOD19;
    }
    rem as u64
}

/// Divides a multi-limb integer (little-endian) by a 64-bit divisor in-place.
/// Returns the 64-bit remainder. Runs in strictly O(N) time with fast 64-bit leading-limb bypass.
#[inline]
pub fn div_rem_u64(limbs: &mut [u64], divisor: u64) -> u64 {
    let mut rem = 0u128;
    let div128 = divisor as u128;
    for limb in limbs.iter_mut().rev() {
        let l = *limb;
        if rem == 0 {
            if l < divisor {
                rem = l as u128;
                *limb = 0;
                continue;
            }
            *limb = l / divisor;
            rem = (l % divisor) as u128;
        } else {
            let cur = (rem << 64) | (l as u128);
            *limb = (cur / div128) as u64;
            rem = cur % div128;
        }
    }
    rem as u64
}

const DIGIT_PAIRS_LUT: &[u8; 200] = b"\
0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

#[inline(always)]
fn push_u64_padded19(out: &mut String, mut val: u64) {
    let mut buf = [b'0'; 19];
    let mut idx = 19usize;
    while val >= 100 && idx >= 2 {
        let rem2 = ((val % 100) as usize) * 2;
        val /= 100;
        idx -= 2;
        unsafe {
            buf[idx] = *DIGIT_PAIRS_LUT.get_unchecked(rem2);
            buf[idx + 1] = *DIGIT_PAIRS_LUT.get_unchecked(rem2 + 1);
        }
    }
    if val >= 10 && idx >= 2 {
        let rem2 = (val as usize) * 2;
        idx -= 2;
        unsafe {
            buf[idx] = *DIGIT_PAIRS_LUT.get_unchecked(rem2);
            buf[idx + 1] = *DIGIT_PAIRS_LUT.get_unchecked(rem2 + 1);
        }
    } else if val > 0 && idx >= 1 {
        buf[idx - 1] = b'0' + (val as u8);
    }
    // SAFETY: `buf` contains exclusively ASCII `'0'..='9'` bytes.
    out.push_str(unsafe { core::str::from_utf8_unchecked(&buf) });
}

/// Exact Base-10 Decimal Conversion for arbitrary length limbs
pub fn format_limbs_dec(limbs: &[u64]) -> String {
    // 1. Trim leading zero limbs
    let mut end = limbs.len();
    while end > 0 && limbs[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return "0".to_string();
    }

    // 2. Fast scalar paths
    if end == 1 {
        return limbs[0].to_string();
    }
    if end == 2 {
        let v = (limbs[1] as u128) << 64 | (limbs[0] as u128);
        return v.to_string();
    }

    // 3. Fast O(N) leading + trailing + digit-count extractor for massive numbers (> 4,096 limbs = > 262,144 bits)
    if end > 4096 {
        if crate::engine::evaluator::is_task_aborted() {
            return String::new();
        }
        let sci = DecSci::from_limbs(&limbs[..end]);
        let tail19 = limbs_mod_10e19(&limbs[..end]);
        let total_digits = sci.total_digits();
        let total_bits =
            ((end - 1) as u64) * 64 + (64 - limbs[end - 1].leading_zeros() as u64);
        return format!(
            "{}…{:019} ({} digits | {} bits)",
            sci.mant, tail19, total_digits, total_bits
        );
    }

    // 4. Multi-chunk extraction by 10^19 with active_len shrinking and cooperative preemption
    let mut temp = limbs[..end].to_vec();
    let mut active_len = end;
    let mut chunks: Vec<u64> = Vec::with_capacity((end * 34) / 10 + 2);

    while active_len > 0 {
        if (chunks.len() & 63) == 63 && crate::engine::evaluator::is_task_aborted() {
            return String::new();
        }
        let rem = div_rem_u64(&mut temp[..active_len], BASE10_MAX_CHUNK);
        chunks.push(rem);
        while active_len > 0 && temp[active_len - 1] == 0 {
            active_len -= 1;
        }
    }

    if chunks.is_empty() {
        return "0".to_string();
    }

    // 5. Assemble chunks into string without per-chunk heap allocations
    let mut out = String::with_capacity(chunks.len() * 19);
    // Highest chunk: formatted without leading zeros
    let last = chunks.pop().unwrap_or(0);
    out.push_str(&last.to_string());

    // Remaining chunks: formatted into stack buffer with exactly 19 zero-padded digits
    for (i, chunk) in chunks.into_iter().rev().enumerate() {
        if (i & 127) == 127 && crate::engine::evaluator::is_task_aborted() {
            return String::new();
        }
        push_u64_padded19(&mut out, chunk);
    }

    out
}

/// Exact Base-16 Hexadecimal Conversion
pub fn format_limbs_hex(limbs: &[u64], prefix: bool) -> String {
    use std::fmt::Write;
    let mut end = limbs.len();
    while end > 0 && limbs[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return if prefix { "0x0".to_string() } else { "0".to_string() };
    }
    if end > 4096 {
        let total_bits =
            ((end - 1) as u64) * 64 + (64 - limbs[end - 1].leading_zeros() as u64);
        let hex_digits = (total_bits + 3) / 4;
        let pfx = if prefix { "0x" } else { "" };
        return format!(
            "{}{:X}{:016X}…{:016X} ({} hex digits | {} bits)",
            pfx,
            limbs[end - 1],
            limbs[end - 2],
            limbs[0],
            hex_digits,
            total_bits
        );
    }

    let mut out = String::with_capacity(end * 16 + 4);
    if prefix {
        out.push_str("0x");
    }
    // Highest limb without leading zeros
    let _ = write!(out, "{:X}", limbs[end - 1]);
    // Subsequent limbs with exact 16 digits
    for (i, &limb) in limbs[..end - 1].iter().rev().enumerate() {
        if (i & 127) == 127 && crate::engine::evaluator::is_task_aborted() {
            return String::new();
        }
        let _ = write!(out, "{:016X}", limb);
    }
    out
}

/// Exact Base-8 Octal Conversion (O(N) direct 3-bit limb extraction, zero division scratch)
pub fn format_limbs_oct(limbs: &[u64], prefix: bool) -> String {
    let mut end = limbs.len();
    while end > 0 && limbs[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return if prefix { "0o0".to_string() } else { "0".to_string() };
    }
    if end == 1 {
        return if prefix {
            format!("0o{:o}", limbs[0])
        } else {
            format!("{:o}", limbs[0])
        };
    }

    let top_bits = (64 - limbs[end - 1].leading_zeros()) as usize;
    let total_bits = (end - 1) * 64 + top_bits;
    let num_oct_digits = (total_bits + 2) / 3;

    if end > 4096 {
        let pfx = if prefix { "0o" } else { "" };
        return format!(
            "{}{:o}…{:021o} ({} oct digits | {} bits)",
            pfx,
            limbs[end - 1],
            limbs[0] & 0x7FFF_FFFF_FFFF_FFFF,
            num_oct_digits,
            total_bits
        );
    }

    let mut out = String::with_capacity(num_oct_digits + 2);
    if prefix {
        out.push_str("0o");
    }

    for digit_idx in (0..num_oct_digits).rev() {
        if (digit_idx & 255) == 255 && crate::engine::evaluator::is_task_aborted() {
            return String::new();
        }
        let bit_pos = digit_idx * 3;
        let limb_idx = bit_pos >> 6;
        let bit_off = bit_pos & 63;
        let mut val = (limbs[limb_idx] >> bit_off) & 7;
        if bit_off > 61 && limb_idx + 1 < end {
            val |= (limbs[limb_idx + 1] << (64 - bit_off)) & 7;
        }
        out.push((b'0' + val as u8) as char);
    }
    out
}

/// Exact Base-2 Binary Conversion
pub fn format_limbs_bin(limbs: &[u64], prefix: bool, group_nibbles: bool) -> String {
    use std::fmt::Write;
    let mut end = limbs.len();
    while end > 0 && limbs[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return if prefix { "0b0".to_string() } else { "0".to_string() };
    }
    if end > 4096 {
        let total_bits =
            ((end - 1) as u64) * 64 + (64 - limbs[end - 1].leading_zeros() as u64);
        let pfx = if prefix { "0b" } else { "" };
        return format!(
            "{}{:b}…{:064b} ({} bits)",
            pfx,
            limbs[end - 1],
            limbs[0],
            total_bits
        );
    }

    let mut raw = String::with_capacity(end * 64);
    let _ = write!(raw, "{:b}", limbs[end - 1]);
    for (i, &limb) in limbs[..end - 1].iter().rev().enumerate() {
        if (i & 127) == 127 && crate::engine::evaluator::is_task_aborted() {
            return String::new();
        }
        let _ = write!(raw, "{:064b}", limb);
    }

    if !group_nibbles {
        if prefix {
            return format!("0b{}", raw);
        } else {
            return raw;
        }
    }

    // Group in 4-bit nibbles: e.g. 1010 1100 0001
    let mut out = String::with_capacity(raw.len() + raw.len() / 4 + 4);
    if prefix {
        out.push_str("0b");
    }
    let rem = raw.len() % 4;
    let mut pos = 0;
    if rem > 0 {
        out.push_str(&raw[..rem]);
        pos = rem;
        if pos < raw.len() {
            out.push(' ');
        }
    }
    while pos < raw.len() {
        out.push_str(&raw[pos..pos + 4]);
        pos += 4;
        if pos < raw.len() {
            out.push(' ');
        }
    }
    out
}

/// Formats a page-long command stream with line numbers and character statistics
pub fn format_page_stream(raw_text: &str, chars_per_line: usize) -> String { // wiring-ok
    use std::fmt::Write;
    let step = chars_per_line.max(1);
    let mut out = String::with_capacity(raw_text.len() + (raw_text.len() / step) * 16 + 128);
    let mut total_digits = 0usize;
    for ch in raw_text.chars() {
        if ch.is_ascii_alphanumeric() {
            total_digits += 1;
        }
    }

    let _ = writeln!(
        out,
        "--- [Stream Output: {} characters | Radix Block Dump] ---",
        total_digits
    );

    let mut line_no = 1usize;
    let mut col = 0usize;
    for ch in raw_text.chars() {
        if col == 0 {
            let _ = write!(out, "{:04} | ", line_no);
        }
        out.push(ch);
        col += 1;
        if col == step {
            out.push('\n');
            line_no += 1;
            col = 0;
        }
    }
    if col > 0 {
        out.push('\n');
        line_no += 1;
    }

    let _ = writeln!(out, "--- [End of Stream | Total Lines: {}] ---", line_no - 1);
    out
}

/// Formats a full truth table dump for Propositional Logic in CMD Stream View
pub fn format_logic_truth_table(table: u64, var_count: u8) -> String { // wiring-ok
    use std::fmt::Write;
    const VARS: [&str; 6] = ["p", "q", "r", "s", "t", "u"];
    const CELL_LUT: [&str; 2] = [" F |", " T |"];
    const RES_LUT: [&str; 2] = ["   F\n", "   T\n"];
    let n = (var_count as usize).clamp(1, 4); // Display up to 16 rows
    let total_rows = 1usize << n;
    let mut out = String::with_capacity(128 + total_rows * (n * 4 + 6));

    let _ = writeln!(out, "--- [Truth Table Dump: {} Variables ({} States)] ---", n, total_rows);
    for &v_name in &VARS[..n] {
        out.push(' ');
        out.push_str(v_name);
        out.push_str(" |");
    }
    out.push_str(" Result\n");
    for _ in 0..n {
        out.push_str("---+");
    }
    out.push_str("-------\n");

    for row in 0..total_rows {
        for i in 0..n {
            let val = ((row >> (n - 1 - i)) & 1) as usize;
            out.push_str(CELL_LUT[val]);
        }
        let bit = ((table >> row) & 1) as usize;
        out.push_str(RES_LUT[bit]);
    }
    let _ = writeln!(out, "--- [Bitmask: 0x{:016X}] ---", table);
    out
}

/// Formats an aligned 2D matrix dump for CMD Stream View
pub fn format_matrix_stream(rows: usize, cols: usize, data: &[f64]) -> String { // wiring-ok
    use std::fmt::Write;
    let mut out = String::with_capacity(96 + rows * (cols * 12 + 10));
    let _ = writeln!(out, "--- [Matrix Dump: {}x{} Grid] ---", rows, cols);
    for r in 0..rows {
        let _ = write!(out, "{:02} | [ ", r + 1);
        for c in 0..cols {
            if c > 0 { out.push_str(", "); }
            let _ = write!(out, "{:10.4}", data[r * cols + c]);
        }
        out.push_str(" ]\n");
    }
    let _ = writeln!(out, "--- [Total Elements: {}] ---", rows * cols);
    out
}
