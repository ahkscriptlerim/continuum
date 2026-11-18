use std::fmt;
use crate::engine::bignum::RadixMode;
use crate::engine::bignum::radix::div_rem_u64;
use crate::engine::bignum::{
    decimal_add, decimal_sub, decimal_mul, decimal_div, decimal_pow,
    format_exact_decimal, normalize_decimal as bignum_normalize_decimal, POW10_I128,
};

#[derive(Clone, Debug, PartialEq)]
pub enum CalcValue {
    /// Tier 0: Single-cycle hardware scalar integer
    FastInt(i64),
    /// Tier 0: Banking-tier exact arbitrary-scale base-10 decimal: mantissa * 10^(-scale)
    ExactDecimal {
        mantissa: i128,
        scale: u32,
    },
    /// Tier 0: Single-cycle hardware IEEE-754 double precision float
    FastFloat(f64),
    /// Tier 1: Small Vector Optimization (SVO) inline limbs up to 256 bits (zero heap, L1 cache)
    InlineBigInt {
        sign: bool,
        limb_count: u8,
        limbs: [u64; 4],
    },
    /// Tier 2: Heap-backed arbitrary precision limbs
    HeapBigInt {
        sign: bool,
        limbs: Vec<u64>,
    },
    /// Tier 3: Ultra-Astronomical Integer (> 4,194,304 bits) with 19-digit normalized decimal
    /// significand (`sci`), exact modulo 10^19 trailing digits (`mod10_19`), and exact bit length (`bit_len`).
    BigIntSci {
        sign: bool,
        sci: crate::engine::bignum::DecSci,
        mod10_19: u64,
        bit_len: u64,
    },
    /// Flat 2D Matrix (row-major)
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    /// 64-bit Bitboard Propositional Logic Truth Table (up to 6 variables)
    Logic {
        table: u64,
        var_count: u8,
        tautology: bool,
        contradiction: bool,
    },
    /// Factored integer result: original value and Unicode factor power display (e.g. 2² × 7)
    Factored {
        original: u64,
        display: String,
    },
    /// Complex number in rectangular form: re + im·i
    Complex {
        re: f64,
        im: f64,
    },
    /// Symbolic univariate polynomial: coeffs[k] is coefficient of x^k
    Polynomial {
        coeffs: Vec<f64>,
    },
}

/// SOTA IEEE-754 binary floating-point representation artifact elimination (ULP noise snapping).
/// Eliminates .99999999999999 and .00000000000004 drift from binary float additions and subtractions
/// by testing for proximity to integers and compact decimal scale candidates within 2e-13.
#[inline(always)]
pub fn snap_float(val: f64) -> f64 {
    const POW10_F64: [f64; 12] = [
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
    ];
    if !val.is_finite() {
        return val;
    }
    let scale_mag = val.abs().max(1.0);
    // Test for integer snap first
    let r = val.round();
    if (val - r).abs() <= 2e-12 * scale_mag {
        return r;
    }
    // Test for small decimal scale snap (1 to 12 decimal places) via L1d exact-integer POW10 LUT (`0` powi, 0.5-ULP exact)
    let tol = 2e-13 * scale_mag;
    for &factor in &POW10_F64 {
        let candidate = (val * factor).round() / factor;
        if (val - candidate).abs() <= tol {
            return candidate;
        }
    }
    val
}

/// SOTA clean float formatter: eliminates trailing zeros and binary floating-point noise
pub fn format_clean_float(fl: f64) -> String {
    if !fl.is_finite() {
        return fl.to_string();
    }
    let snapped = snap_float(fl);
    let r = snapped.round();
    if (snapped - r).abs() < 1e-12 && snapped.abs() < 1e15 {
        return (r as i64).to_string();
    }
    if snapped.abs() >= 1e-5 && snapped.abs() < 1e15 {
        let s = format!("{:.12}", snapped);
        if s.contains('.') {
            let trimmed = s.trim_end_matches('0').trim_end_matches('.');
            return trimmed.to_string();
        }
        return s;
    }
    format!("{}", snapped)
}

/// Extracts (mantissa, scale) if the value can be cleanly represented as an exact base-10 decimal
#[inline]
pub fn to_decimal_parts(val: &CalcValue) -> Option<(i128, u32)> {
    match val {
        CalcValue::FastInt(i) => Some((*i as i128, 0)),
        CalcValue::ExactDecimal { mantissa, scale } => Some((*mantissa, *scale)),
        CalcValue::InlineBigInt { sign, limb_count, limbs } => {
            if *limb_count == 1 {
                let v = limbs[0] as i128;
                Some((if *sign { -v } else { v }, 0))
            } else if *limb_count == 2 {
                let u = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
                if u <= i128::MAX as u128 {
                    let v = u as i128;
                    Some((if *sign { -v } else { v }, 0))
                } else {
                    None
                }
            } else {
                None
            }
        }
        CalcValue::HeapBigInt { sign, limbs } => {
            if limbs.len() == 1 {
                let v = limbs[0] as i128;
                Some((if *sign { -v } else { v }, 0))
            } else if limbs.len() == 2 {
                let u = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
                if u <= i128::MAX as u128 {
                    let v = u as i128;
                    Some((if *sign { -v } else { v }, 0))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

impl CalcValue {
    pub fn from_i64(v: i64) -> Self { // wiring-ok
        CalcValue::FastInt(v)
    }

    pub fn from_f64(v: f64) -> Self {
        let s = snap_float(v);
        if s.is_finite() && s.fract() == 0.0 && s.abs() < (i64::MAX as f64) {
            CalcValue::FastInt(s as i64)
        } else {
            CalcValue::FastFloat(s)
        }
    }

    pub fn from_complex(re: f64, im: f64) -> Self {
        let r_s = snap_float(re);
        let i_s = snap_float(im);
        if i_s.abs() < 1e-12 {
            Self::from_f64(r_s)
        } else {
            CalcValue::Complex {
                re: if r_s.abs() < 1e-12 { 0.0 } else { r_s },
                im: i_s,
            }
        }
    }

    pub fn from_polynomial(mut coeffs: Vec<f64>) -> Self {
        for c in &mut coeffs {
            let s = snap_float(*c);
            *c = if s.abs() < 1e-12 { 0.0 } else { s };
        }
        while coeffs.len() > 1 && coeffs.last().map_or(false, |&c| c.abs() < 1e-12) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            CalcValue::FastInt(0)
        } else if coeffs.len() == 1 {
            Self::from_f64(coeffs[0])
        } else {
            CalcValue::Polynomial { coeffs }
        }
    }

    #[inline]
    pub fn as_complex(&self) -> Option<(f64, f64)> {
        match self {
            CalcValue::Complex { re, im } => Some((*re, *im)),
            CalcValue::FastInt(i) => Some((*i as f64, 0.0)),
            CalcValue::ExactDecimal { .. }
            | CalcValue::FastFloat(_)
            | CalcValue::InlineBigInt { .. }
            | CalcValue::HeapBigInt { .. }
            | CalcValue::Factored { .. } => Some((self.to_f64(), 0.0)),
            _ => None,
        }
    }

    #[inline]
    pub fn as_polynomial(&self) -> Option<Vec<f64>> {
        match self {
            CalcValue::Polynomial { coeffs } => Some(coeffs.clone()),
            CalcValue::FastInt(i) => Some(vec![*i as f64]),
            CalcValue::ExactDecimal { .. }
            | CalcValue::FastFloat(_)
            | CalcValue::InlineBigInt { .. }
            | CalcValue::HeapBigInt { .. }
            | CalcValue::Factored { .. } => Some(vec![self.to_f64()]),
            _ => None,
        }
    }

    /// Extracts an exact 64-bit two's-complement bit pattern (`0` 53-bit `f64` mantissa truncation!)
    /// for programmer bitwise intrinsics (`band`, `bor`, `bxor`, `bnot`, `shl`, `shr`, `popcnt`, `clz`, `ctz`, `rol`, `ror`).
    #[inline]
    pub fn to_exact_u64_bits(&self) -> Result<u64, String> {
        match self {
            CalcValue::FastInt(i) => Ok(*i as u64),
            CalcValue::ExactDecimal { mantissa, scale: 0 } => {
                if *mantissa >= (i64::MIN as i128) && *mantissa <= (u64::MAX as i128) {
                    if *mantissa < 0 {
                        Ok((*mantissa as i64) as u64)
                    } else {
                        Ok(*mantissa as u64)
                    }
                } else {
                    Err("Value exceeds 64-bit bitwise width".to_string())
                }
            }
            CalcValue::InlineBigInt { sign, limb_count, limbs } => {
                let mut c = *limb_count as usize;
                while c > 0 && limbs[c - 1] == 0 {
                    c -= 1;
                }
                if c == 0 {
                    Ok(0)
                } else if c == 1 {
                    if !*sign {
                        Ok(limbs[0])
                    } else if limbs[0] <= (1u64 << 63) {
                        Ok(limbs[0].wrapping_neg())
                    } else {
                        Err("Negative integer exceeds 64-bit two's-complement range".to_string())
                    }
                } else {
                    Err("Integer exceeds 64-bit bitwise width".to_string())
                }
            }
            CalcValue::HeapBigInt { sign, limbs } => {
                let mut c = limbs.len();
                while c > 0 && limbs[c - 1] == 0 {
                    c -= 1;
                }
                if c == 0 {
                    Ok(0)
                } else if c == 1 {
                    if !*sign {
                        Ok(limbs[0])
                    } else if limbs[0] <= (1u64 << 63) {
                        Ok(limbs[0].wrapping_neg())
                    } else {
                        Err("Negative integer exceeds 64-bit two's-complement range".to_string())
                    }
                } else {
                    Err("Integer exceeds 64-bit bitwise width".to_string())
                }
            }
            CalcValue::Factored { original, .. } => Ok(*original),
            CalcValue::FastFloat(f) => {
                if f.is_finite() && f.fract() == 0.0 && *f >= (i64::MIN as f64) && *f <= (u64::MAX as f64) {
                    if *f < 0.0 {
                        Ok((*f as i64) as u64)
                    } else {
                        Ok(*f as u64)
                    }
                } else {
                    Err("Bitwise operation requires an exact 64-bit integer".to_string())
                }
            }
            _ => Err("Bitwise operation requires an exact 64-bit integer".to_string()),
        }
    }

    /// Extracts an exact signed 128-bit integer (`0` float conversion) when the value is an integer within `i128`.
    #[inline]
    pub fn to_exact_i128(&self) -> Option<i128> {
        match self {
            CalcValue::FastInt(i) => Some(*i as i128),
            CalcValue::ExactDecimal { mantissa, scale: 0 } => Some(*mantissa),
            CalcValue::Factored { original, .. } => Some(*original as i128),
            CalcValue::InlineBigInt { sign, limb_count, limbs } => {
                let mut c = *limb_count as usize;
                while c > 0 && limbs[c - 1] == 0 {
                    c -= 1;
                }
                if c == 0 {
                    Some(0)
                } else if c == 1 {
                    let u = limbs[0] as i128;
                    Some(if *sign { -u } else { u })
                } else if c == 2 {
                    let u = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
                    if !*sign && u <= (i128::MAX as u128) {
                        Some(u as i128)
                    } else if *sign && u == (1u128 << 127) {
                        Some(i128::MIN)
                    } else if *sign && u <= (i128::MAX as u128) {
                        Some(-(u as i128))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            CalcValue::HeapBigInt { sign, limbs } => {
                let mut c = limbs.len();
                while c > 0 && limbs[c - 1] == 0 {
                    c -= 1;
                }
                if c == 0 {
                    Some(0)
                } else if c == 1 {
                    let u = limbs[0] as i128;
                    Some(if *sign { -u } else { u })
                } else if c == 2 {
                    let u = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
                    if !*sign && u <= (i128::MAX as u128) {
                        Some(u as i128)
                    } else if *sign && u == (1u128 << 127) {
                        Some(i128::MIN)
                    } else if *sign && u <= (i128::MAX as u128) {
                        Some(-(u as i128))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Exact scalar ordering across `FastInt`, `ExactDecimal`, `InlineBigInt`, `HeapBigInt`, and `BigIntSci`
    /// without truncating integers `> 2^53` or decimals `> 15` digits to `f64`.
    #[inline]
    pub fn cmp_scalar(&self, rhs: &Self) -> core::cmp::Ordering {
        let is_exact_num = |v: &Self| {
            matches!(
                v,
                CalcValue::FastInt(_)
                    | CalcValue::ExactDecimal { .. }
                    | CalcValue::InlineBigInt { .. }
                    | CalcValue::HeapBigInt { .. }
                    | CalcValue::BigIntSci { .. }
                    | CalcValue::Factored { .. }
            )
        };
        if is_exact_num(self) && is_exact_num(rhs) {
            if let Ok(diff) = self.sub(rhs) {
                return match diff {
                    CalcValue::FastInt(0) | CalcValue::ExactDecimal { mantissa: 0, .. } => {
                        core::cmp::Ordering::Equal
                    }
                    CalcValue::FastInt(i) if i < 0 => core::cmp::Ordering::Less,
                    CalcValue::FastInt(_) => core::cmp::Ordering::Greater,
                    CalcValue::ExactDecimal { mantissa, .. } if mantissa < 0 => {
                        core::cmp::Ordering::Less
                    }
                    CalcValue::ExactDecimal { .. } => core::cmp::Ordering::Greater,
                    CalcValue::InlineBigInt { sign: true, .. }
                    | CalcValue::HeapBigInt { sign: true, .. }
                    | CalcValue::BigIntSci { sign: true, .. } => core::cmp::Ordering::Less,
                    _ => core::cmp::Ordering::Greater,
                };
            }
        }
        let a = self.to_f64();
        let b = rhs.to_f64();
        if (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
            && (matches!(self, CalcValue::FastFloat(_)) || matches!(rhs, CalcValue::FastFloat(_)))
        {
            core::cmp::Ordering::Equal
        } else {
            a.partial_cmp(&b).unwrap_or(core::cmp::Ordering::Equal)
        }
    }

    #[inline]
    pub fn exact_sgn(&self) -> i64 {
        match self {
            CalcValue::FastInt(0) | CalcValue::ExactDecimal { mantissa: 0, .. } => 0,
            CalcValue::FastInt(i) => if *i > 0 { 1 } else { -1 },
            CalcValue::ExactDecimal { mantissa, .. } => if *mantissa > 0 { 1 } else { -1 },
            CalcValue::InlineBigInt { sign, limb_count, limbs } => {
                if *limb_count == 0 || limbs[..(*limb_count as usize)].iter().all(|&x| x == 0) {
                    0
                } else if *sign {
                    -1
                } else {
                    1
                }
            }
            CalcValue::HeapBigInt { sign, limbs } => {
                if limbs.iter().all(|&x| x == 0) {
                    0
                } else if *sign {
                    -1
                } else {
                    1
                }
            }
            CalcValue::BigIntSci { sign, sci, .. } => {
                if sci.mant == 0 {
                    0
                } else if *sign {
                    -1
                } else {
                    1
                }
            }
            CalcValue::Factored { original, .. } => if *original == 0 { 0 } else { 1 },
            _ => {
                let f = self.to_f64();
                if f > 0.0 { 1 } else if f < 0.0 { -1 } else { 0 }
            }
        }
    }

    #[inline]
    pub fn exact_abs(&self) -> Self {
        match self {
            CalcValue::FastInt(i) => {
                if *i == i64::MIN {
                    Self::normalize_decimal(-(i64::MIN as i128), 0)
                } else {
                    CalcValue::FastInt(i.abs())
                }
            }
            CalcValue::ExactDecimal { mantissa, scale } => {
                Self::normalize_decimal(mantissa.abs(), *scale)
            }
            CalcValue::InlineBigInt { limb_count, limbs, .. } => CalcValue::InlineBigInt {
                sign: false,
                limb_count: *limb_count,
                limbs: *limbs,
            },
            CalcValue::HeapBigInt { limbs, .. } => CalcValue::HeapBigInt {
                sign: false,
                limbs: limbs.clone(),
            },
            CalcValue::BigIntSci { sci, mod10_19, bit_len, .. } => CalcValue::BigIntSci {
                sign: false,
                sci: *sci,
                mod10_19: *mod10_19,
                bit_len: *bit_len,
            },
            CalcValue::Factored { original, .. } => Self::normalize_decimal(*original as i128, 0),
            CalcValue::Complex { re, im } => Self::from_f64(re.hypot(*im)),
            _ => Self::from_f64(self.to_f64().abs()),
        }
    }

    #[inline]
    pub fn exact_round(&self) -> Self {
        match self {
            CalcValue::FastInt(_)
            | CalcValue::InlineBigInt { .. }
            | CalcValue::HeapBigInt { .. }
            | CalcValue::BigIntSci { .. } => self.clone(),
            CalcValue::Factored { original, .. } => Self::normalize_decimal(*original as i128, 0),
            CalcValue::ExactDecimal { mantissa, scale } => {
                if *scale == 0 {
                    return Self::normalize_decimal(*mantissa, 0);
                }
                let s = *scale as usize;
                if s < POW10_I128.len() {
                    let div = POW10_I128[s];
                    let q = *mantissa / div;
                    let r = (*mantissa % div).abs();
                    let rounded = if r * 2 >= div {
                        if *mantissa >= 0 { q + 1 } else { q - 1 }
                    } else {
                        q
                    };
                    return Self::normalize_decimal(rounded, 0);
                }
                Self::from_f64(self.to_f64().round())
            }
            _ => Self::from_f64(self.to_f64().round()),
        }
    }

    #[inline]
    pub fn exact_floor(&self) -> Self {
        match self {
            CalcValue::FastInt(_)
            | CalcValue::InlineBigInt { .. }
            | CalcValue::HeapBigInt { .. }
            | CalcValue::BigIntSci { .. } => self.clone(),
            CalcValue::Factored { original, .. } => Self::normalize_decimal(*original as i128, 0),
            CalcValue::ExactDecimal { mantissa, scale } => {
                if *scale == 0 {
                    return Self::normalize_decimal(*mantissa, 0);
                }
                let s = *scale as usize;
                if s < POW10_I128.len() {
                    let div = POW10_I128[s];
                    let q = *mantissa / div;
                    let r = *mantissa % div;
                    let fl = if r < 0 { q - 1 } else { q };
                    return Self::normalize_decimal(fl, 0);
                }
                Self::from_f64(self.to_f64().floor())
            }
            _ => Self::from_f64(self.to_f64().floor()),
        }
    }

    #[inline]
    pub fn exact_ceil(&self) -> Self {
        match self {
            CalcValue::FastInt(_)
            | CalcValue::InlineBigInt { .. }
            | CalcValue::HeapBigInt { .. }
            | CalcValue::BigIntSci { .. } => self.clone(),
            CalcValue::Factored { original, .. } => Self::normalize_decimal(*original as i128, 0),
            CalcValue::ExactDecimal { mantissa, scale } => {
                if *scale == 0 {
                    return Self::normalize_decimal(*mantissa, 0);
                }
                let s = *scale as usize;
                if s < POW10_I128.len() {
                    let div = POW10_I128[s];
                    let q = *mantissa / div;
                    let r = *mantissa % div;
                    let cl = if r > 0 { q + 1 } else { q };
                    return Self::normalize_decimal(cl, 0);
                }
                Self::from_f64(self.to_f64().ceil())
            }
            _ => Self::from_f64(self.to_f64().ceil()),
        }
    }

    /// Exact integer/decimal square root (`u128::isqrt`) when the operand is a perfect square in `ℤ` or `ℤ·10⁻²ᵏ`
    /// (`sqrt(0.09) == ExactDecimal(0.3)`, `sqrt(9007199254740993^2) == 9007199254740993`).
    #[inline]
    pub fn try_exact_sqrt(&self) -> Option<Self> {
        let (mut m, mut s) = to_decimal_parts(self)?;
        if m < 0 {
            return None;
        }
        if m == 0 {
            return Some(CalcValue::FastInt(0));
        }
        if (s & 1) == 1 {
            m = m.checked_mul(10)?;
            s += 1;
        }
        let u = m as u128;
        let r = u.isqrt();
        if r.saturating_mul(r) == u {
            Some(Self::normalize_decimal(r as i128, s >> 1))
        } else {
            None
        }
    }

    /// Exact integer/decimal cube root (`icbrt_u128`) when the operand is a perfect cube in `ℤ` or `ℤ·10⁻³ᵏ`
    /// (`cbrt(0.008) == ExactDecimal(0.2)`, `cbrt(-27) == FastInt(-3)`).
    #[inline]
    pub fn try_exact_cbrt(&self) -> Option<Self> {
        let (mut m, mut s) = to_decimal_parts(self)?;
        if m == 0 {
            return Some(CalcValue::FastInt(0));
        }
        let neg = m < 0;
        let rem3 = s % 3;
        if rem3 != 0 {
            let pad = 3 - rem3;
            m = m.checked_mul(POW10_I128[pad as usize])?;
            s += pad;
        }
        let u = m.unsigned_abs();
        let mut lo = 1u128;
        let mut hi = u.min(5_510_825_547_116u128); // floor(u128::MAX^(1/3)) = 6_981_463_658_331
        hi = hi.max(6_981_463_658_332u128.min(u));
        while lo <= hi {
            let mid = lo + ((hi - lo) >> 1);
            match mid.checked_mul(mid).and_then(|sq| sq.checked_mul(mid)) {
                Some(cb) if cb == u => {
                    let signed = if neg { -(mid as i128) } else { mid as i128 };
                    return Some(Self::normalize_decimal(signed, s / 3));
                }
                Some(cb) if cb < u => lo = mid + 1,
                _ => hi = mid - 1,
            }
        }
        None
    }

    #[inline]
    pub fn normalize_decimal(mantissa: i128, scale: u32) -> Self {
        let (m, s) = bignum_normalize_decimal(mantissa, scale);
        if s == 0 {
            if m >= i64::MIN as i128 && m <= i64::MAX as i128 {
                CalcValue::FastInt(m as i64)
            } else {
                let sign = m < 0;
                let abs_val = m.unsigned_abs();
                let lo = abs_val as u64;
                let hi = (abs_val >> 64) as u64;
                if hi == 0 {
                    pack_bignum(sign, vec![lo])
                } else {
                    pack_bignum(sign, vec![lo, hi])
                }
            }
        } else {
            CalcValue::ExactDecimal { mantissa: m, scale: s }
        }
    }

    /// Estimates the exact binary bit-length of an integer/bignum value in O(1) time.
    #[inline]
    pub fn bit_length_estimate(&self) -> u64 {
        match self {
            CalcValue::FastInt(i) => {
                let u = i.unsigned_abs();
                (64 - u.leading_zeros() as u64).max(1)
            }
            CalcValue::InlineBigInt { limb_count, limbs, .. } => {
                let c = (*limb_count as usize).clamp(1, 4);
                let top = limbs[c - 1];
                let top_bits = (64 - top.leading_zeros() as u64).max(1);
                ((c - 1) as u64) * 64 + top_bits
            }
            CalcValue::HeapBigInt { limbs, .. } => {
                if limbs.is_empty() {
                    1
                } else {
                    let c = limbs.len();
                    let top = limbs[c - 1];
                    let top_bits = (64 - top.leading_zeros() as u64).max(1);
                    ((c - 1) as u64) * 64 + top_bits
                }
            }
            CalcValue::BigIntSci { bit_len, .. } => (*bit_len).max(1),
            CalcValue::Factored { original, .. } => (64 - original.leading_zeros() as u64).max(1),
            CalcValue::ExactDecimal { mantissa, scale } => {
                let m_bits = (128 - mantissa.unsigned_abs().leading_zeros() as u64).max(1);
                let scale_bits = ((*scale as f64) * 3.321928).floor() as u64;
                m_bits.saturating_sub(scale_bits).max(1)
            }
            CalcValue::FastFloat(f) => {
                let a = f.abs();
                if a.is_finite() && a > 1.0 {
                    (a.log2().ceil() as u64).max(1)
                } else {
                    1
                }
            }
            _ => 1,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            CalcValue::FastInt(i) => *i as f64,
            CalcValue::ExactDecimal { mantissa, scale } => {
                let s_idx = *scale as usize;
                if s_idx < POW10_I128.len() {
                    (*mantissa as f64) / (POW10_I128[s_idx] as f64)
                } else {
                    (*mantissa as f64) / 10.0f64.powi(*scale as i32)
                }
            }
            CalcValue::FastFloat(f) => *f,
            CalcValue::InlineBigInt { sign, limb_count, limbs } => {
                let mut acc = 0.0f64;
                for i in (0..*limb_count as usize).rev() {
                    acc = acc * (18446744073709551616.0f64) + (limbs[i] as f64);
                }
                if *sign { -acc } else { acc }
            }
            CalcValue::HeapBigInt { sign, limbs } => {
                let mut end = limbs.len();
                while end > 0 && limbs[end - 1] == 0 {
                    end -= 1;
                }
                if end == 0 {
                    return 0.0;
                }
                let take_n = end.min(4);
                let mut acc = 0.0f64;
                for &l in limbs[end - take_n..end].iter().rev() {
                    acc = acc * 18446744073709551616.0f64 + (l as f64);
                }
                if end > 4 {
                    let shift_limbs = end - 4;
                    if shift_limbs > 16 {
                        acc = f64::INFINITY;
                    } else {
                        acc *= 2.0f64.powi((shift_limbs * 64) as i32);
                    }
                }
                if *sign { -acc } else { acc }
            }
            CalcValue::BigIntSci { sign, sci, .. } => {
                let mag = if sci.exp10 > 308 {
                    f64::INFINITY
                } else {
                    (sci.mant as f64) * 10.0f64.powi(sci.exp10.clamp(-308, 308) as i32)
                };
                if *sign { -mag } else { mag }
            }
            CalcValue::Matrix { data, .. } => data.first().copied().unwrap_or(0.0),
            CalcValue::Logic { table, .. } => *table as f64,
            CalcValue::Factored { original, .. } => *original as f64,
            CalcValue::Complex { re, im } => {
                if im.abs() < 1e-12 {
                    *re
                } else {
                    re.hypot(*im)
                }
            }
            CalcValue::Polynomial { coeffs } => coeffs.first().copied().unwrap_or(0.0),
        }
    }

    /// Computes accurate base-2 logarithm `log₂(|self|)` even when `|self| >= 2^1024` (where `to_f64()` overflows).
    #[inline]
    pub fn log2_mag(&self) -> f64 {
        match self {
            CalcValue::HeapBigInt { limbs, .. } => {
                let mut end = limbs.len();
                while end > 0 && limbs[end - 1] == 0 {
                    end -= 1;
                }
                if end == 0 {
                    f64::NEG_INFINITY
                } else if end == 1 {
                    (limbs[0] as f64).log2()
                } else {
                    let top = ((limbs[end - 1] as u128) << 64) | (limbs[end - 2] as u128);
                    ((end - 2) as f64) * 64.0 + (top as f64).log2()
                }
            }
            CalcValue::BigIntSci { sci, .. } => {
                if sci.mant == 0 {
                    f64::NEG_INFINITY
                } else {
                    ((sci.exp10 as f64) + (sci.mant as f64).log10()) * std::f64::consts::LOG2_10
                }
            }
            _ => self.to_f64().abs().log2(),
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            CalcValue::FastInt(_) => true,
            CalcValue::ExactDecimal { scale, .. } => *scale == 0,
            CalcValue::FastFloat(f) => f.fract() == 0.0 && f.is_finite(),
            CalcValue::InlineBigInt { .. } => true,
            CalcValue::HeapBigInt { .. } => true,
            CalcValue::BigIntSci { .. } => true,
            CalcValue::Factored { .. } => true,
            _ => false,
        }
    }

    pub fn add(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        // Symbolic Polynomial addition
        if matches!(self, CalcValue::Polynomial { .. }) || matches!(rhs, CalcValue::Polynomial { .. }) {
            if let (Some(p1), Some(p2)) = (self.as_polynomial(), rhs.as_polynomial()) {
                let n = p1.len().max(p2.len());
                let mut out = vec![0.0; n];
                for (i, &c) in p1.iter().enumerate() {
                    out[i] += c;
                }
                for (i, &c) in p2.iter().enumerate() {
                    out[i] += c;
                }
                return Ok(CalcValue::from_polynomial(out));
            }
        }

        // Complex addition
        if matches!(self, CalcValue::Complex { .. }) || matches!(rhs, CalcValue::Complex { .. }) {
            if let (Some((r1, i1)), Some((r2, i2))) = (self.as_complex(), rhs.as_complex()) {
                return Ok(CalcValue::from_complex(r1 + r2, i1 + i2));
            }
        }

        // Matrix addition
        if let (CalcValue::Matrix { rows: r1, cols: c1, data: d1 }, CalcValue::Matrix { rows: r2, cols: c2, data: d2 }) = (self, rhs) {
            if r1 != r2 || c1 != c2 {
                return Err(format!("Matrix dimension mismatch for addition: {}x{} vs {}x{}", r1, c1, r2, c2));
            }
            let mut res = Vec::with_capacity(d1.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                res.push(snap_float(a + b));
            }
            return Ok(CalcValue::Matrix { rows: *r1, cols: *c1, data: res });
        }

        // Exact integer addition when both are integers
        if self.is_integer() && rhs.is_integer() {
            if let (CalcValue::FastInt(a), CalcValue::FastInt(b)) = (self, rhs) {
                if let Some(res) = a.checked_add(*b) {
                    return Ok(CalcValue::FastInt(res));
                }
            }
            return big_add(self, rhs);
        }

        // Exact banking-tier decimal addition
        if let (Some((m1, s1)), Some((m2, s2))) = (to_decimal_parts(self), to_decimal_parts(rhs)) {
            if let Ok((res_m, res_s)) = decimal_add(m1, s1, m2, s2) {
                return Ok(CalcValue::normalize_decimal(res_m, res_s));
            }
        }

        match (self, rhs) {
            (CalcValue::FastFloat(a), b) => Ok(CalcValue::FastFloat(snap_float(a + b.to_f64()))),
            (a, CalcValue::FastFloat(b)) => Ok(CalcValue::FastFloat(snap_float(a.to_f64() + b))),
            _ => Ok(CalcValue::FastFloat(snap_float(self.to_f64() + rhs.to_f64()))),
        }
    }

    pub fn sub(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        // Symbolic Polynomial subtraction
        if matches!(self, CalcValue::Polynomial { .. }) || matches!(rhs, CalcValue::Polynomial { .. }) {
            if let (Some(p1), Some(p2)) = (self.as_polynomial(), rhs.as_polynomial()) {
                let n = p1.len().max(p2.len());
                let mut out = vec![0.0; n];
                for (i, &c) in p1.iter().enumerate() {
                    out[i] += c;
                }
                for (i, &c) in p2.iter().enumerate() {
                    out[i] -= c;
                }
                return Ok(CalcValue::from_polynomial(out));
            }
        }

        // Complex subtraction
        if matches!(self, CalcValue::Complex { .. }) || matches!(rhs, CalcValue::Complex { .. }) {
            if let (Some((r1, i1)), Some((r2, i2))) = (self.as_complex(), rhs.as_complex()) {
                return Ok(CalcValue::from_complex(r1 - r2, i1 - i2));
            }
        }

        // Matrix subtraction
        if let (CalcValue::Matrix { rows: r1, cols: c1, data: d1 }, CalcValue::Matrix { rows: r2, cols: c2, data: d2 }) = (self, rhs) {
            if r1 != r2 || c1 != c2 {
                return Err(format!("Matrix dimension mismatch for subtraction: {}x{} vs {}x{}", r1, c1, r2, c2));
            }
            let mut res = Vec::with_capacity(d1.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                res.push(snap_float(a - b));
            }
            return Ok(CalcValue::Matrix { rows: *r1, cols: *c1, data: res });
        }

        // Exact integer subtraction when both are integers
        if self.is_integer() && rhs.is_integer() {
            if let (CalcValue::FastInt(a), CalcValue::FastInt(b)) = (self, rhs) {
                if let Some(res) = a.checked_sub(*b) {
                    return Ok(CalcValue::FastInt(res));
                }
            }
            return big_sub(self, rhs);
        }

        // Exact banking-tier decimal subtraction
        if let (Some((m1, s1)), Some((m2, s2))) = (to_decimal_parts(self), to_decimal_parts(rhs)) {
            if let Ok((res_m, res_s)) = decimal_sub(m1, s1, m2, s2) {
                return Ok(CalcValue::normalize_decimal(res_m, res_s));
            }
        }

        match (self, rhs) {
            (CalcValue::FastFloat(a), b) => Ok(CalcValue::FastFloat(snap_float(a - b.to_f64()))),
            (a, CalcValue::FastFloat(b)) => Ok(CalcValue::FastFloat(snap_float(a.to_f64() - b))),
            _ => Ok(CalcValue::FastFloat(snap_float(self.to_f64() - rhs.to_f64()))),
        }
    }

    pub fn mul(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        // Symbolic Polynomial multiplication (convolution)
        if matches!(self, CalcValue::Polynomial { .. }) || matches!(rhs, CalcValue::Polynomial { .. }) {
            if let (Some(p1), Some(p2)) = (self.as_polynomial(), rhs.as_polynomial()) {
                if p1.is_empty() || p2.is_empty() {
                    return Ok(CalcValue::FastInt(0));
                }
                let out_len = p1.len() + p2.len() - 1;
                if out_len > crate::engine::evaluator::MAX_POLY_DEGREE + 1 {
                    return Err(format!(
                        "Polynomial degree exceeds exact limit (max degree {})",
                        crate::engine::evaluator::MAX_POLY_DEGREE
                    ));
                }
                let mut out = vec![0.0; out_len];
                for (i, &c1) in p1.iter().enumerate() {
                    if (i & 31) == 31 {
                        crate::engine::evaluator::check_task_aborted()?;
                    }
                    if c1 != 0.0 {
                        for (j, &c2) in p2.iter().enumerate() {
                            out[i + j] += c1 * c2;
                        }
                    }
                }
                return Ok(CalcValue::from_polynomial(out));
            }
        }

        // Complex multiplication
        if matches!(self, CalcValue::Complex { .. }) || matches!(rhs, CalcValue::Complex { .. }) {
            if let (Some((a, b)), Some((c, d))) = (self.as_complex(), rhs.as_complex()) {
                return Ok(CalcValue::from_complex(a * c - b * d, a * d + b * c));
            }
        }

        // 1. Matrix by Matrix or Matrix by Scalar
        match (self, rhs) {
            (CalcValue::Matrix { rows: r1, cols: c1, data: d1 }, CalcValue::Matrix { rows: r2, cols: c2, data: d2 }) => {
                let m1 = super::matrix::Matrix::new(*r1, *c1, d1.clone());
                let m2 = super::matrix::Matrix::new(*r2, *c2, d2.clone());
                let res = m1.mul(&m2)?;
                return Ok(CalcValue::Matrix { rows: res.rows, cols: res.cols, data: res.data });
            }
            (CalcValue::Matrix { rows, cols, data }, scalar) => {
                let s = scalar.to_f64();
                let scaled = data.iter().map(|&x| x * s).collect();
                return Ok(CalcValue::Matrix { rows: *rows, cols: *cols, data: scaled });
            }
            (scalar, CalcValue::Matrix { rows, cols, data }) => {
                let s = scalar.to_f64();
                let scaled = data.iter().map(|&x| x * s).collect();
                return Ok(CalcValue::Matrix { rows: *rows, cols: *cols, data: scaled });
            }
            _ => {}
        }

        // Exact integer multiplication
        if self.is_integer() && rhs.is_integer() {
            if let (CalcValue::FastInt(a), CalcValue::FastInt(b)) = (self, rhs) {
                if let Some(res) = a.checked_mul(*b) {
                    return Ok(CalcValue::FastInt(res));
                }
            }
            if std::ptr::eq(self, rhs) {
                return big_sqr(self);
            }
            return big_mul(self, rhs);
        }

        // Exact banking-tier decimal multiplication
        if let (Some((m1, s1)), Some((m2, s2))) = (to_decimal_parts(self), to_decimal_parts(rhs)) {
            if let Ok((res_m, res_s)) = decimal_mul(m1, s1, m2, s2) {
                return Ok(CalcValue::normalize_decimal(res_m, res_s));
            }
        }

        match (self, rhs) {
            (CalcValue::FastFloat(a), b) => Ok(CalcValue::FastFloat(snap_float(a * b.to_f64()))),
            (a, CalcValue::FastFloat(b)) => Ok(CalcValue::FastFloat(snap_float(a.to_f64() * b))),
            _ => Ok(CalcValue::FastFloat(snap_float(self.to_f64() * rhs.to_f64()))),
        }
    }

    pub fn div(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        // Symbolic Polynomial division (scalar division or exact polynomial long division)
        if matches!(self, CalcValue::Polynomial { .. }) || matches!(rhs, CalcValue::Polynomial { .. }) {
            if let (Some(p1), Some(p2)) = (self.as_polynomial(), rhs.as_polynomial()) {
                return poly_div(&p1, &p2);
            }
        }

        // Complex division
        if matches!(self, CalcValue::Complex { .. }) || matches!(rhs, CalcValue::Complex { .. }) {
            if let (Some((a, b)), Some((c, d))) = (self.as_complex(), rhs.as_complex()) {
                let denom = c * c + d * d;
                if denom == 0.0 {
                    return Err("Complex division by zero".to_string());
                }
                return Ok(CalcValue::from_complex(
                    (a * c + b * d) / denom,
                    (b * c - a * d) / denom,
                ));
            }
        }

        match (self, rhs) {
            (CalcValue::Matrix { rows: r1, cols: c1, data: d1 }, CalcValue::Matrix { rows: r2, cols: c2, data: d2 }) => {
                let m1 = super::matrix::Matrix::new(*r1, *c1, d1.clone());
                let m2 = super::matrix::Matrix::new(*r2, *c2, d2.clone());
                let inv = m2.inverse()?;
                let res = m1.mul(&inv)?;
                return Ok(CalcValue::Matrix { rows: res.rows, cols: res.cols, data: res.data });
            }
            (CalcValue::Matrix { rows, cols, data }, scalar) => {
                let s = scalar.to_f64();
                if s == 0.0 {
                    return Err("Division by zero in matrix scalar division".to_string());
                }
                let scaled = data.iter().map(|&x| snap_float(x / s)).collect();
                return Ok(CalcValue::Matrix { rows: *rows, cols: *cols, data: scaled });
            }
            _ => {}
        }

        if matches!(self, CalcValue::BigIntSci { .. }) || matches!(rhs, CalcValue::BigIntSci { .. }) {
            return big_sci_div(self, rhs, false);
        }

        // Banking-tier exact division
        if let (Some((m1, s1)), Some((m2, s2))) = (to_decimal_parts(self), to_decimal_parts(rhs)) {
            if m2 == 0 {
                return Err("Division by zero".to_string());
            }
            if s1 == 0 && s2 == 0 && m1 % m2 == 0 {
                let q = m1 / m2;
                if q >= i64::MIN as i128 && q <= i64::MAX as i128 {
                    return Ok(CalcValue::FastInt(q as i64));
                }
            }
            if let Ok((res_m, res_s)) = decimal_div(m1, s1, m2, s2) {
                return Ok(CalcValue::normalize_decimal(res_m, res_s));
            }
        }

        // Exact bignum division by a single-limb integer when evenly divisible
        if self.is_integer() && rhs.is_integer() {
            if let (Some((sign_a, ref_a)), Some((sign_b, div_u64))) =
                (extract_limbs_svo(self), extract_u64_divisor(rhs))
            {
                let rem = rem_limbs_u64(ref_a.as_slice(), div_u64);
                crate::engine::evaluator::check_task_aborted()?;
                if rem == 0 {
                    let mut quot_limbs = ref_a.as_slice().to_vec();
                    let _ = div_rem_u64(&mut quot_limbs, div_u64);
                    crate::engine::evaluator::check_task_aborted()?;
                    return Ok(pack_bignum(sign_a != sign_b, quot_limbs));
                }
            }
        }

        let b_f = rhs.to_f64();
        if b_f == 0.0 {
            return Err("Division by zero".to_string());
        }

        match (self, rhs) {
            (CalcValue::FastInt(a), CalcValue::FastInt(b)) => {
                if *b != 0 && a % b == 0 {
                    Ok(CalcValue::FastInt(a / b))
                } else {
                    Ok(CalcValue::FastFloat(snap_float(*a as f64 / *b as f64)))
                }
            }
            _ => {
                let a_f = self.to_f64();
                if !a_f.is_finite() || !b_f.is_finite() {
                    if self.is_integer() && rhs.is_integer() {
                        return big_sci_div(self, rhs, false);
                    }
                }
                Ok(CalcValue::FastFloat(snap_float(a_f / b_f)))
            }
        }
    }

    pub fn floor_div(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        match (self, rhs) {
            (CalcValue::FastInt(a), CalcValue::FastInt(b)) => {
                if *b == 0 { return Err("Integer division by zero".to_string()); }
                Ok(CalcValue::FastInt(a.div_euclid(*b)))
            }
            _ => {
                if matches!(self, CalcValue::BigIntSci { .. }) || matches!(rhs, CalcValue::BigIntSci { .. }) {
                    return big_sci_div(self, rhs, true);
                }
                if self.is_integer() && rhs.is_integer() {
                    if let (Some((sign_a, ref_a)), Some((sign_b, div_u64))) =
                        (extract_limbs_svo(self), extract_u64_divisor(rhs))
                    {
                        let mut quot_limbs = ref_a.as_slice().to_vec();
                        let rem = div_rem_u64(&mut quot_limbs, div_u64);
                        crate::engine::evaluator::check_task_aborted()?;
                        if sign_a && rem != 0 {
                            let mut carry = 1u64;
                            for limb in &mut quot_limbs {
                                let (sum, c) = limb.overflowing_add(carry);
                                *limb = sum;
                                carry = u64::from(c);
                                if carry == 0 {
                                    break;
                                }
                            }
                            if carry > 0 {
                                quot_limbs.push(carry);
                            }
                        }
                        return Ok(pack_bignum(sign_a != sign_b, quot_limbs));
                    }
                    return big_sci_div(self, rhs, true);
                }
                let div = self.div(rhs)?;
                match div {
                    CalcValue::FastInt(i) => Ok(CalcValue::FastInt(i)),
                    CalcValue::ExactDecimal { mantissa, scale } => {
                        let scale_idx = scale as usize;
                        if scale_idx < POW10_I128.len() {
                            let factor = POW10_I128[scale_idx];
                            let q = mantissa.div_euclid(factor);
                            if q >= i64::MIN as i128 && q <= i64::MAX as i128 {
                                return Ok(CalcValue::FastInt(q as i64));
                            }
                        }
                        Ok(CalcValue::FastInt(div.to_f64().floor() as i64))
                    }
                    _ => Ok(CalcValue::FastInt(div.to_f64().floor() as i64)),
                }
            }
        }
    }

    pub fn modulo(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        match (self, rhs) {
            (CalcValue::FastInt(a), CalcValue::FastInt(b)) => {
                if *b == 0 { return Err("Modulo by zero".to_string()); }
                Ok(CalcValue::FastInt(a.rem_euclid(*b)))
            }
            _ => {
                if matches!(self, CalcValue::BigIntSci { .. }) || matches!(rhs, CalcValue::BigIntSci { .. }) {
                    return big_sci_rem(self, rhs);
                }
                if let (Some((m1, s1)), Some((m2, s2))) = (to_decimal_parts(self), to_decimal_parts(rhs)) {
                    if m2 == 0 { return Err("Modulo by zero".to_string()); }
                    let s_max = s1.max(s2);
                    let diff1 = (s_max - s1) as usize;
                    let diff2 = (s_max - s2) as usize;
                    if diff1 < POW10_I128.len() && diff2 < POW10_I128.len() {
                        if let (Some(m1_scaled), Some(m2_scaled)) = (m1.checked_mul(POW10_I128[diff1]), m2.checked_mul(POW10_I128[diff2])) {
                            let rem = m1_scaled.rem_euclid(m2_scaled);
                            return Ok(CalcValue::normalize_decimal(rem, s_max));
                        }
                    }
                }
                if self.is_integer() && rhs.is_integer() {
                    if let (Some((sign_a, ref_a)), Some((_sign_b, div_u64))) =
                        (extract_limbs_svo(self), extract_u64_divisor(rhs))
                    {
                        let rem = rem_limbs_u64(ref_a.as_slice(), div_u64);
                        crate::engine::evaluator::check_task_aborted()?;
                        let euclid_rem = if sign_a && rem != 0 { div_u64 - rem } else { rem };
                        return Ok(pack_bignum_slice(false, &[euclid_rem]));
                    }
                }
                let b_f = rhs.to_f64();
                if b_f == 0.0 { return Err("Modulo by zero".to_string()); }
                Ok(CalcValue::FastFloat(self.to_f64().rem_euclid(b_f)))
            }
        }
    }

    pub fn pow(&self, rhs: &CalcValue) -> Result<CalcValue, String> {
        // Symbolic Polynomial integer exponentiation: (x + 1)^3 -> x³ + 3x² + 3x + 1
        if matches!(self, CalcValue::Polynomial { .. }) {
            let exp_f = rhs.to_f64();
            if exp_f >= 0.0 && exp_f.fract() == 0.0 && exp_f <= 64.0 {
                let mut e = exp_f as u32;
                let mut res = CalcValue::FastInt(1);
                let mut base = self.clone();
                while e > 0 {
                    if e & 1 == 1 {
                        res = res.mul(&base)?;
                    }
                    if e > 1 {
                        base = base.mul(&base)?;
                    }
                    e >>= 1;
                }
                return Ok(res);
            } else {
                return Err("Polynomial exponent must be a non-negative integer (0..=64)".to_string());
            }
        }

        // Complex exponentiation: z^w = e^(w * ln(z))
        if matches!(self, CalcValue::Complex { .. }) || matches!(rhs, CalcValue::Complex { .. }) {
            if let (Some((a, b)), Some((c, d))) = (self.as_complex(), rhs.as_complex()) {
                // Fast exact path for small integer powers of Gaussian/complex numbers
                if d.abs() < 1e-12 && c.fract() == 0.0 && c >= 0.0 && c <= 64.0 {
                    let mut e = c as u32;
                    let mut res_r = 1.0;
                    let mut res_i = 0.0;
                    let mut base_r = a;
                    let mut base_i = b;
                    while e > 0 {
                        if e & 1 == 1 {
                            let nr = res_r * base_r - res_i * base_i;
                            let ni = res_r * base_i + res_i * base_r;
                            res_r = nr;
                            res_i = ni;
                        }
                        if e > 1 {
                            let nr = base_r * base_r - base_i * base_i;
                            let ni = 2.0 * base_r * base_i;
                            base_r = nr;
                            base_i = ni;
                        }
                        e >>= 1;
                    }
                    return Ok(CalcValue::from_complex(res_r, res_i));
                }
                let r = a.hypot(b);
                if r == 0.0 {
                    return Ok(CalcValue::FastInt(0));
                }
                let theta = b.atan2(a);
                let ln_r = r.ln();
                let mag = (c * ln_r - d * theta).exp();
                let ang = d * ln_r + c * theta;
                return Ok(CalcValue::from_complex(mag * ang.cos(), mag * ang.sin()));
            }
        }

        if let CalcValue::Matrix { rows, cols, data } = self {
            let exp_f = rhs.to_f64();
            if exp_f >= 0.0 && exp_f.fract() == 0.0 && exp_f <= 10_000.0 {
                let mat = super::matrix::Matrix::new(*rows, *cols, data.clone());
                let res = mat.pow(exp_f as u32)?;
                return Ok(CalcValue::Matrix { rows: res.rows, cols: res.cols, data: res.data });
            } else {
                return Err("Matrix exponent must be a non-negative integer".to_string());
            }
        }

        if self.is_integer() && rhs.is_integer() {
            match rhs {
                CalcValue::FastInt(exp_i) => {
                    if *exp_i >= 0 {
                        return big_pow(self, *exp_i as u64);
                    } else {
                        return Ok(CalcValue::FastFloat(self.to_f64().powi((*exp_i).clamp(i32::MIN as i64, i32::MAX as i64) as i32)));
                    }
                }
                CalcValue::ExactDecimal { mantissa, scale: 0 } => {
                    if *mantissa >= 0 && *mantissa <= u64::MAX as i128 {
                        return big_pow(self, *mantissa as u64);
                    } else if *mantissa < 0 {
                        return Ok(CalcValue::FastFloat(self.to_f64().powf(*mantissa as f64)));
                    }
                }
                CalcValue::FastFloat(ef) if ef.is_finite() && ef.fract() == 0.0 => {
                    if *ef >= 0.0 && *ef <= 9_007_199_254_740_992.0 {
                        return big_pow(self, *ef as u64);
                    } else if *ef < 0.0 {
                        return Ok(CalcValue::FastFloat(self.to_f64().powf(*ef)));
                    } else {
                        return Err(
                            "Exponent exceeds exact 53-bit float precision; write exponent as an exact integer".to_string(),
                        );
                    }
                }
                _ => {}
            }
            // Exponent is a multi-limb bignum (InlineBigInt, HeapBigInt, or BigIntSci)
            let rhs_neg = match rhs {
                CalcValue::InlineBigInt { sign, .. }
                | CalcValue::HeapBigInt { sign, .. }
                | CalcValue::BigIntSci { sign, .. } => *sign,
                _ => false,
            };
            if rhs_neg {
                return Ok(CalcValue::FastFloat(self.to_f64().powf(rhs.to_f64())));
            }
            // O(1) fast path for unit bases (-1, 0, 1) raised to arbitrary bignum exponents
            match self {
                CalcValue::FastInt(0) => return Ok(CalcValue::FastInt(0)),
                CalcValue::FastInt(1) => return Ok(CalcValue::FastInt(1)),
                CalcValue::FastInt(-1) => {
                    let is_even = match rhs {
                        CalcValue::InlineBigInt { limbs, .. } => (limbs[0] & 1) == 0,
                        CalcValue::HeapBigInt { limbs, .. } => limbs.first().map_or(true, |l| (l & 1) == 0),
                        CalcValue::BigIntSci { mod10_19, .. } => (mod10_19 & 1) == 0,
                        _ => false,
                    };
                    return Ok(CalcValue::FastInt(if is_even { 1 } else { -1 }));
                }
                _ => {
                    if let CalcValue::InlineBigInt { sign: false, limb_count: 1, limbs } = rhs {
                        return big_pow(self, limbs[0]);
                    }
                    if crate::engine::evaluator::is_deep_compute_active() {
                        return big_pow_bignum_exp(self, rhs);
                    }
                    return Err(format!(
                        "Power result exceeds exact bignum limit (max {} bits)",
                        MAX_EXACT_BIGNUM_BITS
                    ));
                }
            }
        }

        if let (Some((m, s)), CalcValue::FastInt(exp)) = (to_decimal_parts(self), rhs) {
            if *exp >= 0 && *exp <= 100 {
                if let Ok((res_m, res_s)) = decimal_pow(m, s, *exp as u32) {
                    return Ok(CalcValue::normalize_decimal(res_m, res_s));
                }
            }
        }

        let base_f = self.to_f64();
        let exp_f = rhs.to_f64();
        if base_f < 0.0 && exp_f.fract() != 0.0 {
            // Negative real base raised to fractional exponent -> Complex result!
            let r = (-base_f).powf(exp_f);
            let ang = std::f64::consts::PI * exp_f;
            return Ok(CalcValue::from_complex(r * ang.cos(), r * ang.sin()));
        }

        Ok(CalcValue::FastFloat(base_f.powf(exp_f)))
    }

    pub fn format_radix(&self, radix: crate::engine::bignum::RadixMode) -> String {
        use crate::engine::bignum::{format_limbs_dec, format_limbs_hex, format_limbs_oct, format_limbs_bin, RadixMode};
        match self {
            CalcValue::FastInt(i) => match radix {
                RadixMode::Dec => i.to_string(),
                RadixMode::Hex => {
                    if *i >= 0 {
                        format!("0x{:X}", i)
                    } else {
                        format!("-0x{:X}", i.unsigned_abs())
                    }
                }
                RadixMode::Oct => {
                    if *i >= 0 {
                        format!("0o{:o}", i)
                    } else {
                        format!("-0o{:o}", i.unsigned_abs())
                    }
                }
                RadixMode::Bin => {
                    if *i >= 0 {
                        format_limbs_bin(&[*i as u64], true, true)
                    } else {
                        format!("-{}", format_limbs_bin(&[i.unsigned_abs()], true, true))
                    }
                }
            },
            CalcValue::ExactDecimal { mantissa, scale } => match radix {
                RadixMode::Dec => format_exact_decimal(*mantissa, *scale),
                _ => format_exact_decimal_radix(*mantissa, *scale, radix),
            },
            CalcValue::FastFloat(fl) => {
                let snapped = snap_float(*fl);
                let rounded = snapped.round();
                if (snapped - rounded).abs() < 1e-12 && snapped.abs() < 1e15 {
                    let as_int = rounded as i64;
                    CalcValue::FastInt(as_int).format_radix(radix)
                } else {
                    match radix {
                        RadixMode::Dec => format_clean_float(snapped),
                        _ => format_float_radix(snapped, radix),
                    }
                }
            }
            CalcValue::InlineBigInt { sign, limb_count, limbs } => {
                let valid_limbs = &limbs[..*limb_count as usize];
                let body = match radix {
                    RadixMode::Dec => format_limbs_dec(valid_limbs),
                    RadixMode::Hex => format_limbs_hex(valid_limbs, true),
                    RadixMode::Oct => format_limbs_oct(valid_limbs, true),
                    RadixMode::Bin => format_limbs_bin(valid_limbs, true, true),
                };
                if body.is_empty() {
                    String::new()
                } else if *sign {
                    format!("-{}", body)
                } else {
                    body
                }
            }
            CalcValue::HeapBigInt { sign, limbs } => {
                let body = match radix {
                    RadixMode::Dec => format_limbs_dec(limbs),
                    RadixMode::Hex => format_limbs_hex(limbs, true),
                    RadixMode::Oct => format_limbs_oct(limbs, true),
                    RadixMode::Bin => format_limbs_bin(limbs, true, true),
                };
                if body.is_empty() {
                    String::new()
                } else if *sign {
                    format!("-{}", body)
                } else {
                    body
                }
            }
            CalcValue::BigIntSci { sign, sci, mod10_19, bit_len } => {
                if crate::engine::evaluator::is_task_aborted() {
                    return String::new();
                }
                let pfx = if *sign { "-" } else { "" };
                let total_digits = sci.total_digits();
                match radix {
                    RadixMode::Dec => format!(
                        "{}{}…{:019} ({} digits | {} bits)",
                        pfx, sci.mant, mod10_19, total_digits, bit_len
                    ),
                    RadixMode::Hex => format!(
                        "{}0x{:X}… ({} hex digits | {} bits)",
                        pfx,
                        sci.mant,
                        (*bit_len + 3) / 4,
                        bit_len
                    ),
                    RadixMode::Oct => format!(
                        "{}0o{:o}… ({} oct digits | {} bits)",
                        pfx,
                        sci.mant,
                        (*bit_len + 2) / 3,
                        bit_len
                    ),
                    RadixMode::Bin => format!(
                        "{}0b{:b}… ({} bits)",
                        pfx, sci.mant, bit_len
                    ),
                }
            }
            CalcValue::Matrix { rows, cols, data } => {
                use std::fmt::Write;
                let mut out = String::with_capacity(rows * cols * 8 + 4);
                out.push('[');
                for r in 0..*rows {
                    if r > 0 { out.push_str("; "); }
                    for c in 0..*cols {
                        if c > 0 { out.push_str(", "); }
                        let val = data[r * cols + c];
                        let cleaned = (val * 1e12).round() / 1e12;
                        if cleaned.fract() == 0.0 && cleaned.abs() < 1e15 {
                            let _ = write!(out, "{}", cleaned as i64);
                        } else {
                            let _ = write!(out, "{}", cleaned);
                        }
                    }
                }
                out.push(']');
                out
            }
            CalcValue::Logic { table, var_count, tautology, contradiction } => {
                if *tautology {
                    "Tautology (Always TRUE)".to_string()
                } else if *contradiction {
                    "Contradiction (Always FALSE)".to_string()
                } else {
                    match radix {
                        RadixMode::Bin => format_limbs_bin(&[*table], true, true),
                        _ => format!("Contingent (0x{:016X}, {} vars)", table, var_count),
                    }
                }
            }
            CalcValue::Factored { display, .. } => display.clone(),
            CalcValue::Complex { re, im } => format_complex(*re, *im),
            CalcValue::Polynomial { coeffs } => format_polynomial(coeffs),
        }
    }

    /// Formats value for auto-expanding multi-line text editor display
    pub fn format_multiline(&self, radix: RadixMode) -> String {
        match self {
            CalcValue::Matrix { rows, cols, data } => {
                use std::fmt::Write;
                if *rows == 0 || *cols == 0 { return "[]".to_string(); }
                let mut out = String::with_capacity(rows * (cols * 9 + 8));
                for r in 0..*rows {
                    out.push_str("[  ");
                    for c in 0..*cols {
                        if c > 0 { out.push_str(",  "); }
                        let val = data[r * cols + c];
                        let cleaned = (val * 1e10).round() / 1e10;
                        if cleaned.fract() == 0.0 && cleaned.abs() < 1e15 {
                            let _ = write!(out, "{:>4}", cleaned as i64);
                        } else {
                            let _ = write!(out, "{:>6.3}", cleaned);
                        }
                    }
                    if r == *rows - 1 {
                        out.push_str("  ]");
                    } else {
                        out.push_str("  ]\n");
                    }
                }
                out
            }
            CalcValue::Logic { table, var_count, tautology, contradiction } => {
                let status_header = if *tautology {
                    "Tautology (Always TRUE)".to_string()
                } else if *contradiction {
                    "Contradiction (Always FALSE)".to_string()
                } else {
                    let mask = if *var_count <= 4 { (1u64 << (1u64 << *var_count)) - 1 } else { !0u64 };
                    format!("Contingent (Satisfiable | 0x{:04X})", table & mask)
                };
                let mut out = status_header;
                out.push('\n');
                let vars = ["p", "q", "r", "s"];
                let n = (*var_count as usize).clamp(1, 4);
                let total_rows = 1 << n;
                for i in 0..n {
                    out.push_str(&format!(" {} |", vars[i]));
                }
                out.push_str(" Result\n");
                for _ in 0..n {
                    out.push_str("---+");
                }
                out.push_str("-------\n");
                for row in 0..total_rows {
                    for i in 0..n {
                        let bit = (row >> (n - 1 - i)) & 1;
                        out.push_str(if bit == 1 { " T |" } else { " F |" });
                    }
                    let res_bit = (table >> row) & 1;
                    out.push_str(if res_bit == 1 { "   T\n" } else { "   F\n" });
                }
                if out.ends_with('\n') {
                    out.pop();
                }
                out
            }
            CalcValue::HeapBigInt { .. } | CalcValue::InlineBigInt { .. } => {
                let raw = self.format_radix(radix);
                if raw.len() > 60 {
                    let mut out = String::with_capacity(raw.len() + raw.len() / 60 * 2);
                    let chars: Vec<char> = raw.chars().collect();
                    for (idx, chunk) in chars.chunks(60).enumerate() {
                        if idx > 0 { out.push('\n'); }
                        out.extend(chunk);
                    }
                    out
                } else {
                    raw
                }
            }
            _ => self.format_radix(radix),
        }
    }

    /// Decomposes value into exact reduced rational fraction (numerator, denominator)
    pub fn to_rational(&self) -> Option<(i128, i128)> {
        match self {
            CalcValue::FastInt(i) => Some((*i as i128, 1)),
            CalcValue::ExactDecimal { mantissa, scale } => {
                let s = *scale as usize;
                if s < POW10_I128.len() {
                    let factor = POW10_I128[s];
                    fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
                        a = a.abs();
                        b = b.abs();
                        while b != 0 {
                            let t = b;
                            b = a % b;
                            a = t;
                        }
                        a
                    }
                    let g = gcd_i128(*mantissa, factor);
                    let num = mantissa / g;
                    let den = factor / g;
                    if den > 1 {
                        return Some((num, den));
                    }
                }
                None
            }
            CalcValue::FastFloat(fl) => {
                if !fl.is_finite() || fl.abs() > 1e12 { return None; }
                let snapped = snap_float(*fl);
                if snapped.fract() == 0.0 { return None; }
                let sign = if snapped < 0.0 { -1i128 } else { 1i128 };
                let x = snapped.abs();
                let mut m00 = 1i128;
                let mut m01 = 0i128;
                let mut m10 = 0i128;
                let mut m11 = 1i128;
                let mut rem = x;
                for _ in 0..16 {
                    let a = rem.floor() as i128;
                    let t0 = m00 * a + m01;
                    let t1 = m10 * a + m11;
                    m01 = m00;
                    m00 = t0;
                    m11 = m10;
                    m10 = t1;
                    let frac = rem - rem.floor();
                    if frac < 1e-9 || m10 > 1_000_000_000 {
                        break;
                    }
                    rem = 1.0 / frac;
                }
                if m10 > 1 && (x - (m00 as f64 / m10 as f64)).abs() < 1e-6 {
                    Some((sign * m00, m10))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Formats exact rational fraction in prime-factorized `"num/den"` form (`"11²/5"`, `"1/2³"`, `"11×499/(2×5²)"`),
    /// preserving compact power-of-10 factors (`>= 3` repeating trailing zeros, e.g. `"1/10⁵"`, `"3/(5×10⁵)"`).
    pub fn format_rational(&self) -> Option<String> {
        let (num, den) = self.to_rational()?;
        if den > 1 {
            Some(format!(
                "{}/{}",
                format_prime_rational_term(num, false),
                format_prime_rational_term(den, true)
            ))
        } else {
            None
        }
    }

    /// Returns prime factorization string as Unicode factor powers (e.g. "2² × 7")
    pub fn format_prime_factors(&self) -> Option<String> {
        if let CalcValue::Factored { display, .. } = self {
            return Some(display.clone());
        }
        let n: u64 = match self {
            CalcValue::FastInt(i) if *i > 1 => *i as u64,
            CalcValue::ExactDecimal { mantissa, scale: 0 } if *mantissa > 1 => *mantissa as u64,
            CalcValue::InlineBigInt { sign: false, limb_count: 1, limbs } if limbs[0] > 1 => limbs[0],
            _ => return None,
        };
        if n > (1u64 << 60) {
            return None;
        }
        let out = prime_factors_decomposition_str(n);
        if crate::engine::evaluator::is_task_aborted() {
            return None;
        }
        Some(out)
    }
}

impl fmt::Display for CalcValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_radix(crate::engine::bignum::RadixMode::Dec))
    }
}

/// Converts a 128-bit `ExactDecimal` (`mantissa * 10^-scale`) into exact fractional
/// `Hex` (`0x...`), `Oct` (`0o...`), or `Bin` (`0b...`) representation using pure 128-bit integer arithmetic.
fn format_exact_decimal_radix(
    mantissa: i128,
    scale: u32,
    radix: crate::engine::bignum::RadixMode,
) -> String {
    use crate::engine::bignum::{POW10_I128, RadixMode};
    if mantissa == 0 {
        return match radix {
            RadixMode::Dec => "0".to_string(),
            RadixMode::Hex => "0x0".to_string(),
            RadixMode::Oct => "0o0".to_string(),
            RadixMode::Bin => "0b0".to_string(),
        };
    }
    let neg = mantissa < 0;
    let pfx = if neg { "-" } else { "" };
    let u = mantissa.unsigned_abs();
    let denom = POW10_I128
        .get(scale as usize)
        .copied()
        .unwrap_or(1)
        .unsigned_abs()
        .max(1);
    let int_part = u / denom;
    let mut rem = u % denom;

    let (base, rad_pfx, max_frac) = match radix {
        RadixMode::Hex => (16u128, "0x", 12usize),
        RadixMode::Oct => (8u128, "0o", 16usize),
        RadixMode::Bin => (2u128, "0b", 24usize),
        RadixMode::Dec => return crate::engine::bignum::format_exact_decimal(mantissa, scale),
    };

    let int_str = match radix {
        RadixMode::Hex => format!("{int_part:X}"),
        RadixMode::Oct => format!("{int_part:o}"),
        RadixMode::Bin => format!("{int_part:b}"),
        RadixMode::Dec => unreachable!(),
    };

    if rem == 0 {
        return format!("{pfx}{rad_pfx}{int_str}");
    }

    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut frac_buf = String::with_capacity(max_frac);
    for _ in 0..max_frac {
        if rem == 0 {
            break;
        }
        let prod = rem.saturating_mul(base);
        let d = (prod / denom).min(15) as usize;
        rem = prod % denom;
        frac_buf.push(DIGITS[d] as char);
    }
    while frac_buf.ends_with('0') && frac_buf.len() > 1 {
        frac_buf.pop();
    }
    format!("{pfx}{rad_pfx}{int_str}.{frac_buf}")
}

/// Converts a finite non-integer `f64` into fractional `Hex` (`0x...`), `Oct` (`0o...`), or `Bin` (`0b...`).
fn format_float_radix(val: f64, radix: crate::engine::bignum::RadixMode) -> String {
    use crate::engine::bignum::RadixMode;
    if !val.is_finite() || val.abs() >= (u64::MAX as f64) {
        return format_clean_float(val);
    }
    let neg = val < 0.0;
    let pfx = if neg { "-" } else { "" };
    let abs_v = val.abs();
    let int_part = abs_v.trunc() as u64;
    let mut frac = abs_v.fract();

    let (base_f, rad_pfx, max_frac) = match radix {
        RadixMode::Hex => (16.0_f64, "0x", 12usize),
        RadixMode::Oct => (8.0_f64, "0o", 16usize),
        RadixMode::Bin => (2.0_f64, "0b", 24usize),
        RadixMode::Dec => return format_clean_float(val),
    };

    let int_str = match radix {
        RadixMode::Hex => format!("{int_part:X}"),
        RadixMode::Oct => format!("{int_part:o}"),
        RadixMode::Bin => format!("{int_part:b}"),
        RadixMode::Dec => unreachable!(),
    };

    if frac < 1e-15 {
        return format!("{pfx}{rad_pfx}{int_str}");
    }

    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut frac_buf = String::with_capacity(max_frac);
    for _ in 0..max_frac {
        if frac < 1e-12 {
            break;
        }
        let prod = frac * base_f;
        let d = (prod.trunc() as usize).min(15);
        frac = prod - (d as f64);
        frac_buf.push(DIGITS[d] as char);
    }
    while frac_buf.ends_with('0') && frac_buf.len() > 1 {
        frac_buf.pop();
    }
    format!("{pfx}{rad_pfx}{int_str}.{frac_buf}")
}

// =========================================================================
// Complex & Symbolic Polynomial Engines
// =========================================================================

pub fn format_complex(re: f64, im: f64) -> String {
    let r = snap_float(re);
    let i = snap_float(im);
    if i.abs() < 1e-12 {
        return format_clean_float(r);
    }
    if r.abs() < 1e-12 {
        if (i - 1.0).abs() < 1e-12 {
            return "i".to_string();
        }
        if (i + 1.0).abs() < 1e-12 {
            return "-i".to_string();
        }
        return format!("{}i", format_clean_float(i));
    }
    let re_str = format_clean_float(r);
    let i_abs = i.abs();
    let im_str = if (i_abs - 1.0).abs() < 1e-12 {
        "i".to_string()
    } else {
        format!("{}i", format_clean_float(i_abs))
    };
    let op = if i >= 0.0 { " + " } else { " - " };
    format!("{}{}{}", re_str, op, im_str)
}

fn exp_to_sup(exp: usize) -> String {
    let s = exp.to_string();
    s.chars()
        .map(|c| match c {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            _ => c,
        })
        .collect()
}

/// Formats an integer numerator or denominator in an equivalence ratio `P/Q` using prime-power
/// factorization without repetition (`121` -> `11²`, `50` -> `2×5²`), while preserving compact
/// `10^k` factors when `k >= 3` repeating trailing zeros are present (`500000` -> `5×10⁵`).
fn format_prime_rational_term(val: i128, wrap_product_parens: bool) -> String {
    if val == 0 {
        return "0".to_string();
    }
    let neg = val < 0;
    let pfx = if neg { "-" } else { "" };
    let full_u = val.unsigned_abs();
    if full_u == 1 {
        return format!("{pfx}1");
    }

    let mut u = full_u;
    let mut zeros = 0usize;
    while u >= 10 && u.is_multiple_of(10) {
        u /= 10;
        zeros += 1;
    }

    let (rem_to_factor, keep_pow10) = if zeros >= 3 {
        (u, Some(zeros))
    } else {
        (full_u, None)
    };

    let mut factors_out: Vec<String> = Vec::with_capacity(6);
    if rem_to_factor > 1 {
        if rem_to_factor <= (1u128 << 60) {
            let decomp = prime_factors_decomposition(rem_to_factor as u64);
            if !decomp.is_empty() && !crate::engine::evaluator::is_task_aborted() {
                for (p, exp) in decomp {
                    if exp == 1 {
                        factors_out.push(p.to_string());
                    } else {
                        factors_out.push(format!("{p}{}", exp_to_sup(exp as usize)));
                    }
                }
            } else {
                factors_out.push(rem_to_factor.to_string());
            }
        } else {
            factors_out.push(rem_to_factor.to_string());
        }
    }

    if let Some(z) = keep_pow10 {
        factors_out.push(format!("10{}", exp_to_sup(z)));
    }

    let joined = factors_out.join("×");
    if wrap_product_parens && (factors_out.len() >= 2 || neg) {
        format!("({pfx}{joined})")
    } else {
        format!("{pfx}{joined}")
    }
}

pub fn format_polynomial(coeffs: &[f64]) -> String {
    let mut terms: Vec<String> = Vec::new();
    for (deg, &c_raw) in coeffs.iter().enumerate().rev() {
        let c = snap_float(c_raw);
        if c.abs() < 1e-12 {
            continue;
        }
        let is_neg = c < 0.0;
        let abs_c = c.abs();
        let coeff_str = if deg == 0 || (abs_c - 1.0).abs() >= 1e-12 {
            format_clean_float(abs_c)
        } else {
            String::new()
        };
        let var_str = if deg == 0 {
            String::new()
        } else if deg == 1 {
            "x".to_string()
        } else {
            format!("x{}", exp_to_sup(deg))
        };
        let body = format!("{}{}", coeff_str, var_str);
        if terms.is_empty() {
            if is_neg {
                terms.push(format!("-{}", body));
            } else {
                terms.push(body);
            }
        } else if is_neg {
            terms.push(format!(" - {}", body));
        } else {
            terms.push(format!(" + {}", body));
        }
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.concat()
    }
}

pub fn poly_div(num: &[f64], den: &[f64]) -> Result<CalcValue, String> {
    let mut d_deg = den.len();
    while d_deg > 0 && den[d_deg - 1].abs() < 1e-12 {
        d_deg -= 1;
    }
    if d_deg == 0 {
        return Err("Polynomial division by zero".to_string());
    }
    // Scalar divisor
    if d_deg == 1 {
        let s = den[0];
        let out: Vec<f64> = num.iter().map(|&c| c / s).collect();
        return Ok(CalcValue::from_polynomial(out));
    }

    let mut rem = num.to_vec();
    let mut r_deg = rem.len();
    while r_deg > 0 && rem[r_deg - 1].abs() < 1e-12 {
        r_deg -= 1;
    }
    if r_deg < d_deg {
        return Err("Polynomial numerator degree is less than divisor degree".to_string());
    }

    let q_len = r_deg - d_deg + 1;
    let mut quot = vec![0.0; q_len];
    let lead_d = den[d_deg - 1];

    for k in (0..q_len).rev() {
        let cur_coeff = rem[k + d_deg - 1] / lead_d;
        quot[k] = cur_coeff;
        if cur_coeff.abs() > 1e-14 {
            for j in 0..d_deg {
                rem[k + j] -= cur_coeff * den[j];
            }
        }
    }

    if rem.iter().any(|&r| r.abs() > 1e-8) {
        return Err("Non-zero remainder in symbolic polynomial division".to_string());
    }
    Ok(CalcValue::from_polynomial(quot))
}

pub fn poly_deriv(coeffs: &[f64]) -> CalcValue {
    if coeffs.len() <= 1 {
        return CalcValue::FastInt(0);
    }
    let mut out = Vec::with_capacity(coeffs.len() - 1);
    for (deg, &c) in coeffs.iter().enumerate().skip(1) {
        out.push(c * (deg as f64));
    }
    CalcValue::from_polynomial(out)
}

pub fn poly_integ(coeffs: &[f64]) -> CalcValue {
    let mut out = Vec::with_capacity(coeffs.len() + 1);
    out.push(0.0);
    for (deg, &c) in coeffs.iter().enumerate() {
        out.push(c / ((deg + 1) as f64));
    }
    CalcValue::from_polynomial(out)
}

pub fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

pub fn poly_roots(coeffs: &[f64]) -> Result<Vec<(f64, f64)>, String> {
    let mut d = coeffs.len();
    while d > 0 && coeffs[d - 1].abs() < 1e-12 {
        d -= 1;
    }
    if d <= 1 {
        return Err("Constant polynomial has no roots".to_string());
    }
    let deg = d - 1;
    let lead = coeffs[deg];
    let norm: Vec<f64> = coeffs[..d].iter().map(|&c| c / lead).collect();

    if deg == 1 {
        return Ok(vec![(snap_float(-norm[0]), 0.0)]);
    }
    if deg == 2 {
        let b = norm[1];
        let c = norm[0];
        let disc = b * b - 4.0 * c;
        if disc >= -1e-14 {
            let s = disc.max(0.0).sqrt();
            let mut r1 = snap_float((-b - s) * 0.5);
            let mut r2 = snap_float((-b + s) * 0.5);
            if r1 > r2 {
                std::mem::swap(&mut r1, &mut r2);
            }
            return Ok(vec![(r1, 0.0), (r2, 0.0)]);
        } else {
            let re = snap_float(-b * 0.5);
            let im = snap_float((-disc).sqrt() * 0.5);
            return Ok(vec![(re, -im), (re, im)]);
        }
    }

    // Durand-Kerner / Weierstrass simultaneous root finding for degree >= 3
    let mut roots: Vec<(f64, f64)> = (0..deg)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / (deg as f64) + 0.4;
            let r = 0.9;
            (r * angle.cos(), r * angle.sin())
        })
        .collect();

    for _iter in 0..120 {
        crate::engine::evaluator::check_task_aborted()?;
        let mut max_delta = 0.0f64;
        for i in 0..deg {
            let (zr, zi) = roots[i];
            // Evaluate monic polynomial P(z) via Horner's method in complex arithmetic
            let mut pr = 1.0;
            let mut pi = 0.0;
            for k in (0..deg).rev() {
                let nr = pr * zr - pi * zi + norm[k];
                let ni = pr * zi + pi * zr;
                pr = nr;
                pi = ni;
            }
            // Product prod_{j != i} (z_i - z_j)
            let mut dr = 1.0;
            let mut di = 0.0;
            for (j, &(ojr, oji)) in roots.iter().enumerate() {
                if i != j {
                    let diff_r = zr - ojr;
                    let diff_i = zi - oji;
                    let nr = dr * diff_r - di * diff_i;
                    let ni = dr * diff_i + di * diff_r;
                    dr = nr;
                    di = ni;
                }
            }
            let denom = dr * dr + di * di;
            if denom > 1e-28 {
                let qr = (pr * dr + pi * di) / denom;
                let qi = (pi * dr - pr * di) / denom;
                roots[i].0 -= qr;
                roots[i].1 -= qi;
                max_delta = max_delta.max(qr.hypot(qi));
            }
        }
        if max_delta < 1e-13 {
            break;
        }
    }

    for r in &mut roots {
        r.0 = snap_float(r.0);
        r.1 = snap_float(r.1);
        if r.1.abs() < 1e-9 {
            r.1 = 0.0;
        }
    }
    roots.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(roots)
}

// =========================================================================
// Arbitrary Precision & SVO (Small Vector Optimization) BigNum Helpers
// =========================================================================

enum LimbSlice<'a> {
    Stack([u64; 4], usize),
    Heap(&'a [u64]),
}

impl<'a> LimbSlice<'a> {
    #[inline(always)]
    fn as_slice(&self) -> &[u64] {
        match self {
            LimbSlice::Stack(arr, len) => &arr[..*len],
            LimbSlice::Heap(s) => s,
        }
    }
}

#[inline]
fn rem_limbs_u64(limbs: &[u64], divisor: u64) -> u64 {
    if divisor == 0 {
        return 0;
    }
    let mut rem = 0u128;
    let d = divisor as u128;
    for &limb in limbs.iter().rev() {
        let cur = (rem << 64) | (limb as u128);
        rem = cur % d;
    }
    rem as u64
}

#[inline]
fn extract_u64_divisor(v: &CalcValue) -> Option<(bool, u64)> {
    match v {
        CalcValue::FastInt(i) if *i != 0 => Some((*i < 0, i.unsigned_abs())),
        CalcValue::ExactDecimal { mantissa, scale: 0 }
            if *mantissa != 0 && mantissa.unsigned_abs() <= u64::MAX as u128 =>
        {
            Some((*mantissa < 0, mantissa.unsigned_abs() as u64))
        }
        CalcValue::InlineBigInt { sign, limb_count: 1, limbs } if limbs[0] != 0 => {
            Some((*sign, limbs[0]))
        }
        CalcValue::Factored { original, .. } if *original != 0 => Some((false, *original)),
        _ => None,
    }
}

#[inline]
fn extract_limbs_svo(val: &CalcValue) -> Option<(bool, LimbSlice<'_>)> {
    match val {
        CalcValue::FastInt(i) => {
            let sign = *i < 0;
            Some((sign, LimbSlice::Stack([i.unsigned_abs(), 0, 0, 0], 1)))
        }
        CalcValue::ExactDecimal { mantissa, scale: 0 } => {
            let sign = *mantissa < 0;
            let u = mantissa.unsigned_abs();
            let lo = u as u64;
            let hi = (u >> 64) as u64;
            if hi == 0 {
                Some((sign, LimbSlice::Stack([lo, 0, 0, 0], 1)))
            } else {
                Some((sign, LimbSlice::Stack([lo, hi, 0, 0], 2)))
            }
        }
        CalcValue::Factored { original, .. } => {
            Some((false, LimbSlice::Stack([*original, 0, 0, 0], 1)))
        }
        CalcValue::InlineBigInt { sign, limb_count, limbs } => {
            let count = (*limb_count as usize).clamp(1, 4);
            Some((*sign, LimbSlice::Stack(*limbs, count)))
        }
        CalcValue::HeapBigInt { sign, limbs } => Some((*sign, LimbSlice::Heap(limbs.as_slice()))),
        _ => None,
    }
}

fn pack_bignum(sign: bool, mut limbs: Vec<u64>) -> CalcValue {
    while limbs.len() > 1 && limbs.last() == Some(&0) {
        limbs.pop();
    }
    if limbs.is_empty() || (limbs.len() == 1 && limbs[0] == 0) {
        return CalcValue::FastInt(0);
    }
    if limbs.len() == 1 && limbs[0] <= i64::MAX as u64 {
        let v = limbs[0] as i64;
        return CalcValue::FastInt(if sign { -v } else { v });
    }
    if limbs.len() <= 4 {
        let mut out = [0u64; 4];
        for (i, &l) in limbs.iter().enumerate() {
            out[i] = l;
        }
        CalcValue::InlineBigInt {
            sign,
            limb_count: limbs.len() as u8,
            limbs: out,
        }
    } else {
        CalcValue::HeapBigInt {
            sign,
            limbs,
        }
    }
}

#[inline]
fn pack_bignum_slice(sign: bool, limbs: &[u64]) -> CalcValue {
    let mut len = limbs.len();
    while len > 1 && limbs[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 || (len == 1 && limbs[0] == 0) {
        return CalcValue::FastInt(0);
    }
    if len == 1 && limbs[0] <= i64::MAX as u64 {
        let v = limbs[0] as i64;
        return CalcValue::FastInt(if sign { -v } else { v });
    }
    if len <= 4 {
        let mut out = [0u64; 4];
        out[..len].copy_from_slice(&limbs[..len]);
        CalcValue::InlineBigInt {
            sign,
            limb_count: len as u8,
            limbs: out,
        }
    } else {
        CalcValue::HeapBigInt {
            sign,
            limbs: limbs[..len].to_vec(),
        }
    }
}

fn cmp_limbs(a: &[u64], b: &[u64]) -> i32 {
    let mut a_len = a.len();
    while a_len > 0 && a[a_len - 1] == 0 { a_len -= 1; }
    let mut b_len = b.len();
    while b_len > 0 && b[b_len - 1] == 0 { b_len -= 1; }
    if a_len != b_len {
        return if a_len > b_len { 1 } else { -1 };
    }
    for i in (0..a_len).rev() {
        if a[i] > b[i] { return 1; }
        if a[i] < b[i] { return -1; }
    }
    0
}

const MOD_10E19: u128 = 10_000_000_000_000_000_000;

#[inline]
fn pow_mod_10e19(base: u64, mut exp: u64) -> u64 {
    let mut res = 1u128;
    let mut b = (base as u128) % MOD_10E19;
    while exp > 0 {
        if exp & 1 == 1 {
            res = (res * b) % MOD_10E19;
        }
        if exp > 1 {
            b = (b * b) % MOD_10E19;
        }
        exp >>= 1;
    }
    res as u64
}

fn to_big_sci_parts(val: &CalcValue) -> Option<(bool, crate::engine::bignum::DecSci, u64, u64)> {
    match val {
        CalcValue::BigIntSci {
            sign,
            sci,
            mod10_19,
            bit_len,
        } => Some((*sign, *sci, *mod10_19, *bit_len)),
        _ => {
            let (sign, limbs_ref) = extract_limbs_svo(val)?;
            let limbs = limbs_ref.as_slice();
            let sci = crate::engine::bignum::DecSci::from_limbs(limbs);
            let m = crate::engine::bignum::limbs_mod_10e19(limbs);
            let bits = val.bit_length_estimate();
            Some((sign, sci, m, bits))
        }
    }
}

fn big_sci_add(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    let (s1, sci1, m1, b1) = to_big_sci_parts(a).ok_or("Invalid BigIntSci operand")?;
    let (s2, sci2, m2, b2) = to_big_sci_parts(b).ok_or("Invalid BigIntSci operand")?;
    if sci1.mant == 0 {
        return Ok(b.clone());
    }
    if sci2.mant == 0 {
        return Ok(a.clone());
    }
    if s1 == s2 {
        let (big_sci, small_sci) = if sci1.exp10 > sci2.exp10
            || (sci1.exp10 == sci2.exp10 && sci1.mant >= sci2.mant)
        {
            (sci1, sci2)
        } else {
            (sci2, sci1)
        };
        let exp_diff = (big_sci.exp10 - small_sci.exp10).max(0) as u64;
        let out_sci = if exp_diff >= 20 {
            big_sci
        } else {
            let div = 10u128.pow(exp_diff as u32);
            let sum_mant = (big_sci.mant as u128) + (small_sci.mant as u128) / div;
            if sum_mant >= 10_000_000_000_000_000_000 {
                crate::engine::bignum::DecSci {
                    mant: ((sum_mant + 5) / 10) as u64,
                    exp10: big_sci.exp10 + 1,
                }
            } else {
                crate::engine::bignum::DecSci {
                    mant: sum_mant as u64,
                    exp10: big_sci.exp10,
                }
            }
        };
        let mod10_19 = ((m1 as u128 + m2 as u128) % MOD_10E19) as u64;
        Ok(CalcValue::BigIntSci {
            sign: s1,
            sci: out_sci,
            mod10_19,
            bit_len: b1.max(b2),
        })
    } else {
        let cmp_bigger = sci1.exp10 > sci2.exp10
            || (sci1.exp10 == sci2.exp10 && sci1.mant > sci2.mant)
            || (sci1.exp10 == sci2.exp10 && sci1.mant == sci2.mant && m1 >= m2);
        let (big_s, big_sci, big_m, small_sci, small_m, max_b) = if cmp_bigger {
            (s1, sci1, m1, sci2, m2, b1)
        } else {
            (s2, sci2, m2, sci1, m1, b2)
        };
        if big_sci.exp10 == small_sci.exp10 && big_sci.mant == small_sci.mant && big_m == small_m {
            return Ok(CalcValue::FastInt(0));
        }
        let exp_diff = (big_sci.exp10 - small_sci.exp10).max(0) as u64;
        let out_sci = if exp_diff >= 20 {
            big_sci
        } else {
            let div = 10u128.pow(exp_diff as u32);
            let sub_mant = (big_sci.mant as u128).saturating_sub((small_sci.mant as u128) / div);
            crate::engine::bignum::DecSci::from_u128(sub_mant.max(1)).mul(
                crate::engine::bignum::DecSci {
                    mant: 1_000_000_000_000_000_000,
                    exp10: big_sci.exp10 - 18,
                },
            )
        };
        let mod10_19 = ((big_m as u128 + MOD_10E19 - small_m as u128) % MOD_10E19) as u64;
        Ok(CalcValue::BigIntSci {
            sign: big_s,
            sci: out_sci,
            mod10_19,
            bit_len: max_b,
        })
    }
}

fn big_sci_sub(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    let (s2, sci2, m2, b2) = to_big_sci_parts(b).ok_or("Invalid BigIntSci operand")?;
    let neg_b = CalcValue::BigIntSci {
        sign: !s2,
        sci: sci2,
        mod10_19: m2,
        bit_len: b2,
    };
    big_sci_add(a, &neg_b)
}

fn big_sci_mul(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    let (s1, sci1, m1, b1) = to_big_sci_parts(a).ok_or("Invalid BigIntSci operand")?;
    let (s2, sci2, m2, b2) = to_big_sci_parts(b).ok_or("Invalid BigIntSci operand")?;
    if sci1.mant == 0 || sci2.mant == 0 {
        return Ok(CalcValue::FastInt(0));
    }
    let out_sci = sci1.mul(sci2);
    let mod10_19 = ((m1 as u128 * m2 as u128) % MOD_10E19) as u64;
    Ok(CalcValue::BigIntSci {
        sign: s1 != s2,
        sci: out_sci,
        mod10_19,
        bit_len: b1.saturating_add(b2),
    })
}

fn big_sci_div(a: &CalcValue, b: &CalcValue, is_floor: bool) -> Result<CalcValue, String> {
    let (s1, sci1, m1, _b1) = to_big_sci_parts(a).ok_or("Invalid BigIntSci dividend")?;
    let (s2, sci2, m2, _b2) = to_big_sci_parts(b).ok_or("Invalid BigIntSci divisor")?;
    if sci2.mant == 0 {
        return Err(if is_floor {
            "Integer division by zero".to_string()
        } else {
            "Division by zero".to_string()
        });
    }
    if sci1.mant == 0 {
        return Ok(CalcValue::FastInt(0));
    }
    let final_sign = s1 != s2;
    let out_sci = sci1.div(sci2);

    // If |a| < |b| (quotient < 1)
    if out_sci.exp10 < -18 {
        if is_floor {
            return Ok(CalcValue::FastInt(if final_sign { -1 } else { 0 }));
        }
        let exp_diff = (sci1.exp10 - sci2.exp10).clamp(-308, 308) as i32;
        let vf = (sci1.mant as f64 / sci2.mant as f64) * 10.0f64.powi(exp_diff);
        return Ok(CalcValue::from_f64(if final_sign { -vf } else { vf }));
    }

    // If quotient fits in i64 (1..=18 decimal digits)
    if out_sci.exp10 <= -1 {
        let scale_down = 10u64.pow((-out_sci.exp10) as u32);
        let q = (out_sci.mant / scale_down) as i64;
        if !is_floor && out_sci.mant % scale_down != 0 {
            let vf = (out_sci.mant as f64) / (scale_down as f64);
            return Ok(CalcValue::from_f64(if final_sign { -vf } else { vf }));
        }
        return Ok(CalcValue::FastInt(if final_sign { -q } else { q }));
    }

    let mod10_19 = if out_sci.exp10 == 0 {
        out_sci.mant
    } else if m2 != 0 {
        let g = gcd_u64(m2, 10_000_000_000_000_000_000);
        let m2_coprime = m2 / g;
        let m1_reduced = m1 / gcd_u64(m1, g).max(1);
        let inv = pow_mod_10e19(m2_coprime, 4_000_000_000_000_000_000 - 1);
        ((m1_reduced as u128 * inv as u128) % MOD_10E19) as u64
    } else {
        0
    };

    let log2_q = ((out_sci.exp10 as f64) + (out_sci.mant as f64).log10()) * std::f64::consts::LOG2_10;
    let bit_len = (log2_q.ceil().max(1.0).min(u64::MAX as f64)) as u64;

    Ok(CalcValue::BigIntSci {
        sign: final_sign,
        sci: out_sci,
        mod10_19,
        bit_len,
    })
}

fn big_sci_rem(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    let (s1, sci1, m1, _b1) = to_big_sci_parts(a).ok_or("Invalid BigIntSci dividend")?;
    if let Some((_sb, d)) = extract_u64_divisor(b) {
        if d == 0 {
            return Err("Modulo by zero".to_string());
        }
        if d == 1 {
            return Ok(CalcValue::FastInt(0));
        }
        // For any divisor `d` of 10^19 (`d = 2^j * 5^k` with `j, k <= 19`), `m1 % d` is 100% exact
        if 10_000_000_000_000_000_000u64 % d == 0 {
            let rem = m1 % d;
            let euclid_rem = if s1 && rem != 0 { d - rem } else { rem };
            return Ok(pack_bignum_slice(false, &[euclid_rem]));
        }
    }
    let (_s2, sci2, m2, _b2) = to_big_sci_parts(b).ok_or("Invalid BigIntSci divisor")?;
    if sci2.mant == 0 {
        return Err("Modulo by zero".to_string());
    }
    if sci1 == sci2 && m1 == m2 {
        return Ok(CalcValue::FastInt(0));
    }
    if sci1.exp10 < sci2.exp10 || (sci1.exp10 == sci2.exp10 && sci1.mant < sci2.mant) {
        if !s1 {
            return Ok(a.clone());
        }
    }
    let rem = if m2 != 0 { m1 % m2 } else { 0 };
    Ok(pack_bignum_slice(false, &[rem]))
}

fn big_pow_sci(base: &CalcValue, exp: u64) -> Result<CalcValue, String> {
    crate::engine::evaluator::check_task_aborted()?;
    let (base_sign, base_sci, base_mod, base_bits) =
        to_big_sci_parts(base).ok_or("Invalid base for big_pow_sci")?;
    if base_sci.mant == 0 {
        return Ok(CalcValue::FastInt(0));
    }
    let final_sign = base_sign && (exp & 1 == 1);
    let sci = base_sci.pow_u64(exp);
    let mod10_19 = pow_mod_10e19(base_mod, exp);
    let base_log2 = base.log2_mag();
    let bit_len = if base_log2 > 0.0 && base_log2.is_finite() {
        ((exp as f64) * base_log2).ceil().min(u64::MAX as f64) as u64
    } else {
        base_bits.saturating_mul(exp)
    };
    Ok(CalcValue::BigIntSci {
        sign: final_sign,
        sci,
        mod10_19,
        bit_len,
    })
}

fn big_pow_bignum_exp(base: &CalcValue, exp: &CalcValue) -> Result<CalcValue, String> {
    crate::engine::evaluator::check_task_aborted()?;
    let (base_sign, base_sci, base_mod, _) =
        to_big_sci_parts(base).ok_or("Invalid base for bignum power")?;
    if base_sci.mant == 0 {
        return Ok(CalcValue::FastInt(0));
    }
    let (exp_sign, exp_limbs_ref) = match extract_limbs_svo(exp) {
        Some(v) => v,
        None => {
            if let CalcValue::BigIntSci {
                sign,
                sci,
                mod10_19,
                bit_len,
            } = exp
            {
                if *sign {
                    return Ok(CalcValue::FastInt(0));
                }
                let final_sign = base_sign && (mod10_19 & 1 == 1);
                let log10_base = (base_sci.exp10 as f64) + (base_sci.mant as f64).log10();
                let (out_mant, out_exp10, out_bit_len) = if sci.exp10 <= 18 {
                    let exp_val = (sci.mant as f64) * 10.0f64.powi(sci.exp10 as i32);
                    let total_log10 = exp_val * log10_base;
                    if total_log10.is_finite() && total_log10 < (i64::MAX / 2) as f64 {
                        let int_log10 = total_log10.floor();
                        let frac_log10 = total_log10 - int_log10;
                        let mant = (10.0f64.powf(frac_log10 + 18.0).round() as u64)
                            .clamp(1_000_000_000_000_000_000, 9_999_999_999_999_999_999);
                        let exp10 = (int_log10 as i64).saturating_sub(18);
                        let bits = (total_log10 * std::f64::consts::LOG2_10)
                            .ceil()
                            .min(u64::MAX as f64) as u64;
                        (mant, exp10, bits)
                    } else {
                        (base_sci.mant, i64::MAX / 2, u64::MAX)
                    }
                } else {
                    (base_sci.mant, i64::MAX / 2, bit_len.saturating_mul(64).max(u64::MAX / 2))
                };
                // By Carmichael's theorem, lambda(10^19) divides 4 * 10^18, which divides 10^19
                let eff_exp_mod = if *mod10_19 < 19 {
                    *mod10_19 + 4_000_000_000_000_000_000
                } else {
                    *mod10_19
                };
                return Ok(CalcValue::BigIntSci {
                    sign: final_sign,
                    sci: crate::engine::bignum::DecSci {
                        mant: out_mant,
                        exp10: out_exp10,
                    },
                    mod10_19: pow_mod_10e19(base_mod, eff_exp_mod),
                    bit_len: out_bit_len,
                });
            }
            return Err("Invalid exponent for bignum power".to_string());
        }
    };
    if exp_sign {
        return Ok(CalcValue::FastFloat(0.0));
    }
    let exp_limbs = exp_limbs_ref.as_slice();
    let is_odd = exp_limbs.first().map_or(false, |l| (l & 1) == 1);
    let final_sign = base_sign && is_odd;

    // Binary exponentiation over multi-limb exponent in O(bit_length(exp))
    let mut cur_sci = base_sci;
    let mut res_sci = crate::engine::bignum::DecSci::ONE;
    let mut cur_mod = base_mod as u128;
    let mut res_mod = 1u128;

    for (limb_idx, &limb) in exp_limbs.iter().enumerate() {
        crate::engine::evaluator::check_task_aborted()?;
        let bits_in_limb = if limb_idx + 1 == exp_limbs.len() {
            64 - limb.leading_zeros() as usize
        } else {
            64
        };
        for b in 0..bits_in_limb {
            if ((limb >> b) & 1) == 1 {
                res_sci = res_sci.mul(cur_sci);
                res_mod = (res_mod * cur_mod) % MOD_10E19;
            }
            cur_sci = cur_sci.mul(cur_sci);
            cur_mod = (cur_mod * cur_mod) % MOD_10E19;
        }
    }

    let total_dec_digits = res_sci.exp10.saturating_add(19).max(1) as f64;
    let bit_len = (total_dec_digits * std::f64::consts::LOG2_10)
        .ceil()
        .min(u64::MAX as f64) as u64;

    Ok(CalcValue::BigIntSci {
        sign: final_sign,
        sci: res_sci,
        mod10_19: res_mod as u64,
        bit_len,
    })
}

/// Checks if a positive integer `CalcValue` is an exact power of two `2^k` (`k >= 1`), returning `(sign, k)`.
fn exact_power_of_two_shift(val: &CalcValue) -> Option<(bool, u64)> {
    let (sign, limbs_ref) = extract_limbs_svo(val)?;
    let limbs = limbs_ref.as_slice();
    let mut end = limbs.len();
    while end > 0 && limbs[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let top = limbs[end - 1];
    if !top.is_power_of_two() {
        return None;
    }
    for &limb in &limbs[..end - 1] {
        if limb != 0 {
            return None;
        }
    }
    let k = ((end - 1) as u64) * 64 + (top.trailing_zeros() as u64);
    if k == 0 {
        return None;
    }
    Some((sign, k))
}

fn big_add(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    if matches!(a, CalcValue::BigIntSci { .. }) || matches!(b, CalcValue::BigIntSci { .. }) {
        return big_sci_add(a, b);
    }
    let (s1, ref1) = match extract_limbs_svo(a) {
        Some(v) => v,
        None => return Ok(CalcValue::FastFloat(a.to_f64() + b.to_f64())),
    };
    let (s2, ref2) = match extract_limbs_svo(b) {
        Some(v) => v,
        None => return Ok(CalcValue::FastFloat(a.to_f64() + b.to_f64())),
    };
    let l1 = ref1.as_slice();
    let l2 = ref2.as_slice();

    if s1 == s2 {
        let max_len = l1.len().max(l2.len());
        if max_len < 4 {
            let mut stack_out = [0u64; 5];
            let carry = crate::engine::bignum::comba_add(l1, l2, &mut stack_out[..max_len]);
            if carry > 0 {
                stack_out[max_len] = carry;
                return Ok(pack_bignum_slice(s1, &stack_out[..max_len + 1]));
            }
            return Ok(pack_bignum_slice(s1, &stack_out[..max_len]));
        }
        let mut out = vec![0u64; max_len + 1];
        let carry = crate::engine::bignum::comba_add(l1, l2, &mut out[..max_len]);
        if carry > 0 {
            out[max_len] = carry;
        }
        Ok(pack_bignum(s1, out))
    } else {
        let inv_b = pack_bignum_slice(!s2, l2);
        big_sub(a, &inv_b)
    }
}

fn big_sub(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    if matches!(a, CalcValue::BigIntSci { .. }) || matches!(b, CalcValue::BigIntSci { .. }) {
        return big_sci_sub(a, b);
    }
    let (s1, ref1) = match extract_limbs_svo(a) {
        Some(v) => v,
        None => return Ok(CalcValue::FastFloat(a.to_f64() - b.to_f64())),
    };
    let (s2, ref2) = match extract_limbs_svo(b) {
        Some(v) => v,
        None => return Ok(CalcValue::FastFloat(a.to_f64() - b.to_f64())),
    };
    let l1 = ref1.as_slice();
    let l2 = ref2.as_slice();

    if s1 != s2 {
        let inv_b = pack_bignum_slice(!s2, l2);
        return big_add(a, &inv_b);
    }

    let cmp = cmp_limbs(l1, l2);
    if cmp == 0 {
        return Ok(CalcValue::FastInt(0));
    }

    let (larger, smaller, final_sign) = if cmp > 0 {
        (l1, l2, s1)
    } else {
        (l2, l1, !s1)
    };

    if larger.len() <= 4 {
        let mut stack_out = [0u64; 4];
        crate::engine::bignum::comba_sub(larger, smaller, &mut stack_out[..larger.len()]);
        return Ok(pack_bignum_slice(final_sign, &stack_out[..larger.len()]));
    }

    let mut out = vec![0u64; larger.len()];
    crate::engine::bignum::comba_sub(larger, smaller, &mut out);
    Ok(pack_bignum(final_sign, out))
}

/// Instant/standard exact bignum bit-length ceiling (524,288 bits = 8,192 64-bit limbs ≈ 157,826 decimal digits)
pub const MAX_EXACT_BIGNUM_BITS: u64 = 524_288;
pub const MAX_EXACT_BIGNUM_LIMBS: usize = (MAX_EXACT_BIGNUM_BITS / 64) as usize;

/// Settled deep-compute exact bignum bit-length ceiling (4,194,304 bits = 65,536 64-bit limbs ≈ 1,262,611 decimal digits)
pub const MAX_DEEP_BIGNUM_BITS: u64 = 4_194_304;
pub const MAX_DEEP_BIGNUM_LIMBS: usize = (MAX_DEEP_BIGNUM_BITS / 64) as usize;

fn big_mul(a: &CalcValue, b: &CalcValue) -> Result<CalcValue, String> {
    crate::engine::evaluator::check_task_aborted()?;

    if matches!(a, CalcValue::BigIntSci { .. }) || matches!(b, CalcValue::BigIntSci { .. }) {
        return big_sci_mul(a, b);
    }

    let (s1, ref1) = match extract_limbs_svo(a) {
        Some(v) => v,
        None => return Ok(CalcValue::FastFloat(a.to_f64() * b.to_f64())),
    };
    let (s2, ref2) = match extract_limbs_svo(b) {
        Some(v) => v,
        None => return Ok(CalcValue::FastFloat(a.to_f64() * b.to_f64())),
    };
    let l1 = ref1.as_slice();
    let l2 = ref2.as_slice();

    let final_sign = s1 != s2;
    let total_limbs = l1.len() + l2.len();
    let deep_active = crate::engine::evaluator::is_deep_compute_active();
    let max_limbs = if deep_active {
        MAX_DEEP_BIGNUM_LIMBS
    } else {
        MAX_EXACT_BIGNUM_LIMBS
    };

    if total_limbs > max_limbs + 4 {
        if deep_active {
            return big_sci_mul(a, b);
        }
        return Err(format!(
            "Multiplication result exceeds exact bignum limit (max {} bits)",
            MAX_EXACT_BIGNUM_BITS
        ));
    }

    if total_limbs <= 8 {
        let mut stack_out = [0u64; 8];
        crate::engine::bignum::comba_mul(l1, l2, &mut stack_out[..total_limbs]);
        return Ok(pack_bignum_slice(final_sign, &stack_out[..total_limbs]));
    }

    let mut out = vec![0u64; total_limbs];
    let min_len = l1.len().min(l2.len());
    let karatsuba_cutoff = crate::engine::wisdom::karatsuba_cutoff_limbs();
    let par_cutoff = crate::engine::wisdom::toom3_cutoff_limbs();
    let ntt_cutoff = crate::engine::wisdom::ntt_cutoff_limbs();

    if min_len >= ntt_cutoff {
        crate::engine::bignum::ntt_mul(l1, l2, &mut out);
    } else if min_len >= par_cutoff {
        crate::engine::bignum::par_karatsuba_mul(l1, l2, &mut out, karatsuba_cutoff);
    } else if min_len > 16 {
        crate::engine::bignum::karatsuba_mul(l1, l2, &mut out, karatsuba_cutoff);
    } else {
        crate::engine::bignum::comba_mul(l1, l2, &mut out);
    }
    crate::engine::evaluator::check_task_aborted()?;
    Ok(pack_bignum(final_sign, out))
}

fn big_sqr(a: &CalcValue) -> Result<CalcValue, String> {
    crate::engine::evaluator::check_task_aborted()?;
    if matches!(a, CalcValue::BigIntSci { .. }) {
        return big_sci_mul(a, a);
    }
    let (_s1, ref1) = match extract_limbs_svo(a) {
        Some(v) => v,
        None => {
            let f = a.to_f64();
            return Ok(CalcValue::FastFloat(f * f));
        }
    };
    let l1 = ref1.as_slice();
    let total_limbs = l1.len() * 2;
    let deep_active = crate::engine::evaluator::is_deep_compute_active();
    let max_limbs = if deep_active {
        MAX_DEEP_BIGNUM_LIMBS
    } else {
        MAX_EXACT_BIGNUM_LIMBS
    };

    if total_limbs > max_limbs + 4 {
        if deep_active {
            return big_sci_mul(a, a);
        }
        return Err(format!(
            "Multiplication result exceeds exact bignum limit (max {} bits)",
            MAX_EXACT_BIGNUM_BITS
        ));
    }

    if total_limbs <= 8 {
        let mut stack_out = [0u64; 8];
        crate::engine::bignum::comba_mul(l1, l1, &mut stack_out[..total_limbs]);
        return Ok(pack_bignum_slice(false, &stack_out[..total_limbs]));
    }

    let mut out = vec![0u64; total_limbs];
    let len = l1.len();
    let karatsuba_cutoff = crate::engine::wisdom::karatsuba_cutoff_limbs();
    let par_cutoff = crate::engine::wisdom::toom3_cutoff_limbs();
    let ntt_cutoff = crate::engine::wisdom::ntt_cutoff_limbs();

    if len >= ntt_cutoff {
        crate::engine::bignum::ntt_sqr(l1, &mut out);
    } else if len >= par_cutoff {
        crate::engine::bignum::par_karatsuba_mul(l1, l1, &mut out, karatsuba_cutoff);
    } else if len > 16 {
        crate::engine::bignum::karatsuba_mul(l1, l1, &mut out, karatsuba_cutoff);
    } else {
        crate::engine::bignum::comba_mul(l1, l1, &mut out);
    }
    crate::engine::evaluator::check_task_aborted()?;
    Ok(pack_bignum(false, out))
}

fn big_pow(base: &CalcValue, mut exp: u64) -> Result<CalcValue, String> {
    if exp == 0 {
        return Ok(CalcValue::FastInt(1));
    }
    if exp == 1 {
        return Ok(base.clone());
    }
    // O(1) instant fast-paths for -1, 0, 1 raised to any u64 exponent
    if let CalcValue::FastInt(b) = base {
        match *b {
            0 => return Ok(CalcValue::FastInt(0)),
            1 => return Ok(CalcValue::FastInt(1)),
            -1 => return Ok(CalcValue::FastInt(if (exp & 1) == 0 { 1 } else { -1 })),
            _ => {}
        }
    }

    if matches!(base, CalcValue::BigIntSci { .. }) {
        return big_pow_sci(base, exp);
    }

    // Pre-execution exact bit-length ceiling check before allocating or looping
    let base_log2 = base.log2_mag();
    let exact_bits_est = if base_log2 > 0.0 && base_log2.is_finite() {
        ((exp as f64) * base_log2).ceil() as u128
    } else {
        (base.bit_length_estimate().saturating_sub(1).max(1) as u128).saturating_mul(exp as u128)
    };
    let deep_active = crate::engine::evaluator::is_deep_compute_active();
    let max_bits = if deep_active {
        MAX_DEEP_BIGNUM_BITS
    } else {
        MAX_EXACT_BIGNUM_BITS
    };

    if exact_bits_est > max_bits as u128 {
        if deep_active {
            return big_pow_sci(base, exp);
        }
        return Err(format!(
            "Power result exceeds exact bignum limit (~{} bits > {} max bits)",
            exact_bits_est.min(u64::MAX as u128),
            MAX_EXACT_BIGNUM_BITS
        ));
    }

    // O(1) exact bit-shift fast path for powers of two in settled deep compute mode (e.g. 2**1000000, (2**1024)**2000)
    if deep_active {
        if let Some((base_sign, k)) = exact_power_of_two_shift(base) {
            if let Some(total_shift) = k.checked_mul(exp) {
                if total_shift <= MAX_DEEP_BIGNUM_BITS {
                    let limb_idx = (total_shift / 64) as usize;
                    let bit_offset = (total_shift % 64) as u32;
                    let mut limbs = vec![0u64; limb_idx + 1];
                    limbs[limb_idx] = 1u64 << bit_offset;
                    let final_sign = base_sign && (exp & 1 == 1);
                    return Ok(pack_bignum(final_sign, limbs));
                }
            }
        }
    }

    let mut current = base.clone();
    let mut result = CalcValue::FastInt(1);
    while exp > 0 {
        crate::engine::evaluator::check_task_aborted()?;
        if exp & 1 == 1 {
            result = big_mul(&result, &current)?;
            crate::engine::evaluator::check_task_aborted()?;
        }
        if exp > 1 {
            current = big_sqr(&current)?;
            crate::engine::evaluator::check_task_aborted()?;
        }
        exp >>= 1;
    }
    Ok(result)
}

// =========================================================================
// SOTA Deterministic Miller-Rabin & Brent's Pollard Rho Prime Factorization
// =========================================================================

#[inline(always)]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut res = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            res = mul_mod(res, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    res
}

pub fn is_prime_u64(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 || n == 5 || n == 7 { return true; }
    if n % 2 == 0 || n % 3 == 0 || n % 5 == 0 || n % 7 == 0 { return false; }
    if n < 121 { return true; }

    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;

    const BASES: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];
    for &a in &BASES {
        let base = a % n;
        if base == 0 { continue; }
        let mut x = pow_mod(base, d, n);
        if x == 1 || x == n - 1 { continue; }
        let mut composite = true;
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite { return false; }
    }
    true
}

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 { return b; }
    if b == 0 { return a; }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            break;
        }
    }
    a << shift
}

/// Brent's cycle-finding Pollard's rho with batched modular difference accumulation (M = 128)
fn pollard_rho(n: u64) -> u64 {
    if n % 2 == 0 { return 2; }
    let n128 = n as u128;
    let mut c = 1u64;
    loop {
        if crate::engine::evaluator::is_task_aborted() {
            return n;
        }
        let c128 = c as u128;
        let mut y = 2u64;
        let mut r = 1u64;
        let mut q = 1u64;
        let mut g = 1u64;
        let mut x = 2u64;
        let mut ys = 2u64;
        const M: u64 = 128;

        while g == 1 {
            x = y;
            for idx in 0..r {
                if (idx & 1023) == 1023 && crate::engine::evaluator::is_task_aborted() {
                    return n;
                }
                y = ((y as u128 * y as u128 + c128) % n128) as u64;
            }
            let mut k = 0u64;
            while k < r && g == 1 {
                if crate::engine::evaluator::is_task_aborted() {
                    return n;
                }
                ys = y;
                let step = M.min(r - k);
                for _ in 0..step {
                    y = ((y as u128 * y as u128 + c128) % n128) as u64;
                    let diff = if x > y { x - y } else { y - x };
                    q = mul_mod(q, diff, n);
                }
                g = gcd_u64(q, n);
                k += M;
            }
            r <<= 1;
        }

        if g == n {
            let mut step_guard = 0u64;
            loop {
                step_guard += 1;
                if (step_guard & 1023) == 1023 && crate::engine::evaluator::is_task_aborted() {
                    return n;
                }
                ys = ((ys as u128 * ys as u128 + c128) % n128) as u64;
                let diff = if x > ys { x - ys } else { ys - x };
                g = gcd_u64(diff, n);
                if g > 1 {
                    break;
                }
            }
        }
        if g < n {
            return g;
        }
        c += 1;
    }
}

fn factor_large(n: u64, factors: &mut Vec<(u64, u32)>) {
    if n <= 1 || crate::engine::evaluator::is_task_aborted() {
        return;
    }
    if is_prime_u64(n) {
        factors.push((n, 1));
        return;
    }
    let d = pollard_rho(n);
    if d == n || crate::engine::evaluator::is_task_aborted() {
        factors.push((n, 1));
        return;
    }
    factor_large(d, factors);
    factor_large(n / d, factors);
}

pub fn prime_factors_decomposition(mut n: u64) -> Vec<(u64, u32)> {
    if n <= 1 {
        return Vec::new();
    }
    let mut factors = Vec::with_capacity(8);

    // Hardware TZCNT factor-of-2 extraction
    let twos = n.trailing_zeros();
    if twos > 0 {
        factors.push((2, twos));
        n >>= twos;
    }

    let small_primes: [u64; 24] = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ];
    for &p in &small_primes {
        if p * p > n {
            break;
        }
        if n % p == 0 {
            let mut count = 0;
            while n % p == 0 {
                count += 1;
                n /= p;
            }
            factors.push((p, count));
        }
    }

    if n > 1 {
        factor_large(n, &mut factors);
    }

    factors.sort_unstable_by_key(|&(p, _)| p);

    let mut merged: Vec<(u64, u32)> = Vec::with_capacity(factors.len());
    for (p, count) in factors {
        if let Some(last) = merged.last_mut() {
            if last.0 == p {
                last.1 += count;
                continue;
            }
        }
        merged.push((p, count));
    }

    merged
}

pub fn prime_factors_decomposition_str(n: u64) -> String {
    if n <= 1 {
        return n.to_string();
    }
    let decomp = prime_factors_decomposition(n);
    if decomp.is_empty() {
        return n.to_string();
    }
    if decomp.len() == 1 && decomp[0].1 == 1 {
        return format!("{} (Prime)", n);
    }

    let parts: Vec<String> = decomp.iter().map(|&(p, exp)| {
        if exp == 1 {
            p.to_string()
        } else {
            format!("{}{}", p, exp_to_sup(exp as usize))
        }
    }).collect();

    parts.join(" × ")
}
