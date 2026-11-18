//! Pure Mathematics Arbitrary Scaled Decimal Engine.
//! Banking-tier exact base-10 arithmetic without IEEE-754 binary floating-point approximations.
//! Every decimal number is represented by an exact integer mantissa and an exact power-of-10 scale:
//! value = mantissa * 10^(-scale).

pub const POW10_I128: [i128; 39] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
    10_000_000_000_000_000_000,
    100_000_000_000_000_000_000,
    1_000_000_000_000_000_000_000,
    10_000_000_000_000_000_000_000,
    100_000_000_000_000_000_000_000,
    1_000_000_000_000_000_000_000_000,
    10_000_000_000_000_000_000_000_000,
    100_000_000_000_000_000_000_000_000,
    1_000_000_000_000_000_000_000_000_000,
    10_000_000_000_000_000_000_000_000_000,
    100_000_000_000_000_000_000_000_000_000,
    1_000_000_000_000_000_000_000_000_000_000,
    10_000_000_000_000_000_000_000_000_000_000,
    100_000_000_000_000_000_000_000_000_000_000,
    1_000_000_000_000_000_000_000_000_000_000_000,
    10_000_000_000_000_000_000_000_000_000_000_000,
    100_000_000_000_000_000_000_000_000_000_000_000,
    1_000_000_000_000_000_000_000_000_000_000_000_000,
    10_000_000_000_000_000_000_000_000_000_000_000_000,
    100_000_000_000_000_000_000_000_000_000_000_000_000,
];

/// Normalizes an ExactDecimal by stripping redundant trailing zeros from mantissa when scale > 0
#[inline]
pub fn normalize_decimal(mut mantissa: i128, mut scale: u32) -> (i128, u32) {
    if mantissa == 0 {
        return (0, 0);
    }
    while scale > 0 && mantissa % 10 == 0 {
        mantissa /= 10;
        scale -= 1;
    }
    (mantissa, scale)
}

/// Parses a decimal string (e.g. "38.28", "1.5e3", "0.005") into an exact (mantissa, scale) pair
/// Zero-allocation byte scanner (`0` heap allocations on hot path).
#[inline(always)]
pub fn parse_exact_decimal(s: &str) -> Option<(i128, u32)> {
    let bytes = s.trim().as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let mut e_pos: Option<usize> = None;
    for (idx, &b) in bytes.iter().enumerate() {
        if b == b'e' || b == b'E' {
            e_pos = Some(idx);
            break;
        }
    }

    let (base_bytes, exp) = if let Some(ei) = e_pos {
        let exp_str = unsafe { core::str::from_utf8_unchecked(&bytes[ei + 1..]) };
        let exp_val = exp_str.parse::<i32>().ok()?;
        (&bytes[..ei], exp_val)
    } else {
        (bytes, 0i32)
    };

    if base_bytes.is_empty() {
        return None;
    }
    let is_negative = base_bytes[0] == b'-';
    let unsigned_bytes = if is_negative || base_bytes[0] == b'+' {
        &base_bytes[1..]
    } else {
        base_bytes
    };

    let mut unsigned_val: u128 = 0;
    let mut digit_count: usize = 0;
    let mut base_scale: i32 = 0;
    let mut seen_dot = false;

    for &b in unsigned_bytes {
        if b == b'.' {
            if seen_dot {
                return None;
            }
            seen_dot = true;
            continue;
        }
        if !b.is_ascii_digit() {
            return None;
        }
        unsigned_val = unsigned_val.checked_mul(10)?.checked_add((b - b'0') as u128)?;
        digit_count += 1;
        if seen_dot {
            base_scale += 1;
        }
    }

    if digit_count == 0 {
        return None;
    }

    let net_scale = base_scale.checked_sub(exp)?;
    if net_scale < 0 {
        // Exponent expands with trailing zeros: e.g. 1.5e3 -> 1500
        let shift = (-net_scale) as usize;
        if unsigned_val != 0 {
            if shift >= POW10_I128.len() {
                return None;
            }
            unsigned_val = unsigned_val.checked_mul(POW10_I128[shift] as u128)?;
        }
        if unsigned_val > (i128::MAX as u128) {
            return None;
        }
        let mut mantissa = unsigned_val as i128;
        if is_negative {
            mantissa = -mantissa;
        }
        Some(normalize_decimal(mantissa, 0))
    } else {
        if unsigned_val > (i128::MAX as u128) {
            return None;
        }
        let mut mantissa = unsigned_val as i128;
        if is_negative {
            mantissa = -mantissa;
        }
        Some(normalize_decimal(mantissa, net_scale as u32))
    }
}

/// Exact base-10 addition: aligns scales and performs checked integer addition
pub fn decimal_add(m1: i128, s1: u32, m2: i128, s2: u32) -> Result<(i128, u32), String> {
    if s1 == s2 {
        if let Some(sum) = m1.checked_add(m2) {
            return Ok(normalize_decimal(sum, s1));
        }
    } else if s1 > s2 {
        let diff = (s1 - s2) as usize;
        if diff < POW10_I128.len() {
            let factor = POW10_I128[diff];
            if let Some(m2_scaled) = m2.checked_mul(factor) {
                if let Some(sum) = m1.checked_add(m2_scaled) {
                    return Ok(normalize_decimal(sum, s1));
                }
            }
        }
    } else {
        let diff = (s2 - s1) as usize;
        if diff < POW10_I128.len() {
            let factor = POW10_I128[diff];
            if let Some(m1_scaled) = m1.checked_mul(factor) {
                if let Some(sum) = m1_scaled.checked_add(m2) {
                    return Ok(normalize_decimal(sum, s2));
                }
            }
        }
    }
    Err("Decimal sum exceeds 38-digit exact banking limit".to_string())
}

/// Exact base-10 subtraction: aligns scales and performs checked integer subtraction
pub fn decimal_sub(m1: i128, s1: u32, m2: i128, s2: u32) -> Result<(i128, u32), String> {
    if s1 == s2 {
        if let Some(diff) = m1.checked_sub(m2) {
            return Ok(normalize_decimal(diff, s1));
        }
    } else if s1 > s2 {
        let diff = (s1 - s2) as usize;
        if diff < POW10_I128.len() {
            let factor = POW10_I128[diff];
            if let Some(m2_scaled) = m2.checked_mul(factor) {
                if let Some(diff_val) = m1.checked_sub(m2_scaled) {
                    return Ok(normalize_decimal(diff_val, s1));
                }
            }
        }
    } else {
        let diff = (s2 - s1) as usize;
        if diff < POW10_I128.len() {
            let factor = POW10_I128[diff];
            if let Some(m1_scaled) = m1.checked_mul(factor) {
                if let Some(diff_val) = m1_scaled.checked_sub(m2) {
                    return Ok(normalize_decimal(diff_val, s2));
                }
            }
        }
    }
    Err("Decimal difference exceeds 38-digit exact banking limit".to_string())
}

/// Exact base-10 multiplication: multiplies integer mantissas and sums scales
pub fn decimal_mul(m1: i128, s1: u32, m2: i128, s2: u32) -> Result<(i128, u32), String> {
    if let Some(prod) = m1.checked_mul(m2) {
        Ok(normalize_decimal(prod, s1 + s2))
    } else {
        Err("Decimal product exceeds 38-digit exact banking limit".to_string())
    }
}

/// Exact/Banking base-10 division: detects terminating base-10 fractions (powers of 2 and 5),
/// otherwise divides to 28 decimal places with Banker's rounding.
pub fn decimal_div(m1: i128, s1: u32, m2: i128, s2: u32) -> Result<(i128, u32), String> {
    if m2 == 0 {
        return Err("Division by zero".to_string());
    }
    if m1 == 0 {
        return Ok((0, 0));
    }

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

    let g = gcd_i128(m1, m2);
    let mut num = m1 / g;
    let mut den = m2 / g;
    if den < 0 {
        num = -num;
        den = -den;
    }

    // Check if denominator terminates in base 10 (prime factors are strictly 2 and 5)
    let mut temp = den;
    let mut twos = 0usize;
    let mut fives = 0usize;
    while temp % 2 == 0 {
        temp /= 2;
        twos += 1;
    }
    while temp % 5 == 0 {
        temp /= 5;
        fives += 1;
    }

    if temp == 1 {
        // Terminating exact decimal!
        let extra_scale = twos.max(fives);
        if extra_scale < POW10_I128.len() {
            let factor = POW10_I128[extra_scale] / den;
            if let Some(scaled_num) = num.checked_mul(factor) {
                let total_scale = (s1 as i64) - (s2 as i64) + (extra_scale as i64);
                if total_scale >= 0 {
                    return Ok(normalize_decimal(scaled_num, total_scale as u32));
                } else {
                    let shift = (-total_scale) as usize;
                    if shift < POW10_I128.len() {
                        if let Some(expanded) = scaled_num.checked_mul(POW10_I128[shift]) {
                            return Ok((expanded, 0));
                        }
                    }
                }
            }
        }
    }

    // Non-terminating recurring fraction (e.g. 1/3, 1/7):
    // Divide with 28 guard digits and Banker's rounding (round half to even)
    let prec = 28usize;
    let scale_factor = POW10_I128[prec];
    if let Some(scaled_num) = num.checked_mul(scale_factor) {
        let q = scaled_num / den;
        let r = scaled_num % den;
        let rounded = if (r.abs() * 2) > den.abs() || ((r.abs() * 2) == den.abs() && (q % 2 != 0)) {
            if q >= 0 { q + 1 } else { q - 1 }
        } else {
            q
        };
        let total_scale = (s1 as i64) - (s2 as i64) + (prec as i64);
        if total_scale >= 0 {
            return Ok(normalize_decimal(rounded, total_scale as u32));
        }
    }

    Err("Decimal division overflowed precision capacity".to_string())
}

/// Exact integer power of decimal
pub fn decimal_pow(mut m: i128, s: u32, mut exp: u32) -> Result<(i128, u32), String> {
    if exp == 0 {
        return Ok((1, 0));
    }
    let total_scale = match s.checked_mul(exp) {
        Some(ts) => ts,
        None => return Err("Exponent scale overflow".to_string()),
    };

    let mut res_m = 1i128;
    while exp > 0 {
        if exp & 1 == 1 {
            res_m = match res_m.checked_mul(m) {
                Some(p) => p,
                None => return Err("Decimal power exceeds 38-digit exact banking limit".to_string()),
            };
        }
        exp >>= 1;
        if exp > 0 {
            m = match m.checked_mul(m) {
                Some(sq) => sq,
                None => return Err("Decimal power exceeds 38-digit exact banking limit".to_string()),
            };
        }
    }
    Ok(normalize_decimal(res_m, total_scale))
}

/// Formats an exact decimal as a clean string without ANY float conversion
pub fn format_exact_decimal(mantissa: i128, scale: u32) -> String {
    let sign = if mantissa < 0 { "-" } else { "" };
    let abs_m = mantissa.unsigned_abs();
    let s = abs_m.to_string();
    let scale = scale as usize;

    if scale == 0 {
        format!("{}{}", sign, s)
    } else if s.len() <= scale {
        let zeros = "0".repeat(scale - s.len());
        format!("{}0.{}{}", sign, zeros, s)
    } else {
        let split_idx = s.len() - scale;
        let int_part = &s[..split_idx];
        let frac_part = &s[split_idx..];
        format!("{}{}.{}", sign, int_part, frac_part)
    }
}
