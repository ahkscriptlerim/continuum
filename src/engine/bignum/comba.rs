/// Hardware-Accelerated Multi-Precision Limb Arithmetic (x86_64 ADC/SBB/MULX + Scalar Fallback)
/// Operates on contiguous slices of u64 limbs with zero heap allocations.

#[inline(always)]
pub fn comba_add(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    let n = a.len().min(b.len()).min(out.len());
    let mut carry = 0u8;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_addcarry_u64;
        for i in 0..n {
            carry = _addcarry_u64(carry, *a.get_unchecked(i), *b.get_unchecked(i), out.get_unchecked_mut(i));
        }
        if a.len() > n && out.len() > n {
            let limit = a.len().min(out.len());
            for i in n..limit {
                carry = _addcarry_u64(carry, *a.get_unchecked(i), 0, out.get_unchecked_mut(i));
            }
        } else if b.len() > n && out.len() > n {
            let limit = b.len().min(out.len());
            for i in n..limit {
                carry = _addcarry_u64(carry, *b.get_unchecked(i), 0, out.get_unchecked_mut(i));
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..n {
            let (s1, c1) = a[i].overflowing_add(b[i]);
            let (s2, c2) = s1.overflowing_add(carry as u64);
            out[i] = s2;
            carry = (c1 as u8) + (c2 as u8);
        }
        if a.len() > n && out.len() > n {
            for i in n..a.len().min(out.len()) {
                let (s, c) = a[i].overflowing_add(carry as u64);
                out[i] = s;
                carry = c as u8;
            }
        } else if b.len() > n && out.len() > n {
            for i in n..b.len().min(out.len()) {
                let (s, c) = b[i].overflowing_add(carry as u64);
                out[i] = s;
                carry = c as u8;
            }
        }
    }

    carry as u64
}

#[inline]
pub fn comba_sub(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    let n = a.len().min(b.len()).min(out.len());
    let mut borrow = 0u8;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_subborrow_u64;
        for i in 0..n {
            borrow = _subborrow_u64(borrow, *a.get_unchecked(i), *b.get_unchecked(i), out.get_unchecked_mut(i));
        }
        if a.len() > n && out.len() > n {
            let limit = a.len().min(out.len());
            for i in n..limit {
                borrow = _subborrow_u64(borrow, *a.get_unchecked(i), 0, out.get_unchecked_mut(i));
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..n {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow as u64);
            out[i] = d2;
            borrow = (b1 as u8) + (b2 as u8);
        }
        if a.len() > n && out.len() > n {
            for i in n..a.len().min(out.len()) {
                let (d, b_flag) = a[i].overflowing_sub(borrow as u64);
                out[i] = d;
                borrow = b_flag as u8;
            }
        }
    }

    borrow as u64
}

/// In-place limb subtraction: `lhs -= rhs`, eliminating clone allocations in Karatsuba
#[inline(always)]
pub fn comba_sub_assign(lhs: &mut [u64], rhs: &[u64]) -> u64 {
    let n = lhs.len().min(rhs.len());
    let mut borrow = 0u8;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_subborrow_u64;
        for i in 0..n {
            let a_val = *lhs.get_unchecked(i);
            let b_val = *rhs.get_unchecked(i);
            borrow = _subborrow_u64(borrow, a_val, b_val, lhs.get_unchecked_mut(i));
        }
        for i in n..lhs.len() {
            if borrow == 0 {
                break;
            }
            let a_val = *lhs.get_unchecked(i);
            borrow = _subborrow_u64(borrow, a_val, 0, lhs.get_unchecked_mut(i));
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..n {
            let (d1, b1) = lhs[i].overflowing_sub(rhs[i]);
            let (d2, b2) = d1.overflowing_sub(borrow as u64);
            lhs[i] = d2;
            borrow = (b1 as u8) + (b2 as u8);
        }
        for i in n..lhs.len() {
            if borrow == 0 {
                break;
            }
            let (d, b_flag) = lhs[i].overflowing_sub(borrow as u64);
            lhs[i] = d;
            borrow = b_flag as u8;
        }
    }

    borrow as u64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,adx")]
unsafe fn comba_mul_bmi2(a: &[u64], b: &[u64], out: &mut [u64]) {
    use core::arch::x86_64::{_addcarry_u64, _mulx_u64};
    out.fill(0);
    let n = a.len();
    let m = b.len();
    let out_len = out.len();

    for i in 0..n {
        if (i & 31) == 31 && crate::engine::evaluator::is_task_aborted() {
            return;
        }
        let a_i = unsafe { *a.get_unchecked(i) };
        if a_i == 0 {
            continue;
        }
        let mut carry = 0u64;
        let j_limit = m.min(out_len.saturating_sub(i));
        for j in 0..j_limit {
            let mut hi = 0u64;
            let lo = unsafe { _mulx_u64(a_i, *b.get_unchecked(j), &mut hi) };
            let out_ptr = unsafe { out.get_unchecked_mut(i + j) };
            let mut sum1 = 0u64;
            let c1 = _addcarry_u64(0, *out_ptr, lo, &mut sum1);
            let mut sum2 = 0u64;
            let c2 = _addcarry_u64(0, sum1, carry, &mut sum2);
            *out_ptr = sum2;
            carry = hi + (c1 as u64) + (c2 as u64);
        }
        if i + j_limit < out_len {
            unsafe {
                *out.get_unchecked_mut(i + j_limit) = carry;
            }
        }
    }
}

/// Computes `out = a * b` using hardware-accelerated row-wise MAC with BMI2/ADX dispatch
#[inline(always)]
pub fn comba_mul(a: &[u64], b: &[u64], out: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("adx") {
            unsafe {
                comba_mul_bmi2(a, b, out);
            }
            return;
        }
    }

    out.fill(0);
    let n = a.len();
    let m = b.len();
    let out_len = out.len();

    for i in 0..n {
        if (i & 31) == 31 && crate::engine::evaluator::is_task_aborted() {
            return;
        }
        let a_i = unsafe { *a.get_unchecked(i) } as u128;
        if a_i == 0 {
            continue;
        }
        let mut carry = 0u64;
        let j_limit = m.min(out_len.saturating_sub(i));
        for j in 0..j_limit {
            unsafe {
                let prod = a_i * (*b.get_unchecked(j) as u128)
                    + (*out.get_unchecked(i + j) as u128)
                    + (carry as u128);
                *out.get_unchecked_mut(i + j) = prod as u64;
                carry = (prod >> 64) as u64;
            }
        }
        if i + j_limit < out_len {
            unsafe {
                *out.get_unchecked_mut(i + j_limit) = carry;
            }
        }
    }
}
