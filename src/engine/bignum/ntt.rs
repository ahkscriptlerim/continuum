//! SOTA 2-Prime 64-Bit Number-Theoretic Transform (NTT) Bignum Multiplication & Squaring
//! - Operates on base-2^16 digits with primes P1 = 998244353 (g = 3) and P2 = 1004535809 (g = 3).
//! - Single-u64 Garner Chinese Remainder Theorem (CRT) reconstruction (zero u128 CRT division).
//! - Heterogeneous GPU (OpenCL) + Multi-Core CPU (Rayon) Co-Execution across the two independent primes.
//! - Cooperative inner-loop cancellation (`is_task_aborted()`) at every butterfly stage.

use crate::engine::evaluator::{current_task_cancel_ctx, is_task_aborted, EpochGuard};
use crate::engine::gpu::GpuComputeEngine;

pub const NTT_P1: u32 = 998_244_353; // 119 * 2^23 + 1
pub const NTT_P2: u32 = 1_004_535_809; // 479 * 2^21 + 1
pub const NTT_ROOT: u32 = 3;

const fn pow_mod_const(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut res = 1u64;
    base %= m;
    while exp > 0 {
        if (exp & 1) != 0 {
            res = (res * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    res
}

/// P1^-1 mod P2, computed at compile time for 0-cycle Garner CRT reconstruction
const INV_P1_MOD_P2: u64 = pow_mod_const(NTT_P1 as u64, (NTT_P2 - 2) as u64, NTT_P2 as u64);

#[inline(always)]
fn pow_mod_u32(base: u32, exp: u32, m: u32) -> u32 {
    pow_mod_const(base as u64, exp as u64, m as u64) as u32
}

/// 64-bit Barrett modular reduction (`0` hardware `DIV` stalls; compiles to `MULX` + `IMUL` + `CMOV` in ~4 cycles).
/// Valid for all `x < m64 * m64 < 2^62` where `mu = ((1u128 << 64) / (m64 as u128)) as u64`.
#[inline(always)]
fn barrett_rem_u64(x: u64, m64: u64, mu: u64) -> u32 {
    let q = (((x as u128) * (mu as u128)) >> 64) as u64;
    let mut r = x.wrapping_sub(q.wrapping_mul(m64));
    if r >= m64 {
        r -= m64;
    }
    if r >= m64 {
        r -= m64;
    }
    r as u32
}

/// In-place iterative Cooley-Tukey radix-2 NTT over Z_p
#[inline]
pub fn ntt_transform_cpu(a: &mut [u32], invert: bool, mod_p: u32, root: u32) {
    let n = a.len();
    if n <= 1 || !n.is_power_of_two() || mod_p <= 1 {
        return;
    }

    // 1. Bit-reversal permutation (`n` proven power-of-two => `j < n` invariant, `0` bounds panics)
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            unsafe {
                let pa = a.as_mut_ptr();
                core::ptr::swap(pa.add(i), pa.add(j));
            }
        }
    }

    // 2. Iterative butterfly stages with `16 KB` L1d stack twiddle arena (`n <= 8192`) + Barrett `MULX` reduction
    let m64 = mod_p as u64;
    let barrett_mu = ((1u128 << 64) / (m64 as u128)) as u64;
    let mut len = 2usize;
    let mut stack_twiddles = [0u32; 4096];
    let mut heap_twiddles: Vec<u32> = if n / 2 > 4096 {
        vec![0u32; n / 2]
    } else {
        Vec::new()
    };
    let twiddles: &mut [u32] = if n / 2 > 4096 {
        &mut heap_twiddles
    } else {
        &mut stack_twiddles[..n / 2]
    };

    while len <= n {
        if is_task_aborted() {
            return;
        }
        let half = len >> 1;
        let mut wlen = pow_mod_u32(root, (mod_p - 1) / (len as u32), mod_p);
        if invert {
            wlen = pow_mod_u32(wlen, mod_p - 2, mod_p);
        }

        // Precompute stage twiddle table once per stage (zero vec.push, zero DIV)
        let mut w = 1u64;
        let wlen64 = wlen as u64;
        for k in 0..half {
            unsafe {
                *twiddles.get_unchecked_mut(k) = w as u32;
            }
            w = barrett_rem_u64(w * wlen64, m64, barrett_mu) as u64;
        }

        for block in a.chunks_exact_mut(len) {
            let (lo, hi) = block.split_at_mut(half);
            for k in 0..half {
                unsafe {
                    let u = *lo.get_unchecked(k);
                    let v = barrett_rem_u64(
                        (*hi.get_unchecked(k) as u64) * (*twiddles.get_unchecked(k) as u64),
                        m64,
                        barrett_mu,
                    );
                    let mut sum = u + v;
                    if sum >= mod_p {
                        sum -= mod_p;
                    }
                    let diff = if u >= v { u - v } else { u + mod_p - v };
                    *lo.get_unchecked_mut(k) = sum;
                    *hi.get_unchecked_mut(k) = diff;
                }
            }
        }

        len <<= 1;
    }

    if invert {
        let inv_n = pow_mod_u32(n as u32, mod_p - 2, mod_p) as u64;
        for x in a.iter_mut() {
            *x = barrett_rem_u64((*x as u64) * inv_n, m64, barrett_mu);
        }
    }
}

fn convolve_mod_cpu(a: &[u32], b: Option<&[u32]>, mod_p: u32, root: u32) -> Vec<u32> {
    let n = a.len();
    let mut fa = a.to_vec();
    let m64 = mod_p as u64;
    let barrett_mu = ((1u128 << 64) / (m64.max(1) as u128)) as u64;

    if let Some(b_slice) = b {
        let mut fb = b_slice.to_vec();
        if n >= 4096 {
            let ctx = current_task_cancel_ctx();
            rayon::join(
                || {
                    let _g = EpochGuard::from_ctx(ctx);
                    ntt_transform_cpu(&mut fa, false, mod_p, root);
                },
                || {
                    let _g = EpochGuard::from_ctx(ctx);
                    ntt_transform_cpu(&mut fb, false, mod_p, root);
                },
            );
        } else {
            ntt_transform_cpu(&mut fa, false, mod_p, root);
            ntt_transform_cpu(&mut fb, false, mod_p, root);
        }
        if is_task_aborted() {
            return Vec::new();
        }
        for (x, &y) in fa.iter_mut().zip(fb.iter()) {
            *x = barrett_rem_u64((*x as u64) * (y as u64), m64, barrett_mu);
        }
    } else {
        ntt_transform_cpu(&mut fa, false, mod_p, root);
        if is_task_aborted() {
            return Vec::new();
        }
        for x in fa.iter_mut() {
            let v = *x as u64;
            *x = barrett_rem_u64(v * v, m64, barrett_mu);
        }
    }

    ntt_transform_cpu(&mut fa, true, mod_p, root);
    fa
}

#[inline(always)]
fn unpack_u64_to_u16_words(limbs: &[u64], total_words: usize) -> Vec<u32> {
    let mut out = vec![0u32; total_words];
    for (chunk, &limb) in out.chunks_exact_mut(4).zip(limbs.iter()) {
        chunk[0] = (limb & 0xFFFF) as u32;
        chunk[1] = ((limb >> 16) & 0xFFFF) as u32;
        chunk[2] = ((limb >> 32) & 0xFFFF) as u32;
        chunk[3] = (limb >> 48) as u32;
    }
    out
}

#[inline(always)]
fn crt_reconstruct_and_pack(c1: &[u32], c2: &[u32], out: &mut [u64]) {
    out.fill(0);
    let out_len = out.len();
    let max_words = (out_len * 4).min(c1.len()).min(c2.len());
    let mut carry: u64 = 0;
    let p1 = NTT_P1 as u64;
    let p2 = NTT_P2 as u64;
    let mu_p2 = ((1u128 << 64) / (p2 as u128)) as u64;

    let mut limb_acc: u64 = 0;
    for i in 0..max_words {
        if (i & 4095) == 4095 && is_task_aborted() {
            out.fill(0);
            return;
        }
        let (r1, r2) = unsafe { (*c1.get_unchecked(i) as u64, *c2.get_unchecked(i) as u64) };
        let diff = if r2 >= r1 { r2 - r1 } else { r2 + p2 - r1 };
        let k = barrett_rem_u64(diff * INV_P1_MOD_P2, p2, mu_p2) as u64;
        let exact_conv = r1 + k * p1;

        let sum = exact_conv + carry;
        let digit16 = sum & 0xFFFF;
        carry = sum >> 16;

        let sub = i & 3;
        limb_acc |= digit16 << (sub * 16);
        if sub == 3 {
            unsafe {
                *out.get_unchecked_mut(i >> 2) = limb_acc;
            }
            limb_acc = 0;
        }
    }

    let mut word_idx = max_words;
    while (carry > 0 || (word_idx & 3) != 0) && (word_idx >> 2) < out_len {
        let digit16 = carry & 0xFFFF;
        carry >>= 16;
        let sub = word_idx & 3;
        limb_acc |= digit16 << (sub * 16);
        if sub == 3 {
            unsafe {
                *out.get_unchecked_mut(word_idx >> 2) = limb_acc;
            }
            limb_acc = 0;
        }
        word_idx += 1;
    }
}

/// Multiplies arbitrary-length u64 limb slices using 2-Prime NTT with Heterogeneous GPU + CPU Co-Execution.
#[inline]
pub fn ntt_mul(a: &[u64], b: &[u64], out: &mut [u64]) {
    if is_task_aborted() {
        out.fill(0);
        return;
    }
    // Trim leading zero limbs
    let mut len_a = a.len();
    while len_a > 0 && a[len_a - 1] == 0 {
        len_a -= 1;
    }
    let mut len_b = b.len();
    while len_b > 0 && b[len_b - 1] == 0 {
        len_b -= 1;
    }
    if len_a == 0 || len_b == 0 {
        out.fill(0);
        return;
    }

    let words_needed = (len_a + len_b) * 4;
    let n = words_needed.next_power_of_two().max(8);

    // Max transform length supported by NTT_P2 (479 * 2^21 + 1) is 2^21 = 2,097,152 base-2^16 words
    // (= 524,288 limbs = 33,554,432 bits).
    if n > (1 << 21) {
        super::karatsuba::par_karatsuba_mul(&a[..len_a], &b[..len_b], out, 24);
        return;
    }

    let wa = unpack_u64_to_u16_words(&a[..len_a], n);
    let wb = unpack_u64_to_u16_words(&b[..len_b], n);

    let ctx = current_task_cancel_ctx();
    let use_gpu = n >= 4_096 && GpuComputeEngine::global().available;

    let (c1, c2) = rayon::join(
        || {
            let _g = EpochGuard::from_ctx(ctx);
            if use_gpu {
                let mut gpu_out = vec![0u32; n];
                if GpuComputeEngine::global().try_convolve_mod(
                    &wa,
                    Some(&wb),
                    &mut gpu_out,
                    NTT_P1,
                    NTT_ROOT,
                ) {
                    return gpu_out;
                }
            }
            convolve_mod_cpu(&wa, Some(&wb), NTT_P1, NTT_ROOT)
        },
        || {
            let _g = EpochGuard::from_ctx(ctx);
            convolve_mod_cpu(&wa, Some(&wb), NTT_P2, NTT_ROOT)
        },
    );

    if is_task_aborted() || c1.is_empty() || c2.is_empty() {
        out.fill(0);
        return;
    }

    crt_reconstruct_and_pack(&c1, &c2, out);
}

/// Squares an arbitrary-length u64 limb slice (`out = a * a`) using 2-Prime NTT.
/// Requires only 2 NTT transforms per prime instead of 3 (33% faster than general multiplication).
pub fn ntt_sqr(a: &[u64], out: &mut [u64]) {
    if is_task_aborted() {
        out.fill(0);
        return;
    }
    let mut len_a = a.len();
    while len_a > 0 && a[len_a - 1] == 0 {
        len_a -= 1;
    }
    if len_a == 0 {
        out.fill(0);
        return;
    }

    let words_needed = (len_a * 2) * 4;
    let n = words_needed.next_power_of_two().max(8);

    if n > (1 << 21) {
        super::karatsuba::par_karatsuba_mul(&a[..len_a], &a[..len_a], out, 24);
        return;
    }

    let wa = unpack_u64_to_u16_words(&a[..len_a], n);
    let ctx = current_task_cancel_ctx();
    let use_gpu = n >= 4_096 && GpuComputeEngine::global().available;

    let (c1, c2) = rayon::join(
        || {
            let _g = EpochGuard::from_ctx(ctx);
            if use_gpu {
                let mut gpu_out = vec![0u32; n];
                if GpuComputeEngine::global().try_convolve_mod(
                    &wa,
                    None,
                    &mut gpu_out,
                    NTT_P1,
                    NTT_ROOT,
                ) {
                    return gpu_out;
                }
            }
            convolve_mod_cpu(&wa, None, NTT_P1, NTT_ROOT)
        },
        || {
            let _g = EpochGuard::from_ctx(ctx);
            convolve_mod_cpu(&wa, None, NTT_P2, NTT_ROOT)
        },
    );

    if is_task_aborted() || c1.is_empty() || c2.is_empty() {
        out.fill(0);
        return;
    }

    crt_reconstruct_and_pack(&c1, &c2, out);
}
