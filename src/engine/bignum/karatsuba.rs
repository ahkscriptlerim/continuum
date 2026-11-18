use super::comba::{comba_add, comba_mul, comba_sub_assign};
use crate::engine::evaluator::{current_task_cancel_ctx, is_task_aborted, EpochGuard};

/// Karatsuba Multiplication: O(N^1.585) divide-and-conquer
/// Uses an L1-cache-resident stack scratch arena for N <= 256 limbs (zero heap allocations)
/// and a single flat scratch slice for larger limbs.
pub fn karatsuba_mul(a: &[u64], b: &[u64], out: &mut [u64], cutoff_limbs: usize) {
    let n = a.len().max(b.len());
    let cutoff = cutoff_limbs.max(8);
    if n <= cutoff || a.len() <= 8 || b.len() <= 8 {
        comba_mul(a, b, out);
        return;
    }

    let scratch_needed = 16 * n + 512;
    if scratch_needed <= 2304 {
        let mut stack_scratch = [0u64; 2304];
        karatsuba_mul_scratch(a, b, out, &mut stack_scratch[..scratch_needed], cutoff);
    } else {
        let mut heap_scratch = vec![0u64; scratch_needed];
        karatsuba_mul_scratch(a, b, out, &mut heap_scratch, cutoff);
    }
}

/// Multi-core work-stealing Parallel Karatsuba Multiplication via Rayon.
/// Splits the 3 independent sub-products (z0, z2, z1) across hardware threads for N >= 128 limbs,
/// propagating `TaskCancelCtx` so cooperative preemption aborts all worker threads in microseconds.
pub fn par_karatsuba_mul(a: &[u64], b: &[u64], out: &mut [u64], cutoff_limbs: usize) {
    if is_task_aborted() {
        out.fill(0);
        return;
    }
    let n = a.len().max(b.len());
    let cutoff = cutoff_limbs.max(8);
    if n < 128 || a.len() < 64 || b.len() < 64 {
        karatsuba_mul(a, b, out, cutoff);
        return;
    }

    let m = n / 2;
    let rem = n - m;
    let max_half = m.max(rem) + 1;

    let a0 = if a.len() > m { &a[..m] } else { a };
    let a1 = if a.len() > m { &a[m..] } else { &[] };
    let b0 = if b.len() > m { &b[..m] } else { b };
    let b1 = if b.len() > m { &b[m..] } else { &[] };

    let mut a_sum = vec![0u64; max_half];
    let mut b_sum = vec![0u64; max_half];
    let c_a = comba_add(a0, a1, &mut a_sum);
    let idx_a = a0.len().max(a1.len());
    if idx_a < a_sum.len() {
        a_sum[idx_a] = c_a;
    }
    let c_b = comba_add(b0, b1, &mut b_sum);
    let idx_b = b0.len().max(b1.len());
    if idx_b < b_sum.len() {
        b_sum[idx_b] = c_b;
    }

    let mut z0 = vec![0u64; m * 2];
    let mut z2 = vec![0u64; rem * 2 + 2];
    let mut z1 = vec![0u64; max_half * 2];

    let ctx = current_task_cancel_ctx();
    rayon::join(
        || {
            let _g = EpochGuard::from_ctx(ctx);
            par_karatsuba_mul(a0, b0, &mut z0, cutoff);
        },
        || {
            let _g = EpochGuard::from_ctx(ctx);
            rayon::join(
                || {
                    let _g2 = EpochGuard::from_ctx(ctx);
                    if !a1.is_empty() && !b1.is_empty() {
                        par_karatsuba_mul(a1, b1, &mut z2, cutoff);
                    }
                },
                || {
                    let _g3 = EpochGuard::from_ctx(ctx);
                    par_karatsuba_mul(&a_sum, &b_sum, &mut z1, cutoff);
                },
            );
        },
    );

    if is_task_aborted() {
        out.fill(0);
        return;
    }

    comba_sub_assign(&mut z1, &z0);
    comba_sub_assign(&mut z1, &z2);

    out.fill(0);
    add_shifted(out, &z0, 0);
    add_shifted(out, &z1, m);
    if !a1.is_empty() && !b1.is_empty() {
        add_shifted(out, &z2, 2 * m);
    }
}

fn karatsuba_mul_scratch(
    a: &[u64],
    b: &[u64],
    out: &mut [u64],
    scratch: &mut [u64],
    cutoff: usize,
) {
    if is_task_aborted() {
        return;
    }

    let n = a.len().max(b.len());
    if n <= cutoff || a.len() <= 8 || b.len() <= 8 {
        comba_mul(a, b, out);
        return;
    }

    let m = n / 2;
    let rem = n - m;
    let max_half = m.max(rem) + 1;

    let z0_len = m * 2;
    let z2_len = rem * 2 + 2;
    let z1_len = max_half * 2;
    let local_total = z0_len + z2_len + max_half * 2 + z1_len;

    if scratch.len() < local_total {
        comba_mul(a, b, out);
        return;
    }

    let a0 = if a.len() > m { &a[..m] } else { a };
    let a1 = if a.len() > m { &a[m..] } else { &[] };
    let b0 = if b.len() > m { &b[..m] } else { b };
    let b1 = if b.len() > m { &b[m..] } else { &[] };

    let (local, child_scratch) = scratch.split_at_mut(local_total);
    let (z0, rest) = local.split_at_mut(z0_len);
    let (z2, rest) = rest.split_at_mut(z2_len);
    let (a_sum, rest) = rest.split_at_mut(max_half);
    let (b_sum, z1) = rest.split_at_mut(max_half);

    z0.fill(0);
    z2.fill(0);
    a_sum.fill(0);
    b_sum.fill(0);
    z1.fill(0);

    karatsuba_mul_scratch(a0, b0, z0, child_scratch, cutoff);
    if is_task_aborted() {
        return;
    }
    if !a1.is_empty() && !b1.is_empty() {
        karatsuba_mul_scratch(a1, b1, z2, child_scratch, cutoff);
        if is_task_aborted() {
            return;
        }
    }

    // z1 = (a0 + a1)*(b0 + b1) - z0 - z2
    let c_a = comba_add(a0, a1, a_sum);
    let idx_a = a0.len().max(a1.len());
    if idx_a < a_sum.len() {
        a_sum[idx_a] = c_a;
    }
    let c_b = comba_add(b0, b1, b_sum);
    let idx_b = b0.len().max(b1.len());
    if idx_b < b_sum.len() {
        b_sum[idx_b] = c_b;
    }

    karatsuba_mul_scratch(a_sum, b_sum, z1, child_scratch, cutoff);
    if is_task_aborted() {
        return;
    }

    // Subtract z0 and z2 from z1 in-place (zero clone allocations!)
    comba_sub_assign(z1, z0);
    comba_sub_assign(z1, z2);

    out.fill(0);
    // Combine: out = z0 + (z1 << m) + (z2 << 2m)
    add_shifted(out, z0, 0);
    add_shifted(out, z1, m);
    if !a1.is_empty() && !b1.is_empty() {
        add_shifted(out, z2, 2 * m);
    }
}

#[inline(always)]
fn add_shifted(out: &mut [u64], slice: &[u64], shift: usize) {
    let mut carry: u128 = 0;
    for (i, &val) in slice.iter().enumerate() {
        let idx = i + shift;
        if idx >= out.len() {
            break;
        }
        let sum = (out[idx] as u128) + (val as u128) + carry;
        out[idx] = sum as u64;
        carry = sum >> 64;
    }
    let mut idx = slice.len() + shift;
    while carry > 0 && idx < out.len() {
        let sum = (out[idx] as u128) + carry;
        out[idx] = sum as u64;
        carry = sum >> 64;
        idx += 1;
    }
}
