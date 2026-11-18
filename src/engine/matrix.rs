use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fma_row_axpy(dst: &mut [f64], src: &[f64], alpha: f64) {
    use core::arch::x86_64::{
        _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd,
    };
    let n = dst.len().min(src.len());
    let alpha_v = _mm256_broadcast_sd(&alpha);
    let mut j = 0;
    while j + 4 <= n {
        unsafe {
            let d_ptr = dst.as_mut_ptr().add(j);
            let s_ptr = src.as_ptr().add(j);
            let d_v = _mm256_loadu_pd(d_ptr);
            let s_v = _mm256_loadu_pd(s_ptr);
            let res = _mm256_fmadd_pd(alpha_v, s_v, d_v);
            _mm256_storeu_pd(d_ptr, res);
        }
        j += 4;
    }
    while j < n {
        unsafe {
            *dst.get_unchecked_mut(j) += alpha * (*src.get_unchecked(j));
        }
        j += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fma_row_axpy(dst: &mut [f64], src: &[f64], alpha: f64) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};
    let n = dst.len().min(src.len());
    let alpha_v = vdupq_n_f64(alpha);
    let mut j = 0usize;
    while j + 4 <= n {
        unsafe {
            let d0 = dst.as_mut_ptr().add(j);
            let d1 = dst.as_mut_ptr().add(j + 2);
            let s0 = src.as_ptr().add(j);
            let s1 = src.as_ptr().add(j + 2);
            let r0 = vfmaq_f64(vld1q_f64(d0), vld1q_f64(s0), alpha_v);
            let r1 = vfmaq_f64(vld1q_f64(d1), vld1q_f64(s1), alpha_v);
            vst1q_f64(d0, r0);
            vst1q_f64(d1, r1);
        }
        j += 4;
    }
    while j < n {
        unsafe {
            *dst.get_unchecked_mut(j) += alpha * (*src.get_unchecked(j));
        }
        j += 1;
    }
}

#[inline(always)]
fn row_axpy(dst: &mut [f64], src: &[f64], alpha: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                avx2_fma_row_axpy(dst, src, alpha);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_fma_row_axpy(dst, src, alpha);
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        let len = mat.data.len();
        for i in 0..n {
            let idx = i * n + i;
            if idx < len {
                unsafe {
                    *mat.data.get_unchecked_mut(idx) = 1.0;
                }
            }
        }
        mat
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let idx = r * self.cols + c;
        if idx < self.data.len() {
            unsafe { *self.data.get_unchecked(idx) }
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        let idx = r * self.cols + c;
        if idx < self.data.len() {
            unsafe {
                *self.data.get_unchecked_mut(idx) = val;
            }
        }
    }

    pub fn transpose(&self) -> Matrix {
        let rows = self.rows;
        let cols = self.cols;
        let mut out = Matrix::zeros(cols, rows);
        if self.data.len() < rows * cols || out.data.len() < rows * cols {
            return out;
        }
        for r in 0..rows {
            let src_base = r * cols;
            for c in 0..cols {
                unsafe {
                    *out.data.get_unchecked_mut(c * rows + r) = *self.data.get_unchecked(src_base + c);
                }
            }
        }
        out
    }

    /// Cache-blocked AVX2+FMA GEMM matrix multiplication (O(N³)) with Rayon multi-core scaling
    pub fn mul(&self, rhs: &Matrix) -> Result<Matrix, String> {
        if self.cols != rhs.rows {
            return Err(format!(
                "Dimension mismatch: cannot multiply {}x{} by {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            ));
        }

        let m = self.rows;
        let k_dim = self.cols;
        let n = rhs.cols;
        let mut out = Matrix::zeros(m, n);
        if m == 0 || k_dim == 0 || n == 0 || self.data.len() < m * k_dim || rhs.data.len() < k_dim * n {
            return Ok(out);
        }

        // Multi-core Rayon parallel dispatch scaled to probed hardware logical_cores & SIMD width
        if m * k_dim * n >= crate::engine::wisdom::parallel_matrix_flops_cutoff() {
            let cancel_ctx = crate::engine::evaluator::current_task_cancel_ctx();
            out.data
                .par_chunks_exact_mut(n)
                .enumerate()
                .for_each(|(i, out_row)| {
                    if cancel_ctx.is_aborted() {
                        return;
                    }
                    let a_base = i * k_dim;
                    for k in 0..k_dim {
                        let a_ik = unsafe { *self.data.get_unchecked(a_base + k) };
                        if a_ik != 0.0 {
                            let b_row = unsafe { rhs.data.get_unchecked(k * n..(k + 1) * n) };
                            row_axpy(out_row, b_row, a_ik);
                        }
                    }
                });
            crate::engine::evaluator::check_task_aborted()?;
            return Ok(out);
        }

        // Probed L1D-resident (`32 KB` -> 32, `48-64 KB` -> 40, `96+ KB` -> 64) SIMD-blocked micro-kernel
        let block_size = crate::engine::wisdom::l1_matrix_block_dim().max(16);

        for ii in (0..m).step_by(block_size) {
            crate::engine::evaluator::check_task_aborted()?;
            let i_max = (ii + block_size).min(m);
            for kk in (0..k_dim).step_by(block_size) {
                let k_max = (kk + block_size).min(k_dim);
                for jj in (0..n).step_by(block_size) {
                    let j_max = (jj + block_size).min(n);

                    for i in ii..i_max {
                        let out_slice = unsafe { out.data.get_unchecked_mut(i * n + jj..i * n + j_max) };
                        let a_base = i * k_dim;
                        for k in kk..k_max {
                            let a_ik = unsafe { *self.data.get_unchecked(a_base + k) };
                            if a_ik != 0.0 {
                                let b_slice = unsafe { rhs.data.get_unchecked(k * n + jj..k * n + j_max) };
                                row_axpy(out_slice, b_slice, a_ik);
                            }
                        }
                    }
                }
            }
        }

        Ok(out)
    }

    pub fn determinant(&self) -> Result<f64, String> {
        if self.rows != self.cols {
            return Err("Determinant requires a square matrix".to_string());
        }
        let n = self.rows;
        if n == 0 || self.data.len() < n * n {
            return Ok(0.0);
        }
        if n == 1 {
            return Ok(unsafe { *self.data.get_unchecked(0) });
        }
        if n == 2 {
            return Ok(self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0));
        }
        if n == 3 {
            let d = self.get(0, 0) * (self.get(1, 1) * self.get(2, 2) - self.get(1, 2) * self.get(2, 1))
                  - self.get(0, 1) * (self.get(1, 0) * self.get(2, 2) - self.get(1, 2) * self.get(2, 0))
                  + self.get(0, 2) * (self.get(1, 0) * self.get(2, 1) - self.get(1, 1) * self.get(2, 0));
            return Ok(d);
        }

        // LU decomposition with AVX2 row elimination & hoisted reciprocal for N > 3
        let mut a = self.data.clone();
        let mut det = 1.0;
        for i in 0..n {
            if (i & 15) == 15 {
                crate::engine::evaluator::check_task_aborted()?;
            }
            let mut pivot = i;
            let mut max_abs = unsafe { (*a.get_unchecked(i * n + i)).abs() };
            for j in (i + 1)..n {
                let v = unsafe { (*a.get_unchecked(j * n + i)).abs() };
                if v > max_abs {
                    max_abs = v;
                    pivot = j;
                }
            }
            if max_abs < 1e-12 {
                return Ok(0.0);
            }
            if pivot != i {
                let pa = a.as_mut_ptr();
                for k in 0..n {
                    unsafe {
                        core::ptr::swap(pa.add(i * n + k), pa.add(pivot * n + k));
                    }
                }
                det = -det;
            }
            let diag = unsafe { *a.get_unchecked(i * n + i) };
            det *= diag;
            let inv_diag = 1.0 / diag;
            let (top, bottom) = unsafe { a.get_unchecked_mut(..n * n).split_at_mut((i + 1) * n) };
            let pivot_tail = unsafe { top.get_unchecked(i * n + i + 1..(i + 1) * n) };
            for row_j in bottom.chunks_exact_mut(n) {
                let factor = unsafe { *row_j.get_unchecked(i) } * inv_diag;
                if factor.abs() > 1e-15 {
                    let dst_tail = unsafe { row_j.get_unchecked_mut(i + 1..n) };
                    row_axpy(dst_tail, pivot_tail, -factor);
                }
            }
        }
        Ok(det)
    }

    pub fn add(&self, rhs: &Matrix) -> Result<Matrix, String> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(format!(
                "Dimension mismatch for addition: {}x{} vs {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            ));
        }
        let mut out = self.clone();
        row_axpy(&mut out.data, &rhs.data, 1.0);
        Ok(out)
    }

    pub fn sub(&self, rhs: &Matrix) -> Result<Matrix, String> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(format!(
                "Dimension mismatch for subtraction: {}x{} vs {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            ));
        }
        let mut out = self.clone();
        row_axpy(&mut out.data, &rhs.data, -1.0);
        Ok(out)
    }

    pub fn scale(&self, factor: f64) -> Matrix {
        let mut out = Matrix::zeros(self.rows, self.cols);
        row_axpy(&mut out.data, &self.data, factor);
        out
    }

    pub fn trace(&self) -> Result<f64, String> {
        if self.rows != self.cols {
            return Err("Trace requires a square matrix".to_string());
        }
        let n = self.rows;
        if self.data.len() < n * n {
            return Ok(0.0);
        }
        let mut tr = 0.0;
        for i in 0..n {
            tr += unsafe { *self.data.get_unchecked(i * n + i) };
        }
        Ok(tr)
    }

    pub fn frobenius_norm(&self) -> f64 {
        let mut acc0 = 0.0f64;
        let mut acc1 = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut acc3 = 0.0f64;
        let chunks = self.data.chunks_exact(4);
        let rem = chunks.remainder();
        for c in chunks {
            acc0 += c[0] * c[0];
            acc1 += c[1] * c[1];
            acc2 += c[2] * c[2];
            acc3 += c[3] * c[3];
        }
        let mut tail = 0.0f64;
        for &x in rem {
            tail += x * x;
        }
        ((acc0 + acc1) + (acc2 + acc3) + tail).sqrt()
    }

    /// Computes numerical rank via Gaussian row echelon reduction with partial pivoting + AVX2 `row_axpy`
    pub fn rank(&self) -> usize {
        let m = self.rows;
        let n = self.cols;
        if m == 0 || n == 0 || self.data.len() < m * n {
            return 0;
        }
        let mut a = self.data.clone();
        let mut r = 0usize;

        for col in 0..n {
            if r >= m || crate::engine::evaluator::is_task_aborted() {
                break;
            }
            let mut pivot = r;
            let mut max_val = unsafe { (*a.get_unchecked(r * n + col)).abs() };
            for row in (r + 1)..m {
                let v = unsafe { (*a.get_unchecked(row * n + col)).abs() };
                if v > max_val {
                    max_val = v;
                    pivot = row;
                }
            }
            if max_val < 1e-10 {
                continue;
            }
            if pivot != r {
                let pa = a.as_mut_ptr();
                for k in col..n {
                    unsafe {
                        core::ptr::swap(pa.add(r * n + k), pa.add(pivot * n + k));
                    }
                }
            }
            let diag = unsafe { *a.get_unchecked(r * n + col) };
            let inv_diag = 1.0 / diag;
            let (top, bottom) = unsafe { a.get_unchecked_mut(..m * n).split_at_mut((r + 1) * n) };
            let pivot_tail = unsafe { top.get_unchecked(r * n + col..(r + 1) * n) };
            for row_slice in bottom.chunks_exact_mut(n) {
                let factor = unsafe { *row_slice.get_unchecked(col) } * inv_diag;
                if factor.abs() > 1e-14 {
                    let dst_tail = unsafe { row_slice.get_unchecked_mut(col..n) };
                    row_axpy(dst_tail, pivot_tail, -factor);
                }
            }
            r += 1;
        }
        r
    }

    /// Solves linear system `A * X = B` using Gauss-Jordan elimination with partial pivoting + AVX2 `row_axpy`
    pub fn solve(&self, rhs: &Matrix) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("Linear solver requires a square coefficient matrix A".to_string());
        }
        let n = self.rows;
        let rhs_cols = if rhs.rows == n {
            rhs.cols
        } else if rhs.rows == 1 && rhs.cols == n {
            // Automatically treat 1xN row vector as Nx1 column vector
            1
        } else {
            return Err(format!(
                "Dimension mismatch in solve(A, b): A is {}x{}, b is {}x{}",
                n, n, rhs.rows, rhs.cols
            ));
        };
        if n == 0 || rhs_cols == 0 || self.data.len() < n * n || rhs.data.len() < n * rhs_cols {
            return Ok(Matrix::zeros(n, rhs_cols));
        }

        let mut a = self.data.clone();
        let mut x = rhs.data.clone();

        for i in 0..n {
            if (i & 15) == 15 {
                crate::engine::evaluator::check_task_aborted()?;
            }
            let mut pivot = i;
            let mut max_val = unsafe { (*a.get_unchecked(i * n + i)).abs() };
            for j in (i + 1)..n {
                let val = unsafe { (*a.get_unchecked(j * n + i)).abs() };
                if val > max_val {
                    max_val = val;
                    pivot = j;
                }
            }
            if max_val < 1e-12 {
                return Err("Matrix A is singular (system has no unique solution)".to_string());
            }
            if pivot != i {
                let pa = a.as_mut_ptr();
                for k in 0..n {
                    unsafe {
                        core::ptr::swap(pa.add(i * n + k), pa.add(pivot * n + k));
                    }
                }
                let px = x.as_mut_ptr();
                for k in 0..rhs_cols {
                    unsafe {
                        core::ptr::swap(px.add(i * rhs_cols + k), px.add(pivot * rhs_cols + k));
                    }
                }
            }
            let inv_pivot = 1.0 / unsafe { *a.get_unchecked(i * n + i) };
            for val in unsafe { a.get_unchecked_mut(i * n..(i + 1) * n) }.iter_mut() {
                *val *= inv_pivot;
            }
            for val in unsafe { x.get_unchecked_mut(i * rhs_cols..(i + 1) * rhs_cols) }.iter_mut() {
                *val *= inv_pivot;
            }

            let (a_before_i, a_from_i) = unsafe { a.get_unchecked_mut(..n * n).split_at_mut(i * n) };
            let (a_pivot_row, a_after_i) = a_from_i.split_at_mut(n);
            let (x_before_i, x_from_i) = unsafe { x.get_unchecked_mut(..n * rhs_cols).split_at_mut(i * rhs_cols) };
            let (x_pivot_row, x_after_i) = x_from_i.split_at_mut(rhs_cols);

            for (a_row, x_row) in a_before_i.chunks_exact_mut(n).zip(x_before_i.chunks_exact_mut(rhs_cols)) {
                let factor = unsafe { *a_row.get_unchecked(i) };
                if factor.abs() > 1e-14 {
                    row_axpy(a_row, a_pivot_row, -factor);
                    row_axpy(x_row, x_pivot_row, -factor);
                }
            }
            for (a_row, x_row) in a_after_i.chunks_exact_mut(n).zip(x_after_i.chunks_exact_mut(rhs_cols)) {
                let factor = unsafe { *a_row.get_unchecked(i) };
                if factor.abs() > 1e-14 {
                    row_axpy(a_row, a_pivot_row, -factor);
                    row_axpy(x_row, x_pivot_row, -factor);
                }
            }
        }

        let out_rows = if rhs.rows == 1 && rhs.cols == n { 1 } else { n };
        let out_cols = if rhs.rows == 1 && rhs.cols == n { n } else { rhs_cols };
        Ok(Matrix::new(out_rows, out_cols, x))
    }

    /// Computes eigenvalues of a square matrix (exact quadratic formula for 2x2, Jacobi method for symmetric NxN)
    pub fn eigenvalues(&self) -> Result<Vec<(f64, f64)>, String> {
        if self.rows != self.cols {
            return Err("Eigenvalues require a square matrix".to_string());
        }
        let n = self.rows;
        if n == 0 || self.data.len() < n * n {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![(unsafe { *self.data.get_unchecked(0) }, 0.0)]);
        }
        if n == 2 {
            let a = self.get(0, 0);
            let b = self.get(0, 1);
            let c = self.get(1, 0);
            let d = self.get(1, 1);
            let tr = a + d;
            let det = a * d - b * c;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                let s = disc.sqrt();
                return Ok(vec![((tr - s) * 0.5, 0.0), ((tr + s) * 0.5, 0.0)]);
            } else {
                let s = (-disc).sqrt() * 0.5;
                return Ok(vec![(tr * 0.5, -s), (tr * 0.5, s)]);
            }
        }

        // Classical Jacobi iterative diagonalization for symmetric N×N matrices
        let mut a = self.data.clone();
        for _iter in 0..100 {
            crate::engine::evaluator::check_task_aborted()?;
            let mut max_off = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                let row_base = i * n;
                for j in (i + 1)..n {
                    let v = unsafe { (*a.get_unchecked(row_base + j)).abs() };
                    if v > max_off {
                        max_off = v;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_off < 1e-13 {
                break;
            }
            let app = unsafe { *a.get_unchecked(p * n + p) };
            let aqq = unsafe { *a.get_unchecked(q * n + q) };
            let apq = unsafe { *a.get_unchecked(p * n + q) };
            let theta = 0.5 * (aqq - app) / apq;
            let t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                -1.0 / (-theta + (1.0 + theta * theta).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            unsafe {
                *a.get_unchecked_mut(p * n + p) = app - t * apq;
                *a.get_unchecked_mut(q * n + q) = aqq + t * apq;
                *a.get_unchecked_mut(p * n + q) = 0.0;
                *a.get_unchecked_mut(q * n + p) = 0.0;
            }

            for r in 0..n {
                if r != p && r != q {
                    unsafe {
                        let arp = *a.get_unchecked(r * n + p);
                        let arq = *a.get_unchecked(r * n + q);
                        let new_rp = c * arp - s * arq;
                        let new_rq = s * arp + c * arq;
                        *a.get_unchecked_mut(r * n + p) = new_rp;
                        *a.get_unchecked_mut(p * n + r) = new_rp;
                        *a.get_unchecked_mut(r * n + q) = new_rq;
                        *a.get_unchecked_mut(q * n + r) = new_rq;
                    }
                }
            }
        }

        let mut eigs = Vec::with_capacity(n);
        for i in 0..n {
            eigs.push((unsafe { *a.get_unchecked(i * n + i) }, 0.0));
        }
        eigs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(eigs)
    }

    /// Gauss-Jordan elimination with partial pivoting and AVX2 row elimination for Matrix Inverse (O(N³))
    pub fn inverse(&self) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("Inverse requires a square matrix".to_string());
        }
        let n = self.rows;
        if n == 0 || self.data.len() < n * n {
            return Ok(Matrix::zeros(0, 0));
        }
        let mut a = self.data.clone();
        let mut inv = Matrix::identity(n).data;

        for i in 0..n {
            if (i & 15) == 15 {
                crate::engine::evaluator::check_task_aborted()?;
            }
            // Find pivot
            let mut pivot = i;
            let mut max_val = unsafe { (*a.get_unchecked(i * n + i)).abs() };
            for j in (i + 1)..n {
                let val = unsafe { (*a.get_unchecked(j * n + i)).abs() };
                if val > max_val {
                    max_val = val;
                    pivot = j;
                }
            }

            if max_val < 1e-12 {
                return Err("Matrix is singular (not invertible)".to_string());
            }

            // Swap rows in both A and INV
            if pivot != i {
                let pa = a.as_mut_ptr();
                let pinv = inv.as_mut_ptr();
                for k in 0..n {
                    unsafe {
                        core::ptr::swap(pa.add(i * n + k), pa.add(pivot * n + k));
                        core::ptr::swap(pinv.add(i * n + k), pinv.add(pivot * n + k));
                    }
                }
            }

            // Normalize pivot row using reciprocal multiplication
            let inv_pivot = 1.0 / unsafe { *a.get_unchecked(i * n + i) };
            for val in unsafe { a.get_unchecked_mut(i * n..(i + 1) * n) }.iter_mut() {
                *val *= inv_pivot;
            }
            for val in unsafe { inv.get_unchecked_mut(i * n..(i + 1) * n) }.iter_mut() {
                *val *= inv_pivot;
            }

            // Eliminate column elements in all other rows via AVX2+FMA `row_axpy`
            let (a_before_i, a_from_i) = unsafe { a.get_unchecked_mut(..n * n).split_at_mut(i * n) };
            let (a_pivot_row, a_after_i) = a_from_i.split_at_mut(n);
            let (inv_before_i, inv_from_i) = unsafe { inv.get_unchecked_mut(..n * n).split_at_mut(i * n) };
            let (inv_pivot_row, inv_after_i) = inv_from_i.split_at_mut(n);

            for (a_row, inv_row) in a_before_i.chunks_exact_mut(n).zip(inv_before_i.chunks_exact_mut(n)) {
                let factor = unsafe { *a_row.get_unchecked(i) };
                if factor.abs() > 1e-14 {
                    row_axpy(a_row, a_pivot_row, -factor);
                    row_axpy(inv_row, inv_pivot_row, -factor);
                }
            }
            for (a_row, inv_row) in a_after_i.chunks_exact_mut(n).zip(inv_after_i.chunks_exact_mut(n)) {
                let factor = unsafe { *a_row.get_unchecked(i) };
                if factor.abs() > 1e-14 {
                    row_axpy(a_row, a_pivot_row, -factor);
                    row_axpy(inv_row, inv_pivot_row, -factor);
                }
            }
        }

        Ok(Matrix::new(n, n, inv))
    }

    pub fn pow(&self, exp: u32) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("Matrix power requires a square matrix".to_string());
        }
        if exp == 0 {
            return Ok(Matrix::identity(self.rows));
        }
        let mut base = self.clone();
        let mut res = Matrix::identity(self.rows);
        let mut e = exp;

        while e > 0 {
            crate::engine::evaluator::check_task_aborted()?;
            if e & 1 == 1 {
                res = res.mul(&base)?;
            }
            if e > 1 {
                base = base.mul(&base)?;
            }
            e >>= 1;
        }

        Ok(res)
    }
}
