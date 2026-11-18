use std::hash::{Hash, Hasher};
use std::time::Instant;
use rustc_hash::FxHasher;
use crate::config::{update_config, WisdomConfig};

pub struct HardwareProber;

impl HardwareProber {
    /// Zero-allocation `< 2 µs` hardware fingerprint (`CPUID` brand + L1/L2 cache + ISA flags + logical cores)
    /// used to detect moved `.ini` files across machines without invoking DXGI/OpenCL enumeration on warm startup.
    pub fn fast_cpu_fingerprint() -> u64 {
        let mut hasher = FxHasher::default();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);
        cores.hash(&mut hasher);

        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("adx").hash(&mut hasher);
            is_x86_feature_detected!("avx2").hash(&mut hasher);
            is_x86_feature_detected!("bmi2").hash(&mut hasher);
            let max_ext = core::arch::x86_64::__cpuid(0x80000000).eax;
            if max_ext >= 0x80000004 {
                for leaf in [0x80000002, 0x80000003, 0x80000004] {
                    let r = core::arch::x86_64::__cpuid(leaf);
                    r.eax.hash(&mut hasher);
                    r.ebx.hash(&mut hasher);
                    r.ecx.hash(&mut hasher);
                    r.edx.hash(&mut hasher);
                }
            }
            if max_ext >= 0x80000006 {
                core::arch::x86_64::__cpuid(0x80000005).ecx.hash(&mut hasher);
                core::arch::x86_64::__cpuid(0x80000006).ecx.hash(&mut hasher);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::env::consts::ARCH.hash(&mut hasher);
            std::env::consts::OS.hash(&mut hasher);
        }
        let fp = hasher.finish();
        if fp == 0 { 1 } else { fp }
    }

    pub fn probe_and_calibrate(mut existing: WisdomConfig) -> WisdomConfig {
        let probed = Self::probe_system();
        existing.hw_fingerprint = probed.hw_fingerprint;
        existing.calib_sealed = false;
        existing.cpu_name = probed.cpu_name;
        existing.l1_data_cache_kb = probed.l1_data_cache_kb;
        existing.l2_cache_kb = probed.l2_cache_kb;
        existing.logical_cores = probed.logical_cores;
        existing.has_adx = probed.has_adx;
        existing.has_avx2 = probed.has_avx2;
        existing.has_bmi2 = probed.has_bmi2;
        existing.gpu_detected = probed.gpu_detected;
        existing.gpu_dedicated = probed.gpu_dedicated;
        existing.calibration_confidence = 0.0;
        existing.calibration_runs = 0;
        Self::fast_calibrate_startup(existing)
    }

    pub fn probe_system() -> WisdomConfig {
        let mut wisdom = WisdomConfig::default();
        wisdom.hw_fingerprint = Self::fast_cpu_fingerprint();

        // 1. CPU topology & logical cores
        wisdom.logical_cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);

        #[cfg(target_arch = "x86_64")]
        {
            wisdom.has_adx = is_x86_feature_detected!("adx");
            wisdom.has_avx2 = is_x86_feature_detected!("avx2");
            wisdom.has_bmi2 = is_x86_feature_detected!("bmi2");

            let max_ext = core::arch::x86_64::__cpuid(0x80000000).eax;
            if max_ext >= 0x80000004 {
                let mut brand_bytes = [0u8; 48];
                for (idx, leaf) in [0x80000002, 0x80000003, 0x80000004].iter().enumerate() {
                    let res = core::arch::x86_64::__cpuid(*leaf);
                    let offset = idx * 16;
                    brand_bytes[offset..offset + 4].copy_from_slice(&res.eax.to_ne_bytes());
                    brand_bytes[offset + 4..offset + 8].copy_from_slice(&res.ebx.to_ne_bytes());
                    brand_bytes[offset + 8..offset + 12].copy_from_slice(&res.ecx.to_ne_bytes());
                    brand_bytes[offset + 12..offset + 16].copy_from_slice(&res.edx.to_ne_bytes());
                }
                if let Ok(brand_str) = std::str::from_utf8(&brand_bytes) {
                    let trimmed = brand_str.trim_matches(char::from(0)).trim();
                    if !trimmed.is_empty() {
                        wisdom.cpu_name = trimmed.to_string();
                    }
                }
            }
            if max_ext >= 0x80000005 {
                let l1 = (core::arch::x86_64::__cpuid(0x80000005).ecx >> 24) & 0xFF;
                if (16..=128).contains(&l1) {
                    wisdom.l1_data_cache_kb = l1;
                }
            }
            if max_ext >= 0x80000006 {
                let l2 = (core::arch::x86_64::__cpuid(0x80000006).ecx >> 16) & 0xFFFF;
                if (128..=16384).contains(&l2) {
                    wisdom.l2_cache_kb = l2;
                }
            }
            // Intel Deterministic Cache Parameters Leaf `0x04` (populates L1D/L2 on Intel Core CPUs
            // where AMD-style extended leaf `0x80000005` returns 0)
            let max_basic = core::arch::x86_64::__cpuid(0).eax;
            if max_basic >= 4 {
                for subleaf in 0..8u32 {
                    let c = core::arch::x86_64::__cpuid_count(4, subleaf);
                    let cache_type = c.eax & 0x1F;
                    if cache_type == 0 {
                        break;
                    }
                    let level = (c.eax >> 5) & 0x7;
                    let line_size = (c.ebx & 0xFFF) + 1;
                    let partitions = ((c.ebx >> 12) & 0x3FF) + 1;
                    let ways = ((c.ebx >> 22) & 0x3FF) + 1;
                    let sets = c.ecx.saturating_add(1);
                    let kb = ((ways as u64)
                        .saturating_mul(partitions as u64)
                        .saturating_mul(line_size as u64)
                        .saturating_mul(sets as u64)
                        / 1024) as u32;
                    if level == 1 && (cache_type == 1 || cache_type == 3) && (16..=256).contains(&kb) {
                        wisdom.l1_data_cache_kb = kb;
                    } else if level == 2 && (128..=32768).contains(&kb) {
                        wisdom.l2_cache_kb = kb;
                    }
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 (Apple Silicon / Cortex-X): 64 KB+ L1D, 128-bit NEON SIMD standard
            wisdom.l1_data_cache_kb = 64;
            wisdom.l2_cache_kb = 2048;
        }

        #[cfg(unix)]
        if wisdom.cpu_name == "Generic x86-64 Host" {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if let Some((k, v)) = line.split_once(':') {
                        let key = k.trim();
                        if key.eq_ignore_ascii_case("model name")
                            || key.eq_ignore_ascii_case("Hardware")
                            || key.eq_ignore_ascii_case("Processor")
                        {
                            let name = v.trim();
                            if !name.is_empty() {
                                wisdom.cpu_name = name.to_string();
                                break;
                            }
                        }
                    }
                }
            }
            if wisdom.cpu_name == "Generic x86-64 Host" {
                wisdom.cpu_name = format!("Unix {} Host", std::env::consts::ARCH);
            }
        }

        // 2. Query OpenCL GPU Compute Engine + DXGI multi-adapter topology
        let mut gpus: Vec<String> = Vec::new();
        let mut has_dedicated = false;
        let opencl_gpu = super::gpu::GpuComputeEngine::global();
        if opencl_gpu.available {
            has_dedicated = true;
        }

        #[cfg(windows)]
        {
            use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory, IDXGIFactory};
            unsafe {
                if let Ok(factory) = CreateDXGIFactory::<IDXGIFactory>() {
                    let mut i = 0;
                    while let Ok(adapter) = factory.EnumAdapters(i) {
                        if let Ok(desc) = adapter.GetDesc() {
                            let name = String::from_utf16_lossy(&desc.Description)
                                .trim_matches(char::from(0))
                                .trim()
                                .to_string();
                            let is_basic = name.contains("Microsoft Basic Render Driver");
                            let vram_mb = (desc.DedicatedVideoMemory / (1024 * 1024)) as u32;
                            if !is_basic && !name.is_empty() {
                                let matches_opencl = opencl_gpu.available
                                    && (name.contains(&opencl_gpu.device_name)
                                        || opencl_gpu.device_name.contains(&name));
                                if matches_opencl {
                                    has_dedicated = true;
                                    gpus.push(format!(
                                        "{} ({} MB VRAM, {} CUs, OpenCL NTT Active)",
                                        name, vram_mb, opencl_gpu.compute_units
                                    ));
                                } else if vram_mb > 512 {
                                    has_dedicated = true;
                                    gpus.push(format!("{} ({} MB VRAM)", name, vram_mb));
                                } else {
                                    gpus.push(format!("{} (Integrated)", name));
                                }
                            }
                        }
                        i += 1;
                    }
                }
            }
        }

        if gpus.is_empty() && opencl_gpu.available {
            gpus.push(format!(
                "{} ({} CUs, OpenCL NTT Active)",
                opencl_gpu.device_name, opencl_gpu.compute_units
            ));
        }

        if !gpus.is_empty() {
            wisdom.gpu_detected = gpus.join(" + ");
            wisdom.gpu_dedicated = has_dedicated;
        } else {
            wisdom.gpu_detected = "CPU SIMD Engine (No OpenCL GPU ICD)".to_string();
            wisdom.gpu_dedicated = false;
        }

        wisdom
    }

    /// Empirical multi-tier startup calibration (`N = 5` converged runs + `12%` ε-indifference cutoff tie-break:
    /// Comba vs Karatsuba vs Multi-Core Karatsuba vs 2-Prime/GPU NTT).
    pub fn fast_calibrate_startup(mut wisdom: WisdomConfig) -> WisdomConfig {
        if wisdom.hw_fingerprint == 0 {
            wisdom.hw_fingerprint = Self::fast_cpu_fingerprint();
        }
        if wisdom.calib_sealed && wisdom.calibration_confidence >= 1.0 && wisdom.calibration_runs > 0 {
            sync_cutoffs_from_config(&wisdom);
            return wisdom;
        }

        const N_RUNS: usize = 5;
        let mut times_comba = [0u128; N_RUNS];
        let mut times_karatsuba = [0u128; N_RUNS];
        let mut times_fmt = [0u128; N_RUNS];

        let limbs_32 = [0xAAAAAAAAAAAAAAAAu64; 32];
        let mut out = [0u64; 64];

        // Single warm-up pass before timed samples
        super::bignum::comba_mul(&limbs_32, &limbs_32, &mut out);
        super::bignum::karatsuba_mul(&limbs_32, &limbs_32, &mut out, 16);

        for i in 0..N_RUNS {
            let t0 = Instant::now();
            super::bignum::comba_mul(&limbs_32, &limbs_32, &mut out);
            times_comba[i] = t0.elapsed().as_nanos();

            let t1 = Instant::now();
            super::bignum::karatsuba_mul(&limbs_32, &limbs_32, &mut out, 16);
            times_karatsuba[i] = t1.elapsed().as_nanos();

            let t2 = Instant::now();
            let _ = super::bignum::format_limbs_dec(&limbs_32);
            times_fmt[i] = t2.elapsed().as_nanos();
        }

        // Statistical robust analysis (Median & MAD across N = 5)
        times_comba.sort_unstable();
        times_karatsuba.sort_unstable();
        times_fmt.sort_unstable();

        let median_comba = times_comba[N_RUNS / 2];
        let median_karat = times_karatsuba[N_RUNS / 2];
        let median_fmt = times_fmt[N_RUNS / 2];

        let best_mul_ns = median_comba.min(median_karat).clamp(100, 20_000) as u64;
        let best_fmt_ns = median_fmt.clamp(500, 100_000) as u64;

        CALIBRATED_32LIMB_MUL_NS.store(best_mul_ns, std::sync::atomic::Ordering::Relaxed);
        CALIBRATED_32LIMB_FMT_NS.store(best_fmt_ns, std::sync::atomic::Ordering::Relaxed);
        wisdom.calibrated_32limb_mul_ns = best_mul_ns;
        wisdom.calibrated_32limb_fmt_ns = best_fmt_ns;

        if wisdom.calibration_confidence >= 1.0 && wisdom.calibration_runs > 0 {
            wisdom.calib_sealed = true;
            sync_cutoffs_from_config(&wisdom);
            return wisdom;
        }

        // 12% ε-indifference tie-break: prefer branch-straight Comba unless Karatsuba wins by >= 12%
        let cutoff = if median_karat.saturating_mul(100) <= median_comba.saturating_mul(88) {
            20
        } else {
            24
        };
        KARATSUBA_CUTOFF_LIMBS.store(cutoff, std::sync::atomic::Ordering::Relaxed);

        // Calibrate Multi-Core Parallel Karatsuba cutoff (`toom3_cutoff_limbs`) at 128 limbs (median of 3)
        let limbs_128 = [0x5A5A5A5A5A5A5A5Au64; 128];
        let mut out_256 = [0u64; 256];
        let mut seq_128 = [0u128; 3];
        let mut par_128 = [0u128; 3];
        for r in 0..3 {
            let t_seq = Instant::now();
            super::bignum::karatsuba_mul(&limbs_128, &limbs_128, &mut out_256, cutoff);
            seq_128[r] = t_seq.elapsed().as_nanos();

            let t_par = Instant::now();
            super::bignum::par_karatsuba_mul(&limbs_128, &limbs_128, &mut out_256, cutoff);
            par_128[r] = t_par.elapsed().as_nanos();
        }
        seq_128.sort_unstable();
        par_128.sort_unstable();

        // Micro-Horizon Amdahl Gate: require >= 12% wall-clock win before waking Rayon worker pool at 128 limbs
        let par_cutoff = if par_128[1].saturating_mul(100) <= seq_128[1].saturating_mul(88) {
            128
        } else {
            192
        };
        TOOM3_CUTOFF_LIMBS.store(par_cutoff, std::sync::atomic::Ordering::Relaxed);

        // Calibrate 2-Prime + GPU OpenCL NTT cutoff (`ntt_cutoff_limbs`) at 512 limbs (warm-up + median of 3)
        let limbs_512 = [0x9E3779B97F4A7C15u64; 512];
        let mut out_1024 = [0u64; 1024];
        super::bignum::ntt_mul(&limbs_512, &limbs_512, &mut out_1024);

        let mut karat_512 = [0u128; 3];
        let mut ntt_512 = [0u128; 3];
        for r in 0..3 {
            let t_k512 = Instant::now();
            super::bignum::par_karatsuba_mul(&limbs_512, &limbs_512, &mut out_1024, cutoff);
            karat_512[r] = t_k512.elapsed().as_nanos();

            let t_ntt512 = Instant::now();
            super::bignum::ntt_mul(&limbs_512, &limbs_512, &mut out_1024);
            ntt_512[r] = t_ntt512.elapsed().as_nanos();
        }
        karat_512.sort_unstable();
        ntt_512.sort_unstable();

        let ntt_cutoff = if ntt_512[1].saturating_mul(100) <= karat_512[1].saturating_mul(88) {
            512
        } else {
            768
        };
        NTT_CUTOFF_LIMBS.store(ntt_cutoff, std::sync::atomic::Ordering::Relaxed);

        let mad_comba = compute_mad5(&times_comba, median_comba);
        let cv = if median_comba > 0 { (1.4826 * mad_comba as f64) / (median_comba as f64) } else { 0.0 };

        wisdom.calibration_runs += (N_RUNS + 6) as u32;
        wisdom.calibration_cv = cv as f32;
        wisdom.karatsuba_cutoff_limbs = cutoff as u32;
        wisdom.toom3_cutoff_limbs = par_cutoff as u32;
        wisdom.ntt_cutoff_limbs = ntt_cutoff as u32;

        if cv <= 0.12 {
            wisdom.calibration_confidence = 1.0;
            wisdom.calib_sealed = true;
        } else {
            wisdom.calibration_confidence = 0.85;
            wisdom.calib_sealed = false;
        }

        wisdom
    }
}

pub fn sync_cutoffs_from_config(w: &WisdomConfig) {
    if w.calibrated_32limb_mul_ns >= 50 {
        CALIBRATED_32LIMB_MUL_NS.store(
            w.calibrated_32limb_mul_ns.clamp(100, 20_000),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if w.calibrated_32limb_fmt_ns >= 100 {
        CALIBRATED_32LIMB_FMT_NS.store(
            w.calibrated_32limb_fmt_ns.clamp(500, 100_000),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if w.karatsuba_cutoff_limbs >= 12 {
        KARATSUBA_CUTOFF_LIMBS.store(
            w.karatsuba_cutoff_limbs as usize,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if w.toom3_cutoff_limbs >= 64 {
        TOOM3_CUTOFF_LIMBS.store(
            w.toom3_cutoff_limbs as usize,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if w.ntt_cutoff_limbs >= 256 {
        NTT_CUTOFF_LIMBS.store(
            w.ntt_cutoff_limbs as usize,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    // Hardware-Adaptive Register Width, L1D GEMM Tile & L2/Core Parallelism Governance:
    // - SIMD/ALU bit-width tier (`256` AVX2+FMA, `128` NEON/u128, `64` scalar 64-bit, `32` 32-bit)
    let simd_bits: usize = if cfg!(target_arch = "x86_64") && w.has_avx2 {
        256
    } else if cfg!(target_arch = "aarch64") || cfg!(target_pointer_width = "64") {
        128
    } else {
        64
    };
    CPU_SIMD_BITS.store(simd_bits, std::sync::atomic::Ordering::Relaxed);

    // - L1D-Optimal Matrix GEMM Block Dimension (`3 * B^2 * 8 <= 0.75 * L1D_bytes`, aligned to 8 `f64`s = 2 AVX2 vectors)
    let l1_kb = w.l1_data_cache_kb.clamp(16, 256) as usize;
    let block_dim = if l1_kb >= 96 {
        64
    } else if l1_kb >= 48 {
        40
    } else if l1_kb >= 32 {
        32
    } else {
        24
    };
    L1_MATRIX_BLOCK_DIM.store(block_dim, std::sync::atomic::Ordering::Relaxed);

    // - Multi-Core Matrix GEMM Parallel Crossover (`m * k * n` FMAs) scaled to probed `logical_cores`
    let cores = w.logical_cores.max(1) as usize;
    let par_mat_flops = if cores <= 2 {
        128 * 128 * 128
    } else if cores >= 12 {
        56 * 56 * 56
    } else {
        64 * 64 * 64
    };
    PARALLEL_MATRIX_FLOPS.store(par_mat_flops, std::sync::atomic::Ordering::Relaxed);

    // - Concurrent Multi-Radix Formatting Limb Crossover (`DEC/HEX/OCT/BIN` parallel conversion)
    let radix_par_limbs = if cores <= 2 {
        usize::MAX
    } else {
        (w.l2_cache_kb as usize / 8).clamp(64, 256)
    };
    RADIX_PAR_LIMBS.store(radix_par_limbs, std::sync::atomic::Ordering::Relaxed);
}

static CALIBRATED_32LIMB_MUL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(750);
static CALIBRATED_32LIMB_FMT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(4_500);
static KARATSUBA_CUTOFF_LIMBS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(24);
static TOOM3_CUTOFF_LIMBS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);
static NTT_CUTOFF_LIMBS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(768);
static CPU_SIMD_BITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(256);
static L1_MATRIX_BLOCK_DIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(32);
static PARALLEL_MATRIX_FLOPS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(64 * 64 * 64);
static RADIX_PAR_LIMBS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(64);

#[inline(always)]
pub fn cpu_simd_width_bits() -> usize {
    CPU_SIMD_BITS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn l1_matrix_block_dim() -> usize {
    L1_MATRIX_BLOCK_DIM.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn parallel_matrix_flops_cutoff() -> usize {
    PARALLEL_MATRIX_FLOPS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn radix_parallel_limbs_cutoff() -> usize {
    RADIX_PAR_LIMBS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn calibrated_32limb_mul_ns() -> u64 {
    CALIBRATED_32LIMB_MUL_NS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn calibrated_32limb_fmt_ns() -> u64 {
    CALIBRATED_32LIMB_FMT_NS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn karatsuba_cutoff_limbs() -> usize {
    KARATSUBA_CUTOFF_LIMBS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn toom3_cutoff_limbs() -> usize {
    TOOM3_CUTOFF_LIMBS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn ntt_cutoff_limbs() -> usize {
    NTT_CUTOFF_LIMBS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Powers of 3 (`3^0 .. 3^20`) in L1d for 0-float branchless `O(N^{log2 3})` Karatsuba work estimation.
const POW3_U64: [u64; 21] = [
    1,
    3,
    9,
    27,
    81,
    243,
    729,
    2_187,
    6_561,
    19_683,
    59_049,
    177_147,
    531_441,
    1_594_323,
    4_782_969,
    14_348_907,
    43_046_721,
    129_140_163,
    387_420_489,
    1_162_261_467,
    3_486_784_401,
];

/// Pure integer `LZCNT` + piecewise-exact `3^k` estimator for `(limbs / 32)^{log2 3}` in Q10 fixed-point (`1024 == 1.0`),
/// eliminating `f64::powf(1.585)` transcendental calls (`~75 cyc -> 3 cyc`) on every keystroke AST classification.
#[inline(always)]
fn karatsuba_scale_q10(limbs: u64) -> u64 {
    if limbs <= 32 {
        return ((limbs * limbs) << 10) >> 10; // `(limbs / 32)^2 * 1024 == limbs^2`
    }
    // `ratio = limbs / 32 = 2^k * (1 + frac)` where `k = floor(log2(limbs)) - 5`
    let log2_l = (63 - limbs.leading_zeros()) as usize;
    let k = log2_l.saturating_sub(5).min(20);
    let base_pow2 = 32u64 << k;
    // Linear chord between `(base_pow2 -> 3^k)` and `(2*base_pow2 -> 3^{k+1})`: `3^k * (1 + 2 * frac)`
    let p3 = POW3_U64[k];
    let rem = limbs - base_pow2;
    let numer_q10 = (base_pow2 << 10) + (rem << 11);
    p3.saturating_mul(numer_q10) >> (k + 5)
}

/// Estimates CPU nanoseconds required to multiply an `a_bits` integer by a `b_bits` integer
/// and convert the resulting limbs to base-10 using pure integer ALU arithmetic (`0` `f64`/`powf`).
#[inline]
pub fn estimate_bignum_work_nanos(a_bits: u64, b_bits: u64) -> u64 {
    let out_bits = a_bits.saturating_add(b_bits);
    if out_bits <= 64 {
        return 15;
    }
    if out_bits <= 256 {
        return 80;
    }
    let limbs_a = ((a_bits + 63) >> 6).max(1);
    let limbs_b = ((b_bits + 63) >> 6).max(1);
    let out_limbs = limbs_a.saturating_add(limbs_b);

    let mul_32 = calibrated_32limb_mul_ns();
    let fmt_32 = calibrated_32limb_fmt_ns();

    let prod_ab = (limbs_a as u128) * (limbs_b as u128);
    let avg_in = prod_ab.isqrt() as u64;
    let mul_ns = if avg_in <= 24 {
        ((mul_32 as u128) * prod_ab / 1024) as u64
    } else {
        mul_32.saturating_mul(karatsuba_scale_q10(avg_in)) >> 10
    };
    let out_sq = (out_limbs as u128) * (out_limbs as u128);
    let fmt_ns = (((fmt_32 as u128) * out_sq) >> 10).min(u64::MAX as u128) as u64;

    mul_ns.saturating_add(fmt_ns)
}

/// Estimates total CPU nanoseconds required to compute `base ^ exp` (where `base` has `base_bits` bits)
/// via binary exponentiation plus final base-10 radix conversion (`0` float arithmetic).
#[inline]
pub fn estimate_pow_work_nanos(base_bits: u64, exp: u64) -> u64 {
    if exp <= 1 || base_bits <= 1 {
        return 20;
    }
    let out_bits = base_bits.saturating_mul(exp);
    if out_bits <= 64 {
        return 35;
    }
    if out_bits <= 256 {
        return 140;
    }
    let final_half_bits = out_bits >> 1;
    let mul_and_fmt = estimate_bignum_work_nanos(final_half_bits, final_half_bits);
    let half_limbs = ((final_half_bits + 63) >> 6).max(1);
    let extra_mul_steps = calibrated_32limb_mul_ns().saturating_mul(karatsuba_scale_q10(half_limbs)) >> 10;
    mul_and_fmt.saturating_add(extra_mul_steps)
}

fn compute_mad5(sorted_samples: &[u128; 5], median: u128) -> u128 {
    let mut diffs = [0u128; 5];
    for (i, &x) in sorted_samples.iter().enumerate() {
        diffs[i] = if x > median { x - median } else { median - x };
    }
    diffs.sort_unstable();
    diffs[2]
}

#[cfg(windows)]
fn is_system_cpu_quiet_for_calibration() -> bool {
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        lo: u32,
        hi: u32,
    }
    impl FileTime {
        #[inline(always)]
        fn to_u64(self) -> u64 {
            (self.lo as u64) | ((self.hi as u64) << 32)
        }
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemTimes(
            lp_idle_time: *mut FileTime,
            lp_kernel_time: *mut FileTime,
            lp_user_time: *mut FileTime,
        ) -> i32;
    }
    unsafe {
        let (mut i0, mut k0, mut u0) = (FileTime::default(), FileTime::default(), FileTime::default());
        if GetSystemTimes(&mut i0, &mut k0, &mut u0) == 0 {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(60));
        let (mut i1, mut k1, mut u1) = (FileTime::default(), FileTime::default(), FileTime::default());
        if GetSystemTimes(&mut i1, &mut k1, &mut u1) == 0 {
            return true;
        }
        let d_idle = i1.to_u64().saturating_sub(i0.to_u64());
        let d_total = k1
            .to_u64()
            .saturating_sub(k0.to_u64())
            .saturating_add(u1.to_u64().saturating_sub(u0.to_u64()));
        if d_total == 0 {
            return true;
        }
        let busy = d_total.saturating_sub(d_idle);
        busy.saturating_mul(100) <= d_total.saturating_mul(55)
    }
}

#[cfg(unix)]
fn is_system_cpu_quiet_for_calibration() -> bool {
    fn sample_proc_stat() -> Option<(u64, u64)> {
        let txt = std::fs::read_to_string("/proc/stat").ok()?;
        let line = txt.lines().next()?;
        if !line.starts_with("cpu ") {
            return None;
        }
        let mut vals = line[4..].split_whitespace().filter_map(|s| s.parse::<u64>().ok());
        let user = vals.next()?;
        let nice = vals.next()?;
        let system = vals.next()?;
        let idle = vals.next()?;
        let iowait = vals.next().unwrap_or(0);
        let irq = vals.next().unwrap_or(0);
        let softirq = vals.next().unwrap_or(0);
        let steal = vals.next().unwrap_or(0);
        let idle_all = idle.saturating_add(iowait);
        let total = user
            .saturating_add(nice)
            .saturating_add(system)
            .saturating_add(idle_all)
            .saturating_add(irq)
            .saturating_add(softirq)
            .saturating_add(steal);
        Some((idle_all, total))
    }
    let Some((i0, t0)) = sample_proc_stat() else {
        return true;
    };
    std::thread::sleep(std::time::Duration::from_millis(60));
    let Some((i1, t1)) = sample_proc_stat() else {
        return true;
    };
    let d_idle = i1.saturating_sub(i0);
    let d_total = t1.saturating_sub(t0);
    if d_total == 0 {
        return true;
    }
    let busy = d_total.saturating_sub(d_idle);
    busy.saturating_mul(100) <= d_total.saturating_mul(55)
}

#[cfg(all(not(windows), not(unix)))]
fn is_system_cpu_quiet_for_calibration() -> bool {
    true
}

static AFK_CALIBRATING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Spawns background idle calibrator (`N = 5`, external CPU `<= 55%` gate, `CalibSealed`) when window focus is lost
pub fn trigger_afk_calibration() {
    use std::sync::atomic::Ordering;
    let cur_cfg = crate::config::get_config();
    if cur_cfg.wisdom.calib_sealed || cur_cfg.wisdom.calibration_confidence >= 1.0 {
        return;
    }
    if AFK_CALIBRATING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    rayon::spawn(|| {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
            let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }

        let mut config = crate::config::get_config();
        if config.wisdom.calib_sealed
            || config.wisdom.calibration_confidence >= 1.0
            || !is_system_cpu_quiet_for_calibration()
        {
            #[cfg(windows)]
            unsafe {
                use windows::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_NORMAL};
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
            }
            AFK_CALIBRATING.store(false, Ordering::SeqCst);
            return;
        }

        // Deep benchmark (`N = 5` zero-allocation runs)
        const N_RUNS: usize = 5;
        let mut times = [0u128; N_RUNS];
        let test_limbs = [0x123456789ABCDEF0u64; 64];
        let mut out = [0u64; 128];

        for i in 0..N_RUNS {
            let t0 = Instant::now();
            super::bignum::karatsuba_mul(&test_limbs, &test_limbs, &mut out, 24);
            times[i] = t0.elapsed().as_nanos();
        }

        times.sort_unstable();
        let median = times[N_RUNS / 2];
        let mad = compute_mad5(&times, median);
        let cv = if median > 0 { (1.4826 * mad as f64) / (median as f64) } else { 0.0 };

        if cv <= 0.08 {
            config.wisdom.calibration_confidence = 1.0;
            config.wisdom.calib_sealed = true;
            config.wisdom.calibration_cv = cv as f32;
            config.wisdom.calibration_runs += N_RUNS as u32;
            update_config(|c| c.wisdom = config.wisdom.clone());
            crate::config::save_ini(&config);
        }

        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_NORMAL};
            let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
        }
        AFK_CALIBRATING.store(false, Ordering::SeqCst);
    });
}
