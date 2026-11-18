use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use rustc_hash::FxHashMap;

use super::parser::{BinaryOp, Expr, UnaryOp};
use super::value::{CalcValue, MAX_EXACT_BIGNUM_BITS};
use super::logic::LogicEngine;

pub static CANCELLATION_EPOCH: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
pub struct TaskCancelCtx {
    pub epoch: u64,
    pub counter_ptr: *const AtomicU64,
    pub deep_mode: bool,
}

unsafe impl Send for TaskCancelCtx {}
unsafe impl Sync for TaskCancelCtx {}

impl TaskCancelCtx {
    pub const NONE: Self = Self {
        epoch: 0,
        counter_ptr: std::ptr::null(),
        deep_mode: false,
    };

    #[inline(always)]
    pub fn is_aborted(self) -> bool {
        if self.epoch == 0 || self.counter_ptr.is_null() {
            return false;
        }
        unsafe { (*self.counter_ptr).load(Ordering::Relaxed) > self.epoch }
    }
}

thread_local! {
    static THREAD_TASK_CTX: std::cell::Cell<TaskCancelCtx> = const { std::cell::Cell::new(TaskCancelCtx::NONE) };
}

#[inline(always)]
pub fn current_task_cancel_ctx() -> TaskCancelCtx {
    THREAD_TASK_CTX.with(|c| c.get())
}

#[inline(always)]
pub fn is_deep_compute_active() -> bool {
    THREAD_TASK_CTX.with(|c| c.get().deep_mode)
}

/// RAII guard that enables settled deep-compute mode (GPU OpenCL NTT + Multi-Core CPU + BigIntSci) on the current thread.
pub struct DeepComputeGuard {
    prev: TaskCancelCtx,
}

impl DeepComputeGuard {
    #[inline(always)]
    pub fn new(deep_mode: bool) -> Self {
        let prev = THREAD_TASK_CTX.with(|c| {
            let old = c.get();
            c.set(TaskCancelCtx {
                deep_mode,
                ..old
            });
            old
        });
        Self { prev }
    }
}

impl Drop for DeepComputeGuard {
    #[inline(always)]
    fn drop(&mut self) {
        THREAD_TASK_CTX.with(|c| c.set(self.prev));
    }
}

/// RAII guard that binds the current thread to a specific calculation epoch and atomic counter, restoring on drop.
pub struct EpochGuard {
    prev: TaskCancelCtx,
}

impl EpochGuard {
    #[inline(always)]
    pub fn new(epoch: u64) -> Self {
        let prev = THREAD_TASK_CTX.with(|c| {
            let old = c.get();
            let next = if epoch == 0 {
                TaskCancelCtx {
                    epoch: 0,
                    counter_ptr: std::ptr::null(),
                    deep_mode: old.deep_mode,
                }
            } else {
                TaskCancelCtx {
                    epoch,
                    counter_ptr: &CANCELLATION_EPOCH as *const AtomicU64,
                    deep_mode: old.deep_mode,
                }
            };
            c.set(next);
            old
        });
        Self { prev }
    }

    #[inline(always)]
    pub fn with_counter(epoch: u64, counter: &AtomicU64) -> Self { // wiring-ok
        let prev = THREAD_TASK_CTX.with(|c| {
            let old = c.get();
            let next = if epoch == 0 {
                TaskCancelCtx {
                    epoch: 0,
                    counter_ptr: std::ptr::null(),
                    deep_mode: old.deep_mode,
                }
            } else {
                TaskCancelCtx {
                    epoch,
                    counter_ptr: counter as *const AtomicU64,
                    deep_mode: old.deep_mode,
                }
            };
            c.set(next);
            old
        });
        Self { prev }
    }

    #[inline(always)]
    pub fn with_counter_and_mode(epoch: u64, counter: &AtomicU64, deep_mode: bool) -> Self {
        let next = if epoch == 0 {
            TaskCancelCtx {
                epoch: 0,
                counter_ptr: std::ptr::null(),
                deep_mode,
            }
        } else {
            TaskCancelCtx {
                epoch,
                counter_ptr: counter as *const AtomicU64,
                deep_mode,
            }
        };
        let prev = THREAD_TASK_CTX.with(|c| {
            let old = c.get();
            c.set(next);
            old
        });
        Self { prev }
    }

    #[inline(always)]
    pub fn from_ctx(ctx: TaskCancelCtx) -> Self {
        let prev = THREAD_TASK_CTX.with(|c| {
            let old = c.get();
            c.set(ctx);
            old
        });
        Self { prev }
    }
}

impl Drop for EpochGuard {
    #[inline(always)]
    fn drop(&mut self) {
        THREAD_TASK_CTX.with(|c| c.set(self.prev));
    }
}

/// Zero-overhead inner-loop check: returns true if the active calculation epoch on this thread
/// has been superseded by newer user input (`counter > thread_epoch`).
#[inline(always)]
pub fn is_task_aborted() -> bool {
    THREAD_TASK_CTX.with(|c| c.get().is_aborted())
}

/// Returns `Err("Aborted by newer input")` immediately if the current epoch has been superseded.
#[inline(always)]
pub fn check_task_aborted() -> Result<(), String> {
    if is_task_aborted() {
        Err("Aborted by newer input".to_string())
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub value: CalcValue,
    pub elapsed_nanos: u128,
    pub canonical_echo: String,
    pub is_speculative: bool,
}

pub struct Evaluator<'a> {
    pub vars: &'a mut FxHashMap<String, CalcValue>,
    pub epoch: u64,
}

impl<'a> Evaluator<'a> {
    pub fn new(vars: &'a mut FxHashMap<String, CalcValue>, epoch: u64) -> Self {
        Self { vars, epoch }
    }

    pub fn evaluate_expr(&mut self, expr: &Expr) -> Result<CalcValue, String> {
        // Cooperative cancellation check (active only when epoch > 0)
        let ctx = current_task_cancel_ctx();
        let aborted = if ctx.epoch > 0 && !ctx.counter_ptr.is_null() {
            ctx.is_aborted()
        } else {
            self.epoch > 0 && CANCELLATION_EPOCH.load(Ordering::Relaxed) > self.epoch
        };
        if aborted {
            return Err("Aborted by newer input".to_string());
        }

        match expr {
            Expr::Literal(val) => Ok(val.clone()),
            Expr::Var(name) => {
                let lower = name.to_lowercase();
                if let Some(val) = self.vars.get(name) {
                    return Ok(val.clone());
                }
                // Propositional logic variable fallback (p, q, r, s, t, u or a, b, c, d, e, f)
                match lower.as_str() {
                    "true" | "top" => Ok(CalcValue::Logic { table: !0u64, var_count: 1, tautology: true, contradiction: false }),
                    "false" | "bot" => Ok(CalcValue::Logic { table: 0u64, var_count: 1, tautology: false, contradiction: true }),
                    "pi" | "π" => Ok(CalcValue::FastFloat(std::f64::consts::PI)),
                    "e" => Ok(CalcValue::FastFloat(std::f64::consts::E)),
                    "tau" | "τ" => Ok(CalcValue::FastFloat(std::f64::consts::TAU)),
                    "phi" | "φ" => Ok(CalcValue::FastFloat(1.6180339887498948482)),
                    "c" => Ok(CalcValue::FastInt(299_792_458)),
                    "g" => Ok(CalcValue::FastFloat(9.80665)),
                    "h" => Ok(CalcValue::FastFloat(6.62607015e-34)),
                    "k" | "kb" => Ok(CalcValue::FastFloat(1.380649e-23)),
                    "na" | "avogadro" => Ok(CalcValue::FastFloat(6.02214076e23)),
                    "inf" | "infinity" | "∞" => Ok(CalcValue::FastFloat(f64::INFINITY)),
                    "p" | "a" => Ok(CalcValue::Logic { table: LogicEngine::var_bitboard(0), var_count: 1, tautology: false, contradiction: false }),
                    "q" | "b" => Ok(CalcValue::Logic { table: LogicEngine::var_bitboard(1), var_count: 2, tautology: false, contradiction: false }),
                    "r" => Ok(CalcValue::Logic { table: LogicEngine::var_bitboard(2), var_count: 3, tautology: false, contradiction: false }),
                    "s" | "d" => Ok(CalcValue::Logic { table: LogicEngine::var_bitboard(3), var_count: 4, tautology: false, contradiction: false }),
                    "t" => Ok(CalcValue::Logic { table: LogicEngine::var_bitboard(4), var_count: 5, tautology: false, contradiction: false }),
                    "u" | "f" => Ok(CalcValue::Logic { table: LogicEngine::var_bitboard(5), var_count: 6, tautology: false, contradiction: false }),
                    "i" | "j" => Ok(CalcValue::Complex { re: 0.0, im: 1.0 }),
                    "x" => Ok(CalcValue::Polynomial { coeffs: vec![0.0, 1.0] }),
                    _ => Err(format!("Undefined variable or symbol: '{}'", name)),
                }
            }
            Expr::Unary(op, inner) => {
                let v = self.evaluate_expr(inner)?;
                match op {
                    UnaryOp::Pos => Ok(v),
                    UnaryOp::Neg => CalcValue::FastInt(0).sub(&v),
                    UnaryOp::LogicNot => {
                        let (table, var_count) = match Self::as_logic(&v) {
                            Some(lv) => lv,
                            None => return Err("Unary 'not' requires a logic proposition".to_string()),
                        };
                        let inv = LogicEngine::not(table);
                        Ok(CalcValue::Logic {
                            table: inv,
                            var_count,
                            tautology: LogicEngine::is_tautology(inv, var_count),
                            contradiction: LogicEngine::is_contradiction(inv, var_count),
                        })
                    }
                }
            }
            Expr::Binary(op, left, right) => {
                let l = self.evaluate_expr(left)?;
                let r = self.evaluate_expr(right)?;

                match op {
                    BinaryOp::Add => l.add(&r),
                    BinaryOp::Sub => l.sub(&r),
                    BinaryOp::Mul => l.mul(&r),
                    BinaryOp::Div => l.div(&r),
                    BinaryOp::FloorDiv => l.floor_div(&r),
                    BinaryOp::Mod => l.modulo(&r),
                    BinaryOp::Pow => {
                        if matches!((&l, &r), (CalcValue::Logic { .. }, CalcValue::Logic { .. })) {
                            self.eval_logic_binary(&l, &r, LogicEngine::xor)
                        } else {
                            l.pow(&r)
                        }
                    }
                    BinaryOp::Equals => {
                        if let (Some(_), Some(_)) = (Self::as_logic(&l), Self::as_logic(&r)) {
                            self.eval_logic_binary(&l, &r, LogicEngine::iff)
                        } else {
                            let eq = l == r || l.cmp_scalar(&r) == core::cmp::Ordering::Equal;
                            Ok(CalcValue::FastInt(if eq { 1 } else { 0 }))
                        }
                    }
                    // Logic Operations
                    BinaryOp::LogicAnd => self.eval_logic_binary(&l, &r, LogicEngine::and),
                    BinaryOp::LogicOr => self.eval_logic_binary(&l, &r, LogicEngine::or),
                    BinaryOp::LogicXor => self.eval_logic_binary(&l, &r, LogicEngine::xor),
                    BinaryOp::LogicImplies => self.eval_logic_binary(&l, &r, LogicEngine::implies),
                    BinaryOp::LogicIff => self.eval_logic_binary(&l, &r, LogicEngine::iff),
                }
            }
            Expr::FnCall(name, args) => {
                self.eval_fn(name, args)
            }
            Expr::Matrix(rows, cols, elems) => {
                let mut data = Vec::with_capacity(elems.len());
                for e in elems {
                    data.push(self.evaluate_expr(e)?.to_f64());
                }
                Ok(CalcValue::Matrix {
                    rows: *rows,
                    cols: *cols,
                    data,
                })
            }
        }
    }

    fn as_logic(val: &CalcValue) -> Option<(u64, u8)> {
        match val {
            CalcValue::Logic { table, var_count, .. } => Some((*table, *var_count)),
            CalcValue::FastInt(1) => Some((!0u64, 1)),
            CalcValue::FastInt(0) => Some((0u64, 1)),
            _ => None,
        }
    }

    fn eval_logic_binary<F: Fn(u64, u64) -> u64>(&self, l: &CalcValue, r: &CalcValue, gate: F) -> Result<CalcValue, String> {
        if let (Some((t1, v1)), Some((t2, v2))) = (Self::as_logic(l), Self::as_logic(r)) {
            let max_v = v1.max(v2);
            let res_t = gate(t1, t2);
            Ok(CalcValue::Logic {
                table: res_t,
                var_count: max_v,
                tautology: LogicEngine::is_tautology(res_t, max_v),
                contradiction: LogicEngine::is_contradiction(res_t, max_v),
            })
        } else {
            Err("Logic gate requires proposition inputs".to_string())
        }
    }

    fn eval_fn(&mut self, name: &str, args: &[Expr]) -> Result<CalcValue, String> {
        let f_name = name.to_lowercase();
        match f_name.as_str() {
            "sqrt" | "rt" => {
                let val = self.evaluate_expr(args.first().ok_or("sqrt requires 1 argument")?)?;
                if let CalcValue::Complex { re, im } = val {
                    let r = re.hypot(im);
                    let out_re = ((r + re) * 0.5).max(0.0).sqrt();
                    let out_im = ((r - re) * 0.5).max(0.0).sqrt() * if im < 0.0 { -1.0 } else { 1.0 };
                    return Ok(CalcValue::from_complex(out_re, out_im));
                }
                if let Some(exact) = val.try_exact_sqrt() {
                    return Ok(exact);
                }
                let v = val.to_f64();
                if v < 0.0 {
                    return Ok(CalcValue::from_complex(0.0, (-v).sqrt()));
                }
                Ok(CalcValue::from_f64(v.sqrt()))
            }
            "cbrt" => {
                let val = self.evaluate_expr(args.first().ok_or("cbrt requires 1 argument")?)?;
                if let Some(exact) = val.try_exact_cbrt() {
                    return Ok(exact);
                }
                Ok(CalcValue::from_f64(val.to_f64().cbrt()))
            }
            "sq" | "square" => {
                let v = self.evaluate_expr(args.first().ok_or("sq requires 1 argument")?)?;
                v.mul(&v)
            }
            "cb" | "cube" => {
                let v = self.evaluate_expr(args.first().ok_or("cb requires 1 argument")?)?;
                let sq = v.mul(&v)?;
                sq.mul(&v)
            }
            "root" | "nroot" => {
                if args.len() == 1 {
                    let val = self.evaluate_expr(&args[0])?;
                    if let Some(exact) = val.try_exact_sqrt() {
                        return Ok(exact);
                    }
                    let v = val.to_f64();
                    if v < 0.0 { return Err("sqrt of negative number".to_string()); }
                    Ok(CalcValue::from_f64(v.sqrt()))
                } else if args.len() >= 2 {
                    let n = self.evaluate_expr(&args[0])?.to_f64();
                    let x = self.evaluate_expr(&args[1])?.to_f64();
                    if n == 0.0 { return Err("Cannot take 0-th root".to_string()); }
                    if x < 0.0 && ((n as i64) % 2 == 0) {
                        return Err("Even root of negative number".to_string());
                    }
                    if x < 0.0 {
                        Ok(CalcValue::FastFloat(-((-x).powf(1.0 / n))))
                    } else {
                        Ok(CalcValue::FastFloat(x.powf(1.0 / n)))
                    }
                } else {
                    Err("root requires 1 or 2 arguments".to_string())
                }
            }
            "fib" | "fibonacci" => {
                let v = self.evaluate_expr(args.first().ok_or("fib requires 1 argument")?)?;
                let max_fib = if is_deep_compute_active() { 1_000_000.0 } else { 20_000.0 };
                let n = match v {
                    CalcValue::FastInt(i) if i >= 0 => i as u64,
                    CalcValue::ExactDecimal { mantissa, scale: 0 } if mantissa >= 0 && mantissa <= u64::MAX as i128 => mantissa as u64,
                    CalcValue::InlineBigInt { sign: false, limb_count: 1, limbs } => limbs[0],
                    CalcValue::InlineBigInt { sign: false, .. } | CalcValue::HeapBigInt { sign: false, .. } => {
                        return Err("Fibonacci index exceeds limit (n <= 20000)".to_string());
                    }
                    _ => {
                        let f = v.to_f64();
                        if !f.is_finite() || f < 0.0 || f.fract() != 0.0 {
                            return Err("fib requires a non-negative integer".to_string());
                        }
                        if f > max_fib {
                            return Err("Fibonacci index exceeds limit (n <= 20000)".to_string());
                        }
                        f as u64
                    }
                };
                fibonacci(n)
            }
            "primes" | "prime" | "factors" | "factor" => {
                let v = self.evaluate_expr(args.first().ok_or("primes requires 1 argument")?)?;
                let n = match v {
                    CalcValue::FastInt(i) if i >= 0 => i as u64,
                    CalcValue::ExactDecimal { mantissa, scale: 0 } if mantissa >= 0 && mantissa <= u64::MAX as i128 => mantissa as u64,
                    CalcValue::InlineBigInt { sign: false, limb_count: 1, limbs } => limbs[0],
                    CalcValue::InlineBigInt { sign: false, .. } | CalcValue::HeapBigInt { sign: false, .. } => {
                        return Err("primes argument exceeds 64-bit integer limit".to_string());
                    }
                    _ => {
                        let f = v.to_f64();
                        if !f.is_finite() || f < 0.0 || f.fract() != 0.0 {
                            return Err("primes requires a non-negative integer".to_string());
                        }
                        if f > (u64::MAX as f64) {
                            return Err("primes argument exceeds 64-bit integer limit".to_string());
                        }
                        f as u64
                    }
                };
                let s = super::value::prime_factors_decomposition_str(n);
                check_task_aborted()?;
                Ok(CalcValue::Factored { original: n, display: s })
            }
            "sin" => {
                let val = self.evaluate_expr(args.first().ok_or("sin requires 1 argument")?)?;
                if matches!(val, CalcValue::Polynomial { .. }) {
                    return Err("sin requires numerical argument".to_string());
                }
                if let CalcValue::Complex { re, im } = val {
                    return Ok(CalcValue::from_complex(re.sin() * im.cosh(), re.cos() * im.sinh()));
                }
                Ok(CalcValue::FastFloat(val.to_f64().sin()))
            }
            "cos" => {
                let val = self.evaluate_expr(args.first().ok_or("cos requires 1 argument")?)?;
                if matches!(val, CalcValue::Polynomial { .. }) {
                    return Err("cos requires numerical argument".to_string());
                }
                if let CalcValue::Complex { re, im } = val {
                    return Ok(CalcValue::from_complex(re.cos() * im.cosh(), -re.sin() * im.sinh()));
                }
                Ok(CalcValue::FastFloat(val.to_f64().cos()))
            }
            "tan" => {
                let val = self.evaluate_expr(args.first().ok_or("tan requires 1 argument")?)?;
                if matches!(val, CalcValue::Polynomial { .. }) {
                    return Err("tan requires numerical argument".to_string());
                }
                if let CalcValue::Complex { re, im } = val {
                    let num = CalcValue::from_complex(re.sin() * im.cosh(), re.cos() * im.sinh());
                    let den = CalcValue::from_complex(re.cos() * im.cosh(), -re.sin() * im.sinh());
                    return num.div(&den);
                }
                Ok(CalcValue::FastFloat(val.to_f64().tan()))
            }
            "ln" => {
                let val = self.evaluate_expr(args.first().ok_or("ln requires 1 argument")?)?;
                if matches!(val, CalcValue::Polynomial { .. }) {
                    return Err("ln requires numerical argument".to_string());
                }
                if let CalcValue::Complex { re, im } = val {
                    let r = re.hypot(im);
                    if r == 0.0 { return Err("ln(0) is undefined".to_string()); }
                    return Ok(CalcValue::from_complex(r.ln(), im.atan2(re)));
                }
                let v = val.to_f64();
                if v == 0.0 { return Err("ln(0) is undefined".to_string()); }
                if !v.is_finite() {
                    let l2 = val.log2_mag();
                    if l2.is_finite() {
                        let ln_mag = l2 * std::f64::consts::LN_2;
                        if v < 0.0 {
                            return Ok(CalcValue::from_complex(ln_mag, std::f64::consts::PI));
                        }
                        return Ok(CalcValue::from_f64(ln_mag));
                    }
                }
                if v < 0.0 {
                    return Ok(CalcValue::from_complex((-v).ln(), std::f64::consts::PI));
                }
                Ok(CalcValue::FastFloat(v.ln()))
            }
            "log" | "log10" => {
                if args.len() == 1 {
                    let val = self.evaluate_expr(&args[0])?;
                    let v = val.to_f64();
                    if v <= 0.0 { return Err("log of non-positive number".to_string()); }
                    if !v.is_finite() {
                        let l2 = val.log2_mag();
                        if l2.is_finite() {
                            return Ok(CalcValue::from_f64(l2 * std::f64::consts::LOG10_2));
                        }
                    }
                    Ok(CalcValue::FastFloat(v.log10()))
                } else if args.len() >= 2 {
                    let b_val = self.evaluate_expr(&args[0])?;
                    let v_val = self.evaluate_expr(&args[1])?;
                    let base = b_val.to_f64();
                    let v = v_val.to_f64();
                    if base <= 0.0 || (base - 1.0).abs() < 1e-12 { return Err("Invalid log base".to_string()); }
                    if v <= 0.0 { return Err("log of non-positive number".to_string()); }
                    let l2_v = if v.is_finite() { v.log2() } else { v_val.log2_mag() };
                    let l2_b = if base.is_finite() { base.log2() } else { b_val.log2_mag() };
                    Ok(CalcValue::from_f64(l2_v / l2_b))
                } else {
                    Err("log requires 1 or 2 arguments".to_string())
                }
            }
            "log2" => {
                let val = self.evaluate_expr(args.first().ok_or("log2 requires 1 argument")?)?;
                let v = val.to_f64();
                if v <= 0.0 { return Err("log2 of non-positive number".to_string()); }
                if !v.is_finite() {
                    let l2 = val.log2_mag();
                    if l2.is_finite() {
                        return Ok(CalcValue::from_f64(l2));
                    }
                }
                Ok(CalcValue::FastFloat(v.log2()))
            }
            "pmt" => {
                if args.len() < 3 { return Err("pmt requires 3 arguments: pmt(rate, nper, pv)".to_string()); }
                let rate = self.evaluate_expr(&args[0])?.to_f64();
                let nper = self.evaluate_expr(&args[1])?.to_f64();
                let pv = self.evaluate_expr(&args[2])?.to_f64();
                if rate == 0.0 {
                    if nper == 0.0 { return Err("pmt division by zero".to_string()); }
                    Ok(CalcValue::from_f64(pv / nper))
                } else {
                    let denom = 1.0 - (1.0 + rate).powf(-nper);
                    if denom == 0.0 { return Err("pmt division by zero".to_string()); }
                    Ok(CalcValue::from_f64((pv * rate) / denom))
                }
            }
            "compound" => {
                if args.len() < 4 { return Err("compound requires 4 arguments: compound(p, r, n, t)".to_string()); }
                let p = self.evaluate_expr(&args[0])?.to_f64();
                let r = self.evaluate_expr(&args[1])?.to_f64();
                let n = self.evaluate_expr(&args[2])?.to_f64();
                let t = self.evaluate_expr(&args[3])?.to_f64();
                if n == 0.0 { return Err("compound frequency n cannot be 0".to_string()); }
                let res = p * (1.0 + r / n).powf(n * t);
                Ok(CalcValue::from_f64(res))
            }
            "abs" => {
                let v = self.evaluate_expr(args.first().ok_or("abs requires 1 argument")?)?;
                Ok(v.exact_abs())
            }
            "exp" => {
                let val = self.evaluate_expr(args.first().ok_or("exp requires 1 argument")?)?;
                if let CalcValue::Complex { re, im } = val {
                    let ea = re.exp();
                    return Ok(CalcValue::from_complex(ea * im.cos(), ea * im.sin()));
                }
                Ok(CalcValue::FastFloat(val.to_f64().exp()))
            }
            "round" => {
                let v = self.evaluate_expr(args.first().ok_or("round requires 1 argument")?)?;
                Ok(v.exact_round())
            }
            "floor" => {
                let v = self.evaluate_expr(args.first().ok_or("floor requires 1 argument")?)?;
                Ok(v.exact_floor())
            }
            "ceil" => {
                let v = self.evaluate_expr(args.first().ok_or("ceil requires 1 argument")?)?;
                Ok(v.exact_ceil())
            }
            "det" => {
                let v = self.evaluate_expr(args.first().ok_or("det requires 1 matrix argument")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        let d = mat.determinant()?;
                        Ok(CalcValue::FastFloat(d))
                    }
                    _ => Err("det requires a matrix".to_string()),
                }
            }
            "inv" | "inverse" => {
                let v = self.evaluate_expr(args.first().ok_or("inv requires 1 matrix argument")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        let inv_mat = mat.inverse()?;
                        Ok(CalcValue::Matrix { rows: inv_mat.rows, cols: inv_mat.cols, data: inv_mat.data })
                    }
                    _ => Err("inv requires a matrix".to_string()),
                }
            }
            "transpose" | "t" => {
                let v = self.evaluate_expr(args.first().ok_or("transpose requires 1 matrix argument")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        let tr_mat = mat.transpose();
                        Ok(CalcValue::Matrix { rows: tr_mat.rows, cols: tr_mat.cols, data: tr_mat.data })
                    }
                    _ => Err("transpose requires a matrix".to_string()),
                }
            }
            "trace" | "tr" => {
                let v = self.evaluate_expr(args.first().ok_or("trace requires 1 matrix argument")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        let tr = mat.trace()?;
                        Ok(CalcValue::FastFloat(tr))
                    }
                    _ => Err("trace requires a matrix".to_string()),
                }
            }
            "eye" | "identity" => {
                let n_val = self.evaluate_expr(args.first().ok_or("eye requires dimension n")?)?.to_f64() as usize;
                if n_val == 0 || n_val > 100 {
                    return Err("eye dimension must be between 1 and 100".to_string());
                }
                let mat = super::matrix::Matrix::identity(n_val);
                Ok(CalcValue::Matrix { rows: mat.rows, cols: mat.cols, data: mat.data })
            }
            "fact" | "fac" | "factorial" => {
                let val = self.evaluate_expr(args.first().ok_or("fact requires 1 argument")?)?;
                let max_fact = if is_deep_compute_active() { 100_000.0 } else { 10_000.0 };
                match val {
                    CalcValue::FastInt(n) => fact_i64(n),
                    CalcValue::ExactDecimal { mantissa, scale: 0 } => fact_i64(mantissa as i64),
                    _ => {
                        let f = val.to_f64();
                        if f < 0.0 {
                            return Err("Factorial of negative number".to_string());
                        }
                        if f.fract() == 0.0 && f <= max_fact {
                            fact_i64(f as i64)
                        } else {
                            // Lanczos gamma(f + 1) for fractional arguments
                            Ok(CalcValue::FastFloat(gamma_approx(f + 1.0)))
                        }
                    }
                }
            }
            "gcd" => {
                if args.len() < 2 { return Err("gcd requires 2 arguments".to_string()); }
                let va = self.evaluate_expr(&args[0])?;
                let vb = self.evaluate_expr(&args[1])?;
                if let (Some(a), Some(b)) = (va.to_exact_i128(), vb.to_exact_i128()) {
                    let g = gcd_u128_stein(a.unsigned_abs(), b.unsigned_abs());
                    if g <= i128::MAX as u128 {
                        return Ok(CalcValue::normalize_decimal(g as i128, 0));
                    }
                }
                let a = va.to_f64() as i64;
                let b = vb.to_f64() as i64;
                Ok(CalcValue::FastInt(gcd_i64(a, b)))
            }
            "lcm" => {
                if args.len() < 2 { return Err("lcm requires 2 arguments".to_string()); }
                let va = self.evaluate_expr(&args[0])?;
                let vb = self.evaluate_expr(&args[1])?;
                if let (Some(a), Some(b)) = (va.to_exact_i128(), vb.to_exact_i128()) {
                    return lcm_i128(a, b);
                }
                let a = va.to_f64() as i64;
                let b = vb.to_f64() as i64;
                lcm_i64(a, b).map(CalcValue::FastInt)
            }
            "ncr" | "comb" => {
                if args.len() < 2 { return Err("nCr requires 2 arguments".to_string()); }
                let vn = self.evaluate_expr(&args[0])?;
                let vr = self.evaluate_expr(&args[1])?;
                let n = vn.to_exact_i128().unwrap_or_else(|| vn.to_f64() as i128);
                let r = vr.to_exact_i128().unwrap_or_else(|| vr.to_f64() as i128);
                ncr_i128(n, r)
            }
            "npr" | "perm" => {
                if args.len() < 2 { return Err("nPr requires 2 arguments".to_string()); }
                let vn = self.evaluate_expr(&args[0])?;
                let vr = self.evaluate_expr(&args[1])?;
                let n = vn.to_exact_i128().unwrap_or_else(|| vn.to_f64() as i128);
                let r = vr.to_exact_i128().unwrap_or_else(|| vr.to_f64() as i128);
                npr_i128(n, r)
            }
            "hypot" => {
                if args.len() < 2 { return Err("hypot requires 2 arguments".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_f64();
                let b = self.evaluate_expr(&args[1])?.to_f64();
                Ok(CalcValue::from_f64(a.hypot(b)))
            }
            "sgn" | "sign" => {
                let a = self.evaluate_expr(args.first().ok_or("sgn requires 1 argument")?)?;
                Ok(CalcValue::FastInt(a.exact_sgn()))
            }
            "deg" => {
                let rad = self.evaluate_expr(args.first().ok_or("deg requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(rad * (180.0 / std::f64::consts::PI)))
            }
            "rad" => {
                let deg = self.evaluate_expr(args.first().ok_or("rad requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(deg * (std::f64::consts::PI / 180.0)))
            }

            // --- Advanced & Hyperbolic Trigonometry ---
            "asin" | "arcsin" => {
                let v = self.evaluate_expr(args.first().ok_or("asin requires 1 argument")?)?.to_f64();
                if v < -1.0 || v > 1.0 { return Err("asin domain error: argument must be between -1 and 1".to_string()); }
                Ok(CalcValue::FastFloat(v.asin()))
            }
            "acos" | "arccos" => {
                let v = self.evaluate_expr(args.first().ok_or("acos requires 1 argument")?)?.to_f64();
                if v < -1.0 || v > 1.0 { return Err("acos domain error: argument must be between -1 and 1".to_string()); }
                Ok(CalcValue::FastFloat(v.acos()))
            }
            "atan" | "arctan" => {
                let v = self.evaluate_expr(args.first().ok_or("atan requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(v.atan()))
            }
            "atan2" => {
                if args.len() < 2 { return Err("atan2 requires 2 arguments: atan2(y, x)".to_string()); }
                let y = self.evaluate_expr(&args[0])?.to_f64();
                let x = self.evaluate_expr(&args[1])?.to_f64();
                Ok(CalcValue::FastFloat(y.atan2(x)))
            }
            "sinh" => {
                let v = self.evaluate_expr(args.first().ok_or("sinh requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(v.sinh()))
            }
            "cosh" => {
                let v = self.evaluate_expr(args.first().ok_or("cosh requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(v.cosh()))
            }
            "tanh" => {
                let v = self.evaluate_expr(args.first().ok_or("tanh requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(v.tanh()))
            }
            "asinh" | "arcsinh" => {
                let v = self.evaluate_expr(args.first().ok_or("asinh requires 1 argument")?)?.to_f64();
                Ok(CalcValue::FastFloat(v.asinh()))
            }
            "acosh" | "arccosh" => {
                let v = self.evaluate_expr(args.first().ok_or("acosh requires 1 argument")?)?.to_f64();
                if v < 1.0 { return Err("acosh domain error: argument must be >= 1".to_string()); }
                Ok(CalcValue::FastFloat(v.acosh()))
            }
            "atanh" | "arctanh" => {
                let v = self.evaluate_expr(args.first().ok_or("atanh requires 1 argument")?)?.to_f64();
                if v <= -1.0 || v >= 1.0 { return Err("atanh domain error: argument must be between -1 and 1".to_string()); }
                Ok(CalcValue::FastFloat(v.atanh()))
            }
            "sec" => {
                let v = self.evaluate_expr(args.first().ok_or("sec requires 1 argument")?)?.to_f64();
                let c = v.cos();
                if c == 0.0 { return Err("sec division by zero".to_string()); }
                Ok(CalcValue::FastFloat(1.0 / c))
            }
            "csc" => {
                let v = self.evaluate_expr(args.first().ok_or("csc requires 1 argument")?)?.to_f64();
                let s = v.sin();
                if s == 0.0 { return Err("csc division by zero".to_string()); }
                Ok(CalcValue::FastFloat(1.0 / s))
            }
            "cot" => {
                let v = self.evaluate_expr(args.first().ok_or("cot requires 1 argument")?)?.to_f64();
                let t = v.tan();
                if t == 0.0 { return Err("cot division by zero".to_string()); }
                Ok(CalcValue::FastFloat(1.0 / t))
            }

            // --- Bitwise & Programmer Arithmetic Functions (Full 64-Bit Zero-Float Exactness) ---
            "band" => {
                if args.len() < 2 { return Err("band requires 2 arguments".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()?;
                let b = self.evaluate_expr(&args[1])?.to_exact_u64_bits()?;
                Ok(CalcValue::normalize_decimal((a & b) as i128, 0))
            }
            "bor" => {
                if args.len() < 2 { return Err("bor requires 2 arguments".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()?;
                let b = self.evaluate_expr(&args[1])?.to_exact_u64_bits()?;
                Ok(CalcValue::normalize_decimal((a | b) as i128, 0))
            }
            "bxor" => {
                if args.len() < 2 { return Err("bxor requires 2 arguments".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()?;
                let b = self.evaluate_expr(&args[1])?.to_exact_u64_bits()?;
                Ok(CalcValue::normalize_decimal((a ^ b) as i128, 0))
            }
            "bnot" => {
                let a = self.evaluate_expr(args.first().ok_or("bnot requires 1 argument")?)?.to_exact_u64_bits()?;
                Ok(CalcValue::FastInt((!a) as i64))
            }
            "shl" => {
                if args.len() < 2 { return Err("shl requires 2 arguments: shl(val, shift)".to_string()); }
                let va = self.evaluate_expr(&args[0])?;
                let a = va.to_exact_u64_bits()?;
                let b = (self.evaluate_expr(&args[1])?.to_exact_u64_bits()? as u32).min(63);
                if matches!(va, CalcValue::FastInt(i) if i < 0) {
                    Ok(CalcValue::FastInt(((a as i64) << b) as i64))
                } else {
                    Ok(CalcValue::normalize_decimal((a << b) as i128, 0))
                }
            }
            "shr" => {
                if args.len() < 2 { return Err("shr requires 2 arguments: shr(val, shift)".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()?;
                let b = (self.evaluate_expr(&args[1])?.to_exact_u64_bits()? as u32).min(63);
                Ok(CalcValue::normalize_decimal((a >> b) as i128, 0))
            }
            "ashr" => {
                if args.len() < 2 { return Err("ashr requires 2 arguments: ashr(val, shift)".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()? as i64;
                let b = (self.evaluate_expr(&args[1])?.to_exact_u64_bits()? as u32).min(63);
                Ok(CalcValue::FastInt(a >> b))
            }
            "popcnt" | "popcount" => {
                let a = self.evaluate_expr(args.first().ok_or("popcnt requires 1 argument")?)?.to_exact_u64_bits()?;
                Ok(CalcValue::FastInt(a.count_ones() as i64))
            }
            "clz" => {
                let a = self.evaluate_expr(args.first().ok_or("clz requires 1 argument")?)?.to_exact_u64_bits()?;
                Ok(CalcValue::FastInt(a.leading_zeros() as i64))
            }
            "ctz" => {
                let a = self.evaluate_expr(args.first().ok_or("ctz requires 1 argument")?)?.to_exact_u64_bits()?;
                Ok(CalcValue::FastInt(a.trailing_zeros() as i64))
            }
            "rol" => {
                if args.len() < 2 { return Err("rol requires 2 arguments".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()?;
                let b = self.evaluate_expr(&args[1])?.to_exact_u64_bits()? as u32;
                Ok(CalcValue::normalize_decimal(a.rotate_left(b) as i128, 0))
            }
            "ror" => {
                if args.len() < 2 { return Err("ror requires 2 arguments".to_string()); }
                let a = self.evaluate_expr(&args[0])?.to_exact_u64_bits()?;
                let b = self.evaluate_expr(&args[1])?.to_exact_u64_bits()? as u32;
                Ok(CalcValue::normalize_decimal(a.rotate_right(b) as i128, 0))
            }

            // --- Statistical & Aggregation Functions (Preserving ExactDecimal & BigInt) ---
            "min" => {
                if args.is_empty() { return Err("min requires at least 1 argument".to_string()); }
                let mut best = self.evaluate_expr(&args[0])?;
                for a in &args[1..] {
                    let v = self.evaluate_expr(a)?;
                    if v.cmp_scalar(&best) == core::cmp::Ordering::Less {
                        best = v;
                    }
                }
                Ok(best)
            }
            "max" => {
                if args.is_empty() { return Err("max requires at least 1 argument".to_string()); }
                let mut best = self.evaluate_expr(&args[0])?;
                for a in &args[1..] {
                    let v = self.evaluate_expr(a)?;
                    if v.cmp_scalar(&best) == core::cmp::Ordering::Greater {
                        best = v;
                    }
                }
                Ok(best)
            }
            "sum" => {
                if args.is_empty() { return Ok(CalcValue::FastInt(0)); }
                let mut acc = CalcValue::FastInt(0);
                for a in args {
                    let v = self.evaluate_expr(a)?;
                    acc = acc.add(&v)?;
                }
                Ok(acc)
            }
            "avg" | "mean" => {
                if args.is_empty() { return Err("mean requires at least 1 argument".to_string()); }
                let mut acc = CalcValue::FastInt(0);
                for a in args {
                    let v = self.evaluate_expr(a)?;
                    acc = acc.add(&v)?;
                }
                acc.div(&CalcValue::FastInt(args.len() as i64))
            }
            "median" => {
                if args.is_empty() { return Err("median requires at least 1 argument".to_string()); }
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.evaluate_expr(a)?);
                }
                vals.sort_by(|a, b| a.cmp_scalar(b));
                let len = vals.len();
                if len % 2 == 1 {
                    Ok(vals.swap_remove(len / 2))
                } else {
                    let sum_mid = vals[len / 2 - 1].add(&vals[len / 2])?;
                    sum_mid.div(&CalcValue::FastInt(2))
                }
            }
            "var" | "variance" => {
                if args.len() < 2 { return Err("variance requires at least 2 arguments".to_string()); }
                let mut vals = Vec::with_capacity(args.len());
                let mut sum = 0.0;
                for a in args {
                    let v = self.evaluate_expr(a)?.to_f64();
                    sum += v;
                    vals.push(v);
                }
                let mean = sum / vals.len() as f64;
                let var = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
                Ok(CalcValue::from_f64(var))
            }
            "std" | "stddev" => {
                if args.len() < 2 { return Err("stddev requires at least 2 arguments".to_string()); }
                let mut vals = Vec::with_capacity(args.len());
                let mut sum = 0.0;
                for a in args {
                    let v = self.evaluate_expr(a)?.to_f64();
                    sum += v;
                    vals.push(v);
                }
                let mean = sum / vals.len() as f64;
                let var = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
                Ok(CalcValue::from_f64(var.sqrt()))
            }

            // --- Complex Number Functions ---
            "re" | "real" => {
                let v = self.evaluate_expr(args.first().ok_or("re requires 1 argument")?)?;
                match v {
                    CalcValue::Complex { re, .. } => Ok(CalcValue::from_f64(re)),
                    _ => Ok(CalcValue::from_f64(v.to_f64())),
                }
            }
            "im" | "imag" => {
                let v = self.evaluate_expr(args.first().ok_or("im requires 1 argument")?)?;
                match v {
                    CalcValue::Complex { im, .. } => Ok(CalcValue::from_f64(im)),
                    _ => Ok(CalcValue::FastInt(0)),
                }
            }
            "conj" | "conjugate" => {
                let v = self.evaluate_expr(args.first().ok_or("conj requires 1 argument")?)?;
                match v {
                    CalcValue::Complex { re, im } => Ok(CalcValue::from_complex(re, -im)),
                    _ => Ok(v),
                }
            }
            "arg" | "phase" | "angle" => {
                let v = self.evaluate_expr(args.first().ok_or("arg requires 1 argument")?)?;
                match v {
                    CalcValue::Complex { re, im } => Ok(CalcValue::from_f64(im.atan2(re))),
                    _ => {
                        let f = v.to_f64();
                        Ok(CalcValue::from_f64(if f < 0.0 { std::f64::consts::PI } else { 0.0 }))
                    }
                }
            }

            // --- Advanced Matrix & Linear Algebra Functions ---
            "rank" => {
                let v = self.evaluate_expr(args.first().ok_or("rank requires 1 matrix argument")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        Ok(CalcValue::FastInt(mat.rank() as i64))
                    }
                    _ => Err("rank requires a matrix".to_string()),
                }
            }
            "norm" | "frobenius" => {
                let v = self.evaluate_expr(args.first().ok_or("norm requires 1 argument")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        Ok(CalcValue::from_f64(mat.frobenius_norm()))
                    }
                    CalcValue::Complex { re, im } => Ok(CalcValue::from_f64(re.hypot(im))),
                    _ => Ok(CalcValue::from_f64(v.to_f64().abs())),
                }
            }
            "eig" | "eigenvalues" => {
                let v = self.evaluate_expr(args.first().ok_or("eig requires 1 square matrix")?)?;
                match v {
                    CalcValue::Matrix { rows, cols, data } => {
                        let mat = super::matrix::Matrix::new(rows, cols, data);
                        let eigs = mat.eigenvalues()?;
                        if eigs.iter().all(|&(_, im)| im.abs() < 1e-12) {
                            let reals: Vec<f64> = eigs.into_iter().map(|(re, _)| super::value::snap_float(re)).collect();
                            Ok(CalcValue::Matrix {
                                rows: 1,
                                cols: reals.len(),
                                data: reals,
                            })
                        } else if eigs.len() == 2 {
                            Ok(CalcValue::from_complex(eigs[1].0, eigs[1].1))
                        } else {
                            let reals: Vec<f64> = eigs.into_iter().map(|(re, _)| super::value::snap_float(re)).collect();
                            Ok(CalcValue::Matrix {
                                rows: 1,
                                cols: reals.len(),
                                data: reals,
                            })
                        }
                    }
                    _ => Err("eig requires a square matrix".to_string()),
                }
            }
            "solve" | "roots" | "poly_roots" => {
                if args.is_empty() {
                    return Err("solve/roots requires at least 1 argument".to_string());
                }
                if args.len() == 2 {
                    let a_val = self.evaluate_expr(&args[0])?;
                    let b_val = self.evaluate_expr(&args[1])?;
                    if let (
                        CalcValue::Matrix { rows: r1, cols: c1, data: d1 },
                        CalcValue::Matrix { rows: r2, cols: c2, data: d2 },
                    ) = (&a_val, &b_val)
                    {
                        let mat_a = super::matrix::Matrix::new(*r1, *c1, d1.clone());
                        let mat_b = super::matrix::Matrix::new(*r2, *c2, d2.clone());
                        let sol = mat_a.solve(&mat_b)?;
                        return Ok(CalcValue::Matrix {
                            rows: sol.rows,
                            cols: sol.cols,
                            data: sol.data,
                        });
                    }
                }
                // Polynomial roots: either roots(x^2 - 5x + 6) or roots(1, -5, 6) or roots([1, -5, 6])
                let coeffs = if args.len() == 1 {
                    let val = self.evaluate_expr(&args[0])?;
                    match val {
                        CalcValue::Polynomial { coeffs } => coeffs,
                        CalcValue::Matrix { data, .. } => {
                            let mut rev = data;
                            rev.reverse();
                            rev
                        }
                        _ => return Err("roots requires a polynomial in x or coefficient list".to_string()),
                    }
                } else {
                    // Ascending or descending scalar arguments: e.g. roots(1, -5, 6) for 1*x^2 - 5*x + 6
                    let mut c_desc = Vec::with_capacity(args.len());
                    for a in args {
                        c_desc.push(self.evaluate_expr(a)?.to_f64());
                    }
                    c_desc.reverse();
                    c_desc
                };

                let r_list = super::value::poly_roots(&coeffs)?;
                if r_list.len() == 1 {
                    return Ok(CalcValue::from_complex(r_list[0].0, r_list[0].1));
                }
                if r_list.iter().all(|&(_, im)| im.abs() < 1e-9) {
                    let reals: Vec<f64> = r_list.into_iter().map(|(re, _)| re).collect();
                    return Ok(CalcValue::Matrix {
                        rows: 1,
                        cols: reals.len(),
                        data: reals,
                    });
                }
                // Format complex conjugate root pair
                Ok(CalcValue::from_complex(r_list.last().unwrap().0, r_list.last().unwrap().1))
            }

            // --- Symbolic & Numerical Calculus (Differentiation & Quadrature Integration) ---
            "deriv" | "diff" => {
                if args.is_empty() {
                    return Err("diff requires 1 or 2 arguments: diff(expr) or diff(expr, x0)".to_string());
                }
                if args.len() == 1 {
                    let v = self.evaluate_expr(&args[0])?;
                    if let CalcValue::Polynomial { coeffs } = v {
                        return Ok(super::value::poly_deriv(&coeffs));
                    }
                    return Ok(CalcValue::FastInt(0));
                }
                let x0 = if args.len() >= 3 {
                    self.evaluate_expr(&args[2])?.to_f64()
                } else {
                    self.evaluate_expr(&args[1])?.to_f64()
                };
                // Try exact symbolic polynomial derivative first if 'x' is not bound
                let prev_x = self.vars.remove("x");
                if let Ok(CalcValue::Polynomial { coeffs }) = self.evaluate_expr(&args[0]) {
                    if let Some(px) = prev_x {
                        self.vars.insert("x".to_string(), px);
                    }
                    let d_poly = super::value::poly_deriv(&coeffs);
                    let val = match d_poly {
                        CalcValue::Polynomial { coeffs: dc } => super::value::poly_eval(&dc, x0),
                        other => other.to_f64(),
                    };
                    return Ok(CalcValue::from_f64(val));
                }
                if let Some(px) = prev_x {
                    self.vars.insert("x".to_string(), px);
                }
                check_task_aborted()?;

                // 5-point stencil with Richardson extrapolation
                let h1 = 1e-4 * x0.abs().max(1.0);
                let h2 = h1 * 0.5;
                let d1 = self.eval_five_point_stencil(&args[0], x0, h1)?;
                let d2 = self.eval_five_point_stencil(&args[0], x0, h2)?;
                let richardson = (16.0 * d2 - d1) / 15.0;
                Ok(CalcValue::from_f64(richardson))
            }
            "integrate" | "int" => {
                if args.is_empty() {
                    return Err("integrate requires 1 or 3 arguments: integrate(expr) or integrate(expr, a, b)".to_string());
                }
                if args.len() == 1 {
                    let v = self.evaluate_expr(&args[0])?;
                    if let CalcValue::Polynomial { coeffs } = v {
                        return Ok(super::value::poly_integ(&coeffs));
                    }
                    return Ok(CalcValue::from_polynomial(vec![0.0, v.to_f64()]));
                }
                if args.len() < 3 {
                    return Err("Definite integral requires bounds: integrate(expr, a, b)".to_string());
                }
                let (a, b) = if args.len() >= 4 {
                    (
                        self.evaluate_expr(&args[2])?.to_f64(),
                        self.evaluate_expr(&args[3])?.to_f64(),
                    )
                } else {
                    (
                        self.evaluate_expr(&args[1])?.to_f64(),
                        self.evaluate_expr(&args[2])?.to_f64(),
                    )
                };

                // Try exact symbolic polynomial antiderivative first
                let prev_x = self.vars.remove("x");
                if let Ok(CalcValue::Polynomial { coeffs }) = self.evaluate_expr(&args[0]) {
                    if let Some(px) = prev_x {
                        self.vars.insert("x".to_string(), px);
                    }
                    let antideriv = super::value::poly_integ(&coeffs);
                    let (fa, fb) = match antideriv {
                        CalcValue::Polynomial { coeffs: ic } => (
                            super::value::poly_eval(&ic, a),
                            super::value::poly_eval(&ic, b),
                        ),
                        other => {
                            let c = other.to_f64();
                            (c, c)
                        }
                    };
                    return Ok(CalcValue::from_f64(fb - fa));
                }
                if let Some(px) = prev_x {
                    self.vars.insert("x".to_string(), px);
                }
                check_task_aborted()?;

                // Composite 8-point Gauss-Legendre quadrature across 16 sub-intervals
                let integral = self.eval_gauss_legendre_quadrature(&args[0], a, b)?;
                Ok(CalcValue::from_f64(integral))
            }

            // --- Unit Conversion (Exact Rational SI/NIST `i128` first, `f64` fallback for `rad`) ---
            "convert" => {
                if args.len() < 3 { return Err("convert requires 3 arguments: convert(val, from_unit, to_unit)".to_string()); }
                let val_cv = self.evaluate_expr(&args[0])?;
                let from_str = match &args[1] {
                    Expr::Var(s) => s.clone(),
                    Expr::Literal(val) => val.to_string(),
                    _ => self.evaluate_expr(&args[1])?.to_string(),
                };
                let to_str = match &args[2] {
                    Expr::Var(s) => s.clone(),
                    Expr::Literal(val) => val.to_string(),
                    _ => self.evaluate_expr(&args[2])?.to_string(),
                };
                if let Some((m, s)) = super::value::to_decimal_parts(&val_cv) {
                    if let Some((out_m, out_s)) =
                        crate::engine::units::convert_units_exact(m, s, &from_str, &to_str)?
                    {
                        return Ok(CalcValue::normalize_decimal(out_m, out_s));
                    }
                }
                let res = crate::engine::units::convert_units(val_cv.to_f64(), &from_str, &to_str)?;
                Ok(CalcValue::from_f64(res))
            }
            _ => Err(format!("Unknown function '{}'", name)),
        }
    }

    fn eval_at_x(&mut self, expr: &Expr, x_val: f64) -> Result<f64, String> {
        let prev = self.vars.insert("x".to_string(), CalcValue::FastFloat(x_val));
        let res = self.evaluate_expr(expr);
        if let Some(p) = prev {
            self.vars.insert("x".to_string(), p);
        } else {
            self.vars.remove("x");
        }
        Ok(res?.to_f64())
    }

    fn eval_five_point_stencil(&mut self, expr: &Expr, x0: f64, h: f64) -> Result<f64, String> {
        let fm2 = self.eval_at_x(expr, x0 - 2.0 * h)?;
        let fm1 = self.eval_at_x(expr, x0 - h)?;
        let fp1 = self.eval_at_x(expr, x0 + h)?;
        let fp2 = self.eval_at_x(expr, x0 + 2.0 * h)?;
        Ok((fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h))
    }

    fn eval_gauss_legendre_quadrature(&mut self, expr: &Expr, a: f64, b: f64) -> Result<f64, String> {
        if (b - a).abs() < 1e-15 {
            return Ok(0.0);
        }
        // 8-point Gauss-Legendre abscissas and weights on [-1, 1]
        const NODES: [f64; 4] = [
            0.183434642495649805,
            0.525532409916328986,
            0.796666477413626740,
            0.960289856497536232,
        ];
        const WEIGHTS: [f64; 4] = [
            0.362683783378361983,
            0.313706645877887287,
            0.222381034453374471,
            0.101228536290376259,
        ];
        const PANELS: usize = 16;
        let step = (b - a) / (PANELS as f64);
        let half_step = step * 0.5;
        let mut total = 0.0;

        for p in 0..PANELS {
            check_task_aborted()?;
            let mid = a + ((p as f64) + 0.5) * step;
            let mut panel_sum = 0.0;
            for i in 0..4 {
                let dx = half_step * NODES[i];
                let y1 = self.eval_at_x(expr, mid - dx)?;
                let y2 = self.eval_at_x(expr, mid + dx)?;
                panel_sum += WEIGHTS[i] * (y1 + y2);
            }
            total += panel_sum * half_step;
        }
        Ok(total)
    }
}

/// Stein's Binary GCD (`TZCNT` shifts + subtraction, `0` hardware `DIV` stalls!) across 128-bit unsigned integers.
#[inline]
fn gcd_u128_stein(mut u: u128, mut v: u128) -> u128 {
    if u == 0 {
        return v;
    }
    if v == 0 {
        return u;
    }
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    loop {
        v >>= v.trailing_zeros();
        if u > v {
            core::mem::swap(&mut u, &mut v);
        }
        v -= u;
        if v == 0 {
            break;
        }
    }
    u << shift
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    gcd_u128_stein(a.unsigned_abs() as u128, b.unsigned_abs() as u128) as i64
}

fn lcm_i128(a: i128, b: i128) -> Result<CalcValue, String> {
    if a == 0 || b == 0 {
        return Ok(CalcValue::FastInt(0));
    }
    let ua = a.unsigned_abs();
    let ub = b.unsigned_abs();
    let g = gcd_u128_stein(ua, ub);
    let q = ua / g;
    if let Some(prod) = q.checked_mul(ub) {
        if prod <= i128::MAX as u128 {
            return Ok(CalcValue::normalize_decimal(prod as i128, 0));
        }
    }
    CalcValue::normalize_decimal(q as i128, 0).mul(&CalcValue::normalize_decimal(ub as i128, 0))
}

fn lcm_i64(a: i64, b: i64) -> Result<i64, String> {
    if a == 0 || b == 0 { return Ok(0); }
    let g = gcd_i64(a, b);
    let (part, ov1) = (a.abs() / g).overflowing_mul(b.abs());
    if ov1 {
        Err("LCM integer overflow".to_string())
    } else {
        Ok(part)
    }
}

fn ncr_i128(n: i128, r: i128) -> Result<CalcValue, String> {
    if r < 0 || r > n { return Ok(CalcValue::FastInt(0)); }
    if r == 0 || r == n { return Ok(CalcValue::FastInt(1)); }
    let k = r.min(n - r);
    let mut res = 1u128;
    let un = n as u128;
    let uk = k as u128;
    for i in 1..=uk {
        let num = un - i + 1;
        let g = gcd_u128_stein(num, i);
        let div_res = res / (i / g);
        res = div_res.checked_mul(num / g).ok_or("nCr overflow")?;
    }
    if res <= i128::MAX as u128 {
        Ok(CalcValue::normalize_decimal(res as i128, 0))
    } else {
        Err("nCr overflow".to_string())
    }
}

fn npr_i128(n: i128, r: i128) -> Result<CalcValue, String> {
    if r < 0 || r > n { return Ok(CalcValue::FastInt(0)); }
    if r == 0 { return Ok(CalcValue::FastInt(1)); }
    let mut res = 1u128;
    let un = n as u128;
    for i in 0..(r as u128) {
        res = res.checked_mul(un - i).ok_or("nPr overflow")?;
    }
    if res <= i128::MAX as u128 {
        Ok(CalcValue::normalize_decimal(res as i128, 0))
    } else {
        Err("nPr overflow".to_string())
    }
}

/// Divide-and-conquer binary-splitting factorial (unlocks sub-quadratic Karatsuba/NTT & Rayon multi-core multiplication)
fn fact_prod_range(low: u64, high: u64) -> Result<CalcValue, String> {
    check_task_aborted()?;
    if low > high {
        return Ok(CalcValue::FastInt(1));
    }
    if low == high {
        return Ok(CalcValue::FastInt(low as i64));
    }
    if high - low == 1 {
        if let Some(p) = (low as i64).checked_mul(high as i64) {
            return Ok(CalcValue::FastInt(p));
        }
        return CalcValue::FastInt(low as i64).mul(&CalcValue::FastInt(high as i64));
    }
    // Pack up to 4 small factors into a single u64 before multiplying
    if high - low <= 3 && high <= 100_000 {
        let mut acc = 1u64;
        for x in low..=high {
            acc *= x;
        }
        if acc <= i64::MAX as u64 {
            return Ok(CalcValue::FastInt(acc as i64));
        }
    }
    let mid = low + (high - low) / 2;
    if high - low >= 2048 {
        let ctx = current_task_cancel_ctx();
        let (left_res, right_res) = rayon::join(
            || {
                let _g = EpochGuard::from_ctx(ctx);
                fact_prod_range(low, mid)
            },
            || {
                let _g = EpochGuard::from_ctx(ctx);
                fact_prod_range(mid + 1, high)
            },
        );
        let left = left_res?;
        let right = right_res?;
        left.mul(&right)
    } else {
        let left = fact_prod_range(low, mid)?;
        let right = fact_prod_range(mid + 1, high)?;
        left.mul(&right)
    }
}

fn fact_i64(n: i64) -> Result<CalcValue, String> {
    if n < 0 { return Err("Factorial of negative integer".to_string()); }
    if n <= 20 {
        let mut acc = 1u64;
        for i in 2..=n as u64 {
            acc *= i;
        }
        return Ok(CalcValue::FastInt(acc as i64));
    }
    let max_n = if is_deep_compute_active() { 100_000 } else { 10_000 };
    if n <= max_n {
        return fact_prod_range(2, n as u64);
    }
    Err("Factorial operand exceeds limit (n <= 10000)".to_string())
}

fn fibonacci(n: u64) -> Result<CalcValue, String> {
    let max_n = if is_deep_compute_active() { 1_000_000 } else { 20_000 };
    if n > max_n {
        return Err("Fibonacci index exceeds limit (n <= 20000)".to_string());
    }
    if n == 0 { return Ok(CalcValue::FastInt(0)); }
    if n <= 2 { return Ok(CalcValue::FastInt(1)); }

    fn fast_doubling(n: u64) -> Result<(CalcValue, CalcValue), String> {
        check_task_aborted()?;
        if n == 0 {
            return Ok((CalcValue::FastInt(0), CalcValue::FastInt(1)));
        }
        let (a, b) = fast_doubling(n >> 1)?;
        let two_b = b.add(&b)?;
        let diff = two_b.sub(&a)?;
        if n >= 8192 {
            let ctx = current_task_cancel_ctx();
            let (c_res, (a_sq_res, b_sq_res)) = rayon::join(
                || {
                    let _g = EpochGuard::from_ctx(ctx);
                    a.mul(&diff)
                },
                || {
                    rayon::join(
                        || {
                            let _g = EpochGuard::from_ctx(ctx);
                            a.mul(&a)
                        },
                        || {
                            let _g = EpochGuard::from_ctx(ctx);
                            b.mul(&b)
                        },
                    )
                },
            );
            let c = c_res?;
            let d = a_sq_res?.add(&b_sq_res?)?;
            if n & 1 == 0 {
                Ok((c, d))
            } else {
                let c_plus_d = c.add(&d)?;
                Ok((d, c_plus_d))
            }
        } else {
            let c = a.mul(&diff)?;
            let a_sq = a.mul(&a)?;
            let b_sq = b.mul(&b)?;
            let d = a_sq.add(&b_sq)?;

            if n & 1 == 0 {
                Ok((c, d))
            } else {
                let c_plus_d = c.add(&d)?;
                Ok((d, c_plus_d))
            }
        }
    }

    Ok(fast_doubling(n)?.0)
}

fn gamma_approx(x: f64) -> f64 {
    let p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_approx(1.0 - x))
    } else {
        let z = x - 1.0;
        let mut a = p[0];
        let t = z + 7.5;
        for (i, &val) in p.iter().enumerate().skip(1) {
            a += val / (z + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
    }
}

pub fn execute_with_telemetry(raw_input: &str, vars: &mut FxHashMap<String, CalcValue>) -> Result<EvalResult, String> {
    execute_with_telemetry_epoch(raw_input, vars, 0)
}

pub fn execute_settled_with_telemetry(
    raw_input: &str,
    vars: &mut FxHashMap<String, CalcValue>,
) -> Result<EvalResult, String> {
    let _deep = DeepComputeGuard::new(true);
    execute_with_telemetry_epoch(raw_input, vars, 0)
}

pub fn execute_with_telemetry_epoch(raw_input: &str, vars: &mut FxHashMap<String, CalcValue>, epoch: u64) -> Result<EvalResult, String> {
    execute_with_telemetry_with_counter(raw_input, vars, epoch, &CANCELLATION_EPOCH)
}

pub fn execute_with_telemetry_with_counter(
    raw_input: &str,
    vars: &mut FxHashMap<String, CalcValue>,
    epoch: u64,
    counter: &AtomicU64,
) -> Result<EvalResult, String> {
    execute_with_telemetry_with_counter_mode(raw_input, vars, epoch, counter, is_deep_compute_active())
}

pub fn execute_with_telemetry_with_counter_mode(
    raw_input: &str,
    vars: &mut FxHashMap<String, CalcValue>,
    epoch: u64,
    counter: &AtomicU64,
    deep_mode: bool,
) -> Result<EvalResult, String> {
    let _epoch_guard = EpochGuard::with_counter_and_mode(epoch, counter, deep_mode);
    check_task_aborted()?;

    let t0 = Instant::now();

    let (tokens, _) = super::lexer::Lexer::tokenize(raw_input);
    let mut parser = super::parser::Parser::new(&tokens);
    let ast = parser.parse()?;

    check_task_aborted()?;
    let mut evaluator = Evaluator::new(vars, epoch);
    let val = evaluator.evaluate_expr(&ast)?;
    check_task_aborted()?;

    let elapsed = t0.elapsed().as_nanos();
    let canonical = ast.to_canonical_string();
    check_task_aborted()?;
    let trimmed = raw_input.trim_end();
    Ok(EvalResult {
        value: val,
        elapsed_nanos: elapsed,
        canonical_echo: canonical,
        is_speculative: trimmed.ends_with('+') || trimmed.ends_with('-') || trimmed.ends_with('*') || trimmed.ends_with('/') || trimmed.ends_with('^'),
    })
}

// =========================================================================
// SOTA Pre-Execution AST Cost Estimator & Input-Wall Tier Classifier
// =========================================================================

pub const FAST_PATH_MAX_BITS: u64 = 4_096;      // 64 limbs (~1,233 decimal digits)
pub const FAST_PATH_MAX_NANOS: u64 = 150_000;   // 150 µs synchronous UI budget
pub const MAX_POLY_DEGREE: usize = 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExecTier {
    /// Guaranteed < 150 µs and <= 4,096 bits: executes synchronously on the UI thread with 0 delay
    InstantFastPath = 0,
    /// Multi-millisecond or large-limb calculation (4,096..=524,288 bits, large factorials/fib/primes):
    /// Strictly offloaded to LiveCalcWorker behind the adaptive Input Wall & cooperative preemption
    GuardedAsyncWorker = 1,
    /// Exceeds exact bignum ceiling (> 524,288 bits) or function domain limits:
    /// Pre-decided before starting so 0 CPU cycles or heap allocations are wasted
    ExceedsExactLimit = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstKind {
    ExactInt,
    ScalarFloat,
    Matrix(usize),
    Polynomial(usize),
    Logic,
}

#[derive(Clone, Debug)]
pub struct AstCostEstimate {
    pub tier: ExecTier,
    pub kind: EstKind,
    pub est_bits: u64,
    pub est_nanos: u64,
    pub approx_val: f64,
    pub limit_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub enum InputClassification {
    SyntaxError(String),
    Classified(AstCostEstimate),
}

pub fn classify_input_cost(raw_input: &str, vars: &FxHashMap<String, CalcValue>) -> InputClassification {
    let (tokens, _) = super::lexer::Lexer::tokenize(raw_input);
    let mut parser = super::parser::Parser::new(&tokens);
    match parser.parse() {
        Ok(ast) => {
            let mut cost = estimate_expr_cost(&ast, vars);
            // Account for PreparedDisplay::from_value running Pollard's Rho prime factorization
            // on 45..=60-bit integers
            if cost.tier == ExecTier::InstantFastPath
                && cost.kind == EstKind::ExactInt
                && cost.est_bits >= 45
                && cost.est_bits <= 60
            {
                let abs_v = cost.approx_val.abs();
                if abs_v >= 1.75e13 && abs_v <= (1u64 << 60) as f64 {
                    cost.est_nanos = cost.est_nanos.saturating_add(180_000);
                    cost.tier = ExecTier::GuardedAsyncWorker;
                }
            }
            InputClassification::Classified(cost)
        }
        Err(e) => InputClassification::SyntaxError(e),
    }
}

fn estimate_value_cost(val: &CalcValue) -> AstCostEstimate {
    let approx_val = val.to_f64();
    let est_bits = val.bit_length_estimate();
    let kind = match val {
        CalcValue::FastInt(_)
        | CalcValue::InlineBigInt { .. }
        | CalcValue::HeapBigInt { .. }
        | CalcValue::Factored { .. }
        | CalcValue::ExactDecimal { scale: 0, .. } => EstKind::ExactInt,
        CalcValue::Matrix { rows, cols, .. } => EstKind::Matrix((*rows).max(*cols)),
        CalcValue::Polynomial { coeffs } => EstKind::Polynomial(coeffs.len().saturating_sub(1)),
        CalcValue::Logic { .. } => EstKind::Logic,
        _ => EstKind::ScalarFloat,
    };
    let est_nanos = if est_bits <= 256 { 20 } else { ((est_bits / 64) * 15).max(50) };
    let tier = if kind == EstKind::ExactInt && est_bits > MAX_EXACT_BIGNUM_BITS {
        ExecTier::ExceedsExactLimit
    } else if (kind == EstKind::ExactInt && est_bits > FAST_PATH_MAX_BITS) || est_nanos > FAST_PATH_MAX_NANOS {
        ExecTier::GuardedAsyncWorker
    } else {
        ExecTier::InstantFastPath
    };
    AstCostEstimate {
        tier,
        kind,
        est_bits,
        est_nanos,
        approx_val,
        limit_reason: if tier == ExecTier::ExceedsExactLimit {
            Some(format!("Literal exceeds exact bignum limit (max {} bits)", MAX_EXACT_BIGNUM_BITS))
        } else {
            None
        },
    }
}

pub fn estimate_expr_cost(expr: &Expr, vars: &FxHashMap<String, CalcValue>) -> AstCostEstimate {
    match expr {
        Expr::Literal(val) => estimate_value_cost(val),
        Expr::Var(name) => {
            if let Some(val) = vars.get(name) {
                return estimate_value_cost(val);
            }
            let lower = name.to_lowercase();
            let (kind, approx_val) = match lower.as_str() {
                "pi" | "π" => (EstKind::ScalarFloat, std::f64::consts::PI),
                "e" => (EstKind::ScalarFloat, std::f64::consts::E),
                "tau" | "τ" => (EstKind::ScalarFloat, std::f64::consts::TAU),
                "phi" | "φ" => (EstKind::ScalarFloat, 1.618033988749895),
                "c" => (EstKind::ExactInt, 299_792_458.0),
                "g" => (EstKind::ScalarFloat, 9.80665),
                "x" => (EstKind::Polynomial(1), 1.0),
                "i" | "j" => (EstKind::ScalarFloat, 1.0),
                "p" | "q" | "r" | "s" | "t" | "u" | "true" | "false" | "top" | "bot" => (EstKind::Logic, 1.0),
                _ => (EstKind::ScalarFloat, 1.0),
            };
            AstCostEstimate {
                tier: ExecTier::InstantFastPath,
                kind,
                est_bits: 32,
                est_nanos: 25,
                approx_val,
                limit_reason: None,
            }
        }
        Expr::Unary(op, inner) => {
            let mut sub = estimate_expr_cost(inner, vars);
            if matches!(op, UnaryOp::Neg) {
                sub.approx_val = -sub.approx_val;
            } else if matches!(op, UnaryOp::LogicNot) {
                sub.kind = EstKind::Logic;
            }
            sub.est_nanos = sub.est_nanos.saturating_add(20);
            sub
        }
        Expr::Binary(op, left, right) => {
            let l = estimate_expr_cost(left, vars);
            if l.tier == ExecTier::ExceedsExactLimit {
                return l;
            }
            let r = estimate_expr_cost(right, vars);
            if r.tier == ExecTier::ExceedsExactLimit {
                return r;
            }

            let est_bits: u64;
            let op_nanos: u64;
            let mut approx_val = 0.0f64;
            let mut kind = if l.kind == EstKind::ExactInt && r.kind == EstKind::ExactInt {
                EstKind::ExactInt
            } else {
                EstKind::ScalarFloat
            };

            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    if let (EstKind::Matrix(d1), EstKind::Matrix(d2)) = (l.kind, r.kind) {
                        let dim = d1.max(d2);
                        kind = EstKind::Matrix(dim);
                        est_bits = (dim * dim * 64) as u64;
                        op_nanos = (dim * dim * 4) as u64;
                    } else if matches!(l.kind, EstKind::Polynomial(_)) || matches!(r.kind, EstKind::Polynomial(_)) {
                        let deg1 = match l.kind { EstKind::Polynomial(d) => d, _ => 0 };
                        let deg2 = match r.kind { EstKind::Polynomial(d) => d, _ => 0 };
                        let deg = deg1.max(deg2);
                        kind = EstKind::Polynomial(deg);
                        est_bits = ((deg + 1) * 64) as u64;
                        op_nanos = ((deg + 1) * 8) as u64;
                    } else if kind == EstKind::ExactInt {
                        est_bits = l.est_bits.max(r.est_bits).saturating_add(1);
                        let limbs = (est_bits + 63) / 64;
                        op_nanos = 25 + limbs.saturating_mul(3);
                        approx_val = if matches!(op, BinaryOp::Add) {
                            l.approx_val + r.approx_val
                        } else {
                            l.approx_val - r.approx_val
                        };
                    } else {
                        est_bits = 64;
                        op_nanos = 25;
                        approx_val = if matches!(op, BinaryOp::Add) {
                            l.approx_val + r.approx_val
                        } else {
                            l.approx_val - r.approx_val
                        };
                    }
                }
                BinaryOp::Mul => {
                    if matches!(l.kind, EstKind::Matrix(_)) || matches!(r.kind, EstKind::Matrix(_)) {
                        let d1 = match l.kind { EstKind::Matrix(d) => d, _ => 1 };
                        let d2 = match r.kind { EstKind::Matrix(d) => d, _ => 1 };
                        let dim = d1.max(d2);
                        kind = EstKind::Matrix(dim);
                        est_bits = (dim * dim * 64) as u64;
                        op_nanos = if matches!((l.kind, r.kind), (EstKind::Matrix(_), EstKind::Matrix(_))) {
                            (dim as u64).pow(3) * 4
                        } else {
                            (dim * dim * 4) as u64
                        };
                        approx_val = 1.0;
                    } else if matches!(l.kind, EstKind::Polynomial(_)) || matches!(r.kind, EstKind::Polynomial(_)) {
                        let deg1 = match l.kind { EstKind::Polynomial(d) => d, _ => 0 };
                        let deg2 = match r.kind { EstKind::Polynomial(d) => d, _ => 0 };
                        let deg = deg1.saturating_add(deg2);
                        if deg > MAX_POLY_DEGREE {
                            return AstCostEstimate {
                                tier: ExecTier::ExceedsExactLimit,
                                kind: EstKind::Polynomial(deg),
                                est_bits: ((deg + 1) * 64) as u64,
                                est_nanos: u64::MAX,
                                approx_val: f64::INFINITY,
                                limit_reason: Some(format!("Polynomial degree exceeds limit ({} > {})", deg, MAX_POLY_DEGREE)),
                            };
                        }
                        kind = EstKind::Polynomial(deg);
                        est_bits = ((deg + 1) * 64) as u64;
                        op_nanos = ((deg1 + 1) * (deg2 + 1) * 5) as u64;
                    } else if kind == EstKind::ExactInt {
                        est_bits = l.est_bits.saturating_add(r.est_bits);
                        if est_bits > MAX_EXACT_BIGNUM_BITS {
                            return AstCostEstimate {
                                tier: ExecTier::ExceedsExactLimit,
                                kind: EstKind::ExactInt,
                                est_bits,
                                est_nanos: u64::MAX,
                                approx_val: f64::INFINITY,
                                limit_reason: Some(format!(
                                    "Multiplication result exceeds exact bignum limit (~{} bits > {} max bits)",
                                    est_bits, MAX_EXACT_BIGNUM_BITS
                                )),
                            };
                        }
                        op_nanos = crate::engine::wisdom::estimate_bignum_work_nanos(l.est_bits, r.est_bits);
                        approx_val = l.approx_val * r.approx_val;
                    } else {
                        est_bits = 64;
                        op_nanos = 30;
                        approx_val = l.approx_val * r.approx_val;
                    }
                }
                BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod => {
                    if let EstKind::Matrix(dim) = l.kind {
                        kind = EstKind::Matrix(dim);
                        est_bits = (dim * dim * 64) as u64;
                        op_nanos = if matches!(r.kind, EstKind::Matrix(_)) {
                            (dim as u64).pow(3) * 12
                        } else {
                            (dim * dim * 4) as u64
                        };
                    } else if let EstKind::Polynomial(deg) = l.kind {
                        kind = EstKind::Polynomial(deg);
                        est_bits = ((deg + 1) * 64) as u64;
                        op_nanos = ((deg + 1) * 20) as u64;
                    } else {
                        if matches!(op, BinaryOp::Div) {
                            kind = EstKind::ScalarFloat;
                        }
                        est_bits = if kind == EstKind::ExactInt { l.est_bits } else { 64 };
                        let limbs = (l.est_bits + 63) / 64;
                        op_nanos = 50 + limbs.saturating_mul(20);
                        approx_val = if r.approx_val != 0.0 { l.approx_val / r.approx_val } else { f64::NAN };
                    }
                }
                BinaryOp::Pow => {
                    if matches!(l.kind, EstKind::Logic) && matches!(r.kind, EstKind::Logic) {
                        kind = EstKind::Logic;
                        est_bits = 64;
                        op_nanos = 25;
                    } else if let EstKind::Matrix(dim) = l.kind {
                        kind = EstKind::Matrix(dim);
                        est_bits = (dim * dim * 64) as u64;
                        let exp_steps = (r.approx_val.abs().max(1.0).log2().ceil() as u64).clamp(1, 16);
                        op_nanos = exp_steps.saturating_mul((dim as u64).pow(3) * 4);
                        approx_val = 1.0;
                    } else if let EstKind::Polynomial(deg) = l.kind {
                        let exp_u = r.approx_val.max(0.0) as usize;
                        let new_deg = deg.saturating_mul(exp_u);
                        if exp_u > 64 || new_deg > MAX_POLY_DEGREE {
                            return AstCostEstimate {
                                tier: ExecTier::ExceedsExactLimit,
                                kind: EstKind::Polynomial(new_deg),
                                est_bits: ((new_deg + 1) * 64) as u64,
                                est_nanos: u64::MAX,
                                approx_val: f64::INFINITY,
                                limit_reason: Some("Polynomial exponent or degree exceeds limit".to_string()),
                            };
                        }
                        kind = EstKind::Polynomial(new_deg);
                        est_bits = ((new_deg + 1) * 64) as u64;
                        op_nanos = ((new_deg + 1) * (new_deg + 1) * 5) as u64;
                    } else {
                        approx_val = l.approx_val.powf(r.approx_val);
                        let is_int_base = l.kind == EstKind::ExactInt
                            || (l.approx_val.is_finite() && l.approx_val.fract() == 0.0);
                        let is_int_exp = r.kind == EstKind::ExactInt
                            || (r.approx_val.is_finite() && r.approx_val.fract() == 0.0)
                            || r.est_bits > 60;

                        if is_int_base && is_int_exp && (r.approx_val >= 0.0 || !r.approx_val.is_finite()) {
                            kind = EstKind::ExactInt;
                            let abs_base = l.approx_val.abs();
                            let is_unit_base = l.est_bits <= 52 && l.approx_val.is_finite() && abs_base <= 1.0;
                            let is_zero_exp = r.est_bits <= 52 && r.approx_val == 0.0;

                            if is_unit_base || is_zero_exp {
                                est_bits = 1;
                                op_nanos = 30;
                            } else {
                                let base_bits_f64 = if l.est_bits > 52 || !l.approx_val.is_finite() {
                                    l.est_bits.max(2) as f64
                                } else {
                                    abs_base.max(2.0).log2()
                                };
                                let exp_f64 = if r.est_bits > 62 || !r.approx_val.is_finite() {
                                    f64::INFINITY
                                } else {
                                    r.approx_val
                                };
                                let out_bits_f64 = exp_f64 * base_bits_f64;
                                if !out_bits_f64.is_finite() || out_bits_f64 > (MAX_EXACT_BIGNUM_BITS as f64) {
                                    let reported_bits = if out_bits_f64.is_finite() {
                                        out_bits_f64.min(u64::MAX as f64) as u64
                                    } else {
                                        u64::MAX
                                    };
                                    return AstCostEstimate {
                                        tier: ExecTier::ExceedsExactLimit,
                                        kind: EstKind::ExactInt,
                                        est_bits: reported_bits,
                                        est_nanos: u64::MAX,
                                        approx_val: f64::INFINITY,
                                        limit_reason: Some(format!(
                                            "Power result exceeds exact bignum limit (~{} bits > {} max bits)",
                                            reported_bits, MAX_EXACT_BIGNUM_BITS
                                        )),
                                    };
                                }
                                est_bits = (out_bits_f64.ceil() as u64).max(1);
                                let base_bits_u64 = (base_bits_f64.ceil() as u64).max(2);
                                op_nanos = crate::engine::wisdom::estimate_pow_work_nanos(base_bits_u64, exp_f64 as u64);
                            }
                        } else {
                            kind = EstKind::ScalarFloat;
                            est_bits = 64;
                            op_nanos = 120;
                        }
                    }
                }
                _ => {
                    kind = EstKind::Logic;
                    est_bits = 64;
                    op_nanos = 40;
                }
            }

            let total_nanos = l.est_nanos.saturating_add(r.est_nanos).saturating_add(op_nanos);
            let tier = if kind == EstKind::ExactInt && est_bits > MAX_EXACT_BIGNUM_BITS {
                ExecTier::ExceedsExactLimit
            } else if l.tier == ExecTier::GuardedAsyncWorker
                || r.tier == ExecTier::GuardedAsyncWorker
                || (kind == EstKind::ExactInt && est_bits > FAST_PATH_MAX_BITS)
                || total_nanos > FAST_PATH_MAX_NANOS
            {
                ExecTier::GuardedAsyncWorker
            } else {
                ExecTier::InstantFastPath
            };

            AstCostEstimate {
                tier,
                kind,
                est_bits,
                est_nanos: total_nanos,
                approx_val,
                limit_reason: None,
            }
        }
        Expr::FnCall(name, args) => {
            let mut total_nanos = 50u64;
            let mut max_tier = ExecTier::InstantFastPath;
            let mut first_arg: Option<AstCostEstimate> = None;

            for (idx, a) in args.iter().enumerate() {
                let c = estimate_expr_cost(a, vars);
                if c.tier == ExecTier::ExceedsExactLimit {
                    return c;
                }
                if c.tier > max_tier {
                    max_tier = c.tier;
                }
                total_nanos = total_nanos.saturating_add(c.est_nanos);
                if idx == 0 {
                    first_arg = Some(c);
                }
            }

            let mut est_bits = first_arg.as_ref().map_or(64, |c| c.est_bits);
            let mut approx_val = first_arg.as_ref().map_or(1.0, |c| c.approx_val);
            let mut kind = first_arg.as_ref().map_or(EstKind::ScalarFloat, |c| c.kind);
            let lower = name.to_lowercase();

            match lower.as_str() {
                "fact" | "fac" | "factorial" => {
                    let n = approx_val;
                    if !n.is_finite() || n > 10_000.0 {
                        return AstCostEstimate {
                            tier: ExecTier::ExceedsExactLimit,
                            kind: EstKind::ExactInt,
                            est_bits: MAX_EXACT_BIGNUM_BITS + 1,
                            est_nanos: u64::MAX,
                            approx_val: f64::INFINITY,
                            limit_reason: Some("Factorial operand exceeds limit (n <= 10000)".to_string()),
                        };
                    }
                    kind = EstKind::ExactInt;
                    if n <= 20.0 && n >= 0.0 {
                        let mut acc = 1u64;
                        for i in 2..=(n as u64) {
                            acc = acc.saturating_mul(i);
                        }
                        approx_val = acc as f64;
                        est_bits = (64 - acc.leading_zeros() as u64).max(1);
                    } else if n > 20.0 {
                        // Stirling approximation for log2(n!): n * log2(n / e)
                        let bits = (n * (n / std::f64::consts::E).max(1.0).log2()).ceil() as u64;
                        est_bits = bits.max(64);
                        let fmt_cost = crate::engine::wisdom::estimate_bignum_work_nanos(est_bits / 2, est_bits / 2);
                        total_nanos = total_nanos.saturating_add(fmt_cost);
                        approx_val = if n <= 170.0 { gamma_approx(n + 1.0) } else { f64::INFINITY };
                    }
                }
                "fib" | "fibonacci" => {
                    let n = approx_val;
                    if !n.is_finite() || n > 20_000.0 {
                        return AstCostEstimate {
                            tier: ExecTier::ExceedsExactLimit,
                            kind: EstKind::ExactInt,
                            est_bits: MAX_EXACT_BIGNUM_BITS + 1,
                            est_nanos: u64::MAX,
                            approx_val: f64::INFINITY,
                            limit_reason: Some("Fibonacci index exceeds limit (n <= 20000)".to_string()),
                        };
                    }
                    kind = EstKind::ExactInt;
                    if n >= 2.0 {
                        est_bits = ((n * 0.6942419).ceil() as u64).max(1);
                        approx_val = if n <= 1470.0 {
                            (1.618033988749895_f64.powf(n) / 2.23606797749979).round()
                        } else {
                            f64::INFINITY
                        };
                        if n > 90.0 {
                            let fmt_cost = crate::engine::wisdom::estimate_bignum_work_nanos(est_bits / 2, est_bits / 2);
                            total_nanos = total_nanos.saturating_add(fmt_cost);
                        }
                    } else {
                        est_bits = 1;
                        approx_val = n.max(0.0);
                    }
                }
                "sq" | "square" => {
                    if let Some(ref c) = first_arg {
                        match c.kind {
                            EstKind::ExactInt => {
                                est_bits = c.est_bits.saturating_mul(2);
                                if est_bits > MAX_EXACT_BIGNUM_BITS {
                                    return AstCostEstimate {
                                        tier: ExecTier::ExceedsExactLimit,
                                        kind: EstKind::ExactInt,
                                        est_bits,
                                        est_nanos: u64::MAX,
                                        approx_val: f64::INFINITY,
                                        limit_reason: Some(format!(
                                            "Square result exceeds exact bignum limit (~{} bits > {} max bits)",
                                            est_bits, MAX_EXACT_BIGNUM_BITS
                                        )),
                                    };
                                }
                                total_nanos = total_nanos.saturating_add(
                                    crate::engine::wisdom::estimate_bignum_work_nanos(c.est_bits, c.est_bits),
                                );
                                approx_val = c.approx_val * c.approx_val;
                            }
                            EstKind::Matrix(dim) => {
                                total_nanos = total_nanos.saturating_add((dim as u64).pow(3) * 4);
                            }
                            _ => {
                                approx_val = c.approx_val * c.approx_val;
                            }
                        }
                    }
                }
                "cb" | "cube" => {
                    if let Some(ref c) = first_arg {
                        match c.kind {
                            EstKind::ExactInt => {
                                est_bits = c.est_bits.saturating_mul(3);
                                if est_bits > MAX_EXACT_BIGNUM_BITS {
                                    return AstCostEstimate {
                                        tier: ExecTier::ExceedsExactLimit,
                                        kind: EstKind::ExactInt,
                                        est_bits,
                                        est_nanos: u64::MAX,
                                        approx_val: f64::INFINITY,
                                        limit_reason: Some(format!(
                                            "Cube result exceeds exact bignum limit (~{} bits > {} max bits)",
                                            est_bits, MAX_EXACT_BIGNUM_BITS
                                        )),
                                    };
                                }
                                let sq_cost = crate::engine::wisdom::estimate_bignum_work_nanos(c.est_bits, c.est_bits);
                                let cb_cost = crate::engine::wisdom::estimate_bignum_work_nanos(c.est_bits * 2, c.est_bits);
                                total_nanos = total_nanos.saturating_add(sq_cost).saturating_add(cb_cost);
                                approx_val = c.approx_val.powi(3);
                            }
                            EstKind::Matrix(dim) => {
                                total_nanos = total_nanos.saturating_add((dim as u64).pow(3) * 8);
                            }
                            _ => {
                                approx_val = c.approx_val.powi(3);
                            }
                        }
                    }
                }
                "primes" | "prime" | "factors" | "factor" => {
                    kind = EstKind::ExactInt;
                    let n = approx_val.abs();
                    if n > 1e12 || est_bits >= 40 {
                        // Pollard's rho on > 40-bit numbers can take 200 µs .. 3 ms
                        total_nanos = total_nanos.saturating_add(250_000);
                    }
                }
                "integrate" | "int" => {
                    // 128 quadrature evaluations of the integrand
                    let integrand_ns = first_arg.as_ref().map_or(100, |c| c.est_nanos);
                    total_nanos = total_nanos.saturating_add(integrand_ns.saturating_mul(128));
                    if args.len() >= 3 {
                        kind = EstKind::ScalarFloat;
                        est_bits = 64;
                    }
                }
                "deriv" | "diff" => {
                    let integrand_ns = first_arg.as_ref().map_or(100, |c| c.est_nanos);
                    total_nanos = total_nanos.saturating_add(integrand_ns.saturating_mul(9));
                    if args.len() >= 2 {
                        kind = EstKind::ScalarFloat;
                        est_bits = 64;
                    }
                }
                "eye" | "identity" => {
                    let dim = (approx_val.max(1.0) as usize).min(100);
                    kind = EstKind::Matrix(dim);
                    est_bits = (dim * dim * 64) as u64;
                    total_nanos = total_nanos.saturating_add((dim * dim * 5) as u64);
                }
                "det" | "rank" | "norm" | "frobenius" | "trace" | "tr" => {
                    if let Some(EstKind::Matrix(dim)) = first_arg.as_ref().map(|c| c.kind) {
                        let d = dim as u64;
                        total_nanos = total_nanos.saturating_add(d * d * d * 10);
                    }
                    kind = if lower == "rank" { EstKind::ExactInt } else { EstKind::ScalarFloat };
                    est_bits = 64;
                }
                "inv" | "inverse" | "transpose" | "t" => {
                    if let Some(EstKind::Matrix(dim)) = first_arg.as_ref().map(|c| c.kind) {
                        let d = dim as u64;
                        if lower.starts_with("inv") {
                            total_nanos = total_nanos.saturating_add(d * d * d * 12);
                        } else {
                            total_nanos = total_nanos.saturating_add(d * d * 4);
                        }
                        kind = EstKind::Matrix(dim);
                        est_bits = (dim * dim * 64) as u64;
                    }
                }
                "eig" | "eigenvalues" => {
                    if let Some(EstKind::Matrix(dim)) = first_arg.as_ref().map(|c| c.kind) {
                        let d = dim as u64;
                        total_nanos = total_nanos.saturating_add(d * d * d * 45 + 100 * d * d);
                        kind = EstKind::Matrix(dim);
                        est_bits = (dim * 64) as u64;
                    }
                }
                "solve" | "roots" | "poly_roots" => {
                    if let Some(ref c) = first_arg {
                        match c.kind {
                            EstKind::Matrix(dim) => {
                                let d = dim as u64;
                                total_nanos = total_nanos.saturating_add(d * d * d * 12);
                                kind = EstKind::Matrix(dim);
                            }
                            EstKind::Polynomial(deg) => {
                                let d = deg as u64;
                                total_nanos = total_nanos.saturating_add(120 * d * d * 15);
                                kind = EstKind::Matrix(deg);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    kind = EstKind::ScalarFloat;
                    est_bits = 64;
                }
            }

            if kind == EstKind::ExactInt && est_bits > MAX_EXACT_BIGNUM_BITS {
                max_tier = ExecTier::ExceedsExactLimit;
            } else if (kind == EstKind::ExactInt && est_bits > FAST_PATH_MAX_BITS) || total_nanos > FAST_PATH_MAX_NANOS {
                max_tier = max_tier.max(ExecTier::GuardedAsyncWorker);
            }

            AstCostEstimate {
                tier: max_tier,
                kind,
                est_bits,
                est_nanos: total_nanos,
                approx_val,
                limit_reason: None,
            }
        }
        Expr::Matrix(rows, cols, elems) => {
            let mut total_nanos = 100u64;
            let mut max_tier = ExecTier::InstantFastPath;
            for e in elems {
                let c = estimate_expr_cost(e, vars);
                if c.tier == ExecTier::ExceedsExactLimit {
                    return c;
                }
                if c.tier > max_tier {
                    max_tier = c.tier;
                }
                total_nanos = total_nanos.saturating_add(c.est_nanos);
            }
            let dim = (*rows).max(*cols);
            let d = dim as u64;
            total_nanos = total_nanos.saturating_add(d * d * d * 10);
            if total_nanos > FAST_PATH_MAX_NANOS {
                max_tier = max_tier.max(ExecTier::GuardedAsyncWorker);
            }
            AstCostEstimate {
                tier: max_tier,
                kind: EstKind::Matrix(dim),
                est_bits: (elems.len() as u64) * 64,
                est_nanos: total_nanos,
                approx_val: 1.0,
                limit_reason: None,
            }
        }
    }
}
