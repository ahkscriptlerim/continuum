#![allow(dead_code, unused_imports)]

pub mod value;
pub mod bignum;
pub mod matrix;
pub mod logic;
pub mod lexer;
pub mod parser;
pub mod evaluator;
pub mod wisdom;
pub mod diagnostics;
pub mod units;
pub mod gpu;

pub use value::{CalcValue, MAX_EXACT_BIGNUM_BITS, MAX_DEEP_BIGNUM_BITS};
pub use evaluator::{
    execute_with_telemetry, execute_with_telemetry_epoch, execute_with_telemetry_with_counter,
    execute_with_telemetry_with_counter_mode, execute_settled_with_telemetry, EvalResult,
    classify_input_cost, estimate_expr_cost, AstCostEstimate, EstKind, ExecTier, InputClassification,
    is_task_aborted, check_task_aborted, is_deep_compute_active, current_task_cancel_ctx,
    TaskCancelCtx, EpochGuard, DeepComputeGuard, CANCELLATION_EPOCH,
};
pub use wisdom::{HardwareProber, trigger_afk_calibration};
pub use diagnostics::{run_all_diagnostics, print_diagnostics_report, DiagnosticsReport};
pub use units::convert_units;
pub use gpu::GpuComputeEngine;
