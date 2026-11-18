#[path = "../engine/mod.rs"]
mod engine;
#[path = "../config.rs"]
mod config;

fn main() {
    let _ = config::load_or_create_ini();
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "{} ({}) - SOTA Diagnostic & Benchmark Suite",
            env!("APP_DISPLAY_NAME"),
            env!("APP_PUBLISHER")
        );
        println!();
        println!("USAGE:");
        println!("    calc-diag.exe [OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("    --filter <category>    Run only tests matching the specified category");
        println!("                           (e.g., Arithmetic, Punctuation, Matrix, Logic, Bignum, LaTeX)");
        println!("    --expr <expression>    Evaluate a single custom expression with microsecond latency breakdown");
        println!("    --bench <N>            Run N benchmark passes across the entire suite for statistical telemetry");
        println!("    --json                 Output full report in structured JSON format for automated tooling");
        println!("    -h, --help             Show this help information");
        return;
    }

    // Custom expression test mode
    if let Some(pos) = args.iter().position(|a| a == "--expr") {
        if let Some(expr) = args.get(pos + 1) {
            println!("=== Evaluating Custom Expression: '{}' ===", expr);
            match engine::diagnostics::evaluate_custom_expression(expr) {
                Ok((val, echo, nanos)) => {
                    println!("Status   : SUCCESS");
                    println!("Result   : {}", val);
                    if !echo.is_empty() {
                        println!("Echo     : {}", echo);
                    }
                    let lat = if nanos < 1_000 {
                        format!("{} ns", nanos)
                    } else if nanos < 1_000_000 {
                        format!("{:.2} µs", nanos as f64 / 1_000.0)
                    } else {
                        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
                    };
                    println!("Latency  : {}", lat);
                }
                Err(e) => {
                    eprintln!("Status   : FAILED");
                    eprintln!("Error    : {}", e);
                    std::process::exit(1);
                }
            }
            return;
        }
    }

    // Benchmark mode
    if let Some(pos) = args.iter().position(|a| a == "--bench") {
        let passes: usize = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
        println!("Running {} benchmark passes across all test cases...", passes);
        let mut total_durations = Vec::with_capacity(passes);
        let mut final_report = None;
        for _ in 0..passes {
            let r = engine::diagnostics::run_all_diagnostics();
            total_durations.push((r.total_duration_ms * 1_000.0) as u128); // in µs
            final_report = Some(r);
        }
        total_durations.sort_unstable();
        let p50 = total_durations[passes / 2];
        let p99 = total_durations[(passes * 99 / 100).min(passes - 1)];
        let min = total_durations[0];
        let max = *total_durations.last().unwrap();
        println!("Benchmark Results (Suite duration per pass):");
        println!("  Passes: {}", passes);
        println!("  Min   : {} µs ({:.3} ms)", min, min as f64 / 1000.0);
        println!("  p50   : {} µs ({:.3} ms)", p50, p50 as f64 / 1000.0);
        println!("  p99   : {} µs ({:.3} ms)", p99, p99 as f64 / 1000.0);
        println!("  Max   : {} µs ({:.3} ms)", max, max as f64 / 1000.0);
        if let Some(rep) = final_report {
            if rep.total_failed > 0 {
                std::process::exit(1);
            }
        }
        return;
    }

    // Filter mode
    let filter = if let Some(pos) = args.iter().position(|a| a == "--filter") {
        args.get(pos + 1).map(|s| s.as_str())
    } else {
        None
    };

    let report = engine::diagnostics::run_diagnostics_filtered(filter);

    if args.iter().any(|a| a == "--json") {
        if let Ok(j) = serde_json::to_string_pretty(&report) {
            println!("{}", j);
        }
    } else {
        engine::diagnostics::print_diagnostics_report(&report);
    }

    if report.total_failed > 0 {
        std::process::exit(1);
    }
}
