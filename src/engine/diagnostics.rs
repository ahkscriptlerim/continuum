use std::time::Instant;
use rustc_hash::FxHashMap;
use super::{execute_with_telemetry, CalcValue, EvalResult};

pub struct TestCase {
    pub category: &'static str,
    pub expression: &'static str,
    pub expected: &'static str,
    pub is_substring: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TestResult {
    pub category: &'static str,
    pub expression: &'static str,
    pub passed: bool,
    pub actual: String,
    pub expected: &'static str,
    pub elapsed_nanos: u128,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct DiagnosticsReport {
    pub results: Vec<TestResult>,
    pub total_passed: usize,
    pub total_failed: usize,
    pub min_nanos: u128,
    pub mean_nanos: u128,
    pub p50_nanos: u128,
    pub p90_nanos: u128,
    pub p99_nanos: u128,
    pub max_nanos: u128,
    pub total_duration_ms: f64,
}

pub static TEST_CASES: &[TestCase] = &[
    // 1. Basic Arithmetic & Precedence
    TestCase { category: "Arithmetic", expression: "2 + 3 * 4", expected: "14", is_substring: false },
    TestCase { category: "Arithmetic", expression: "(2 + 3) * 4", expected: "20", is_substring: false },
    TestCase { category: "Arithmetic", expression: "10 / 2", expected: "5", is_substring: false },
    TestCase { category: "Arithmetic", expression: "10 // 3", expected: "3", is_substring: false },
    TestCase { category: "Arithmetic", expression: "10 % 3", expected: "1", is_substring: false },
    TestCase { category: "Arithmetic", expression: "2^3^2", expected: "512", is_substring: false },

    // 2. Dual Number Punctuation (Continental vs Anglo-American)
    TestCase { category: "Punctuation", expression: "1.000,0 + 5", expected: "1005", is_substring: false },
    TestCase { category: "Punctuation", expression: "1,000.0 + 5", expected: "1005", is_substring: false },
    TestCase { category: "Punctuation", expression: "1.000.000,5 * 2", expected: "2000001", is_substring: false },
    TestCase { category: "Punctuation", expression: "1,000,000.5 * 2", expected: "2000001", is_substring: false },
    TestCase { category: "Punctuation", expression: "1,5 * 4", expected: "6", is_substring: false },
    TestCase { category: "Punctuation", expression: "0,25 + 0,75", expected: "1", is_substring: false },
    TestCase { category: "Punctuation", expression: "3,14159 * 2", expected: "6.28318", is_substring: true },

    // 3. Deterministic 'x' Disambiguation Matrix
    TestCase { category: "Disambiguation", expression: "3x5", expected: "15", is_substring: false },
    TestCase { category: "Disambiguation", expression: "3 x 5", expected: "15", is_substring: false },
    TestCase { category: "Disambiguation", expression: "(2) x (4)", expected: "8", is_substring: false },
    TestCase { category: "Disambiguation", expression: "0xFF", expected: "255", is_substring: false },
    TestCase { category: "Disambiguation", expression: "0b1010", expected: "10", is_substring: false },

    // 4. Implied Multiplication & Consecutive Signs
    TestCase { category: "Implied", expression: "3(5)", expected: "15", is_substring: false },
    TestCase { category: "Implied", expression: "(2)(3)(4)", expected: "24", is_substring: false },
    TestCase { category: "Implied", expression: "3^2(100)", expected: "900", is_substring: false },
    TestCase { category: "Implied", expression: "4sqrt(16)", expected: "16", is_substring: false },
    TestCase { category: "Arithmetic", expression: "1 + 1 = 2", expected: "1", is_substring: false },
    TestCase { category: "Arithmetic", expression: "2 = 3", expected: "0", is_substring: false },
    TestCase { category: "Signs", expression: "3 + - 5", expected: "-2", is_substring: false },
    TestCase { category: "Signs", expression: "3 - - 5", expected: "8", is_substring: false },
    TestCase { category: "Signs", expression: "3 ++ 5", expected: "8", is_substring: false },
    TestCase { category: "Signs", expression: "3 -- 5", expected: "8", is_substring: false },

    // 5. LaTeX Desugaring & KaTeX Environments
    TestCase { category: "LaTeX", expression: r"\frac{10}{2}", expected: "5", is_substring: false },
    TestCase { category: "LaTeX", expression: r"\sqrt{64}", expected: "8", is_substring: false },
    TestCase { category: "LaTeX", expression: r"\sqrt[3]{27}", expected: "3", is_substring: false },
    TestCase { category: "LaTeX", expression: r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} * 2", expected: "[2, 4; 6, 8]", is_substring: false },
    TestCase { category: "LaTeX", expression: r"\begin{vmatrix} 4 & 7 \\ 2 & 6 \end{vmatrix}", expected: "10", is_substring: false },
    TestCase { category: "LaTeX", expression: r"p \lor \neg p", expected: "Tautology", is_substring: true },
    TestCase { category: "LaTeX", expression: r"p \land \neg p", expected: "Contradiction", is_substring: true },

    // 6. Bignums & Powers
    TestCase { category: "Bignum", expression: "2^10", expected: "1024", is_substring: false },
    TestCase { category: "Bignum", expression: "2**32", expected: "4294967296", is_substring: false },
    TestCase { category: "Bignum", expression: "2**100", expected: "1267650600228229401496703205376", is_substring: false },
    TestCase { category: "Bignum", expression: "1232133 * 2131231231231313322", expected: "2625960330630731777375826", is_substring: false },

    // 7. Propositional Logic & Shorthands (pvq, p v q, p')
    TestCase { category: "Logic", expression: "p | !p", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "p & !p", expected: "Contradiction", is_substring: true },
    TestCase { category: "Logic", expression: "pvq", expected: "Contingent", is_substring: true },
    TestCase { category: "Logic", expression: "p v q", expected: "Contingent", is_substring: true },
    TestCase { category: "Logic", expression: "p' | p", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "(p & (p -> q)) -> q", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "p <-> p", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "p = p", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "p + !p", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "p * !p", expected: "Contradiction", is_substring: true },
    TestCase { category: "Logic", expression: "'p = p'", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "'p = p", expected: "Contradiction", is_substring: true },
    TestCase { category: "Logic", expression: "\"p = p\"", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: r"\top \lor p", expected: "Tautology", is_substring: true },

    // 8. Linear Algebra & Matrices
    TestCase { category: "Matrix", expression: "[1, 2; 3, 4] * [2, 0; 1, 2]", expected: "[4, 4; 10, 8]", is_substring: false },
    TestCase { category: "Matrix", expression: "2 * [1, 2; 3, 4]", expected: "[2, 4; 6, 8]", is_substring: false },
    TestCase { category: "Matrix", expression: "[1, 2; 3, 4] * 3", expected: "[3, 6; 9, 12]", is_substring: false },
    TestCase { category: "Matrix", expression: "[1, 2; 3, 4] + [2, 1; 0, 1]", expected: "[3, 3; 3, 5]", is_substring: false },
    TestCase { category: "Matrix", expression: "[1, 2; 3, 4] - [1, 1; 1, 1]", expected: "[0, 1; 2, 3]", is_substring: false },
    TestCase { category: "Matrix", expression: "det([1, 2; 3, 4])", expected: "-2", is_substring: false },
    TestCase { category: "Matrix", expression: "inv([4, 7; 2, 6])", expected: "[0.6, -0.7; -0.2, 0.4]", is_substring: false },
    TestCase { category: "Matrix", expression: "transpose([1, 2; 3, 4])", expected: "[1, 3; 2, 4]", is_substring: false },
    TestCase { category: "Matrix", expression: "trace([1, 2; 3, 4])", expected: "5", is_substring: false },
    TestCase { category: "Matrix", expression: "eye(2)", expected: "[1, 0; 0, 1]", is_substring: false },
    TestCase { category: "Matrix", expression: "det(eye(3))", expected: "1", is_substring: false },
    TestCase { category: "Matrix", expression: "trace(eye(3))", expected: "3", is_substring: false },
    TestCase { category: "Matrix", expression: "[1, 2; 3, 4]^2", expected: "[7, 10; 15, 22]", is_substring: false },

    // 9. Advanced Math, Factorials & Combinatorics
    TestCase { category: "Factorial", expression: "5!", expected: "120", is_substring: false },
    TestCase { category: "Factorial", expression: "0!", expected: "1", is_substring: false },
    TestCase { category: "Factorial", expression: "3! + 4", expected: "10", is_substring: false },
    TestCase { category: "Combinatorics", expression: "gcd(48, 18)", expected: "6", is_substring: false },
    TestCase { category: "Combinatorics", expression: "lcm(12, 15)", expected: "60", is_substring: false },
    TestCase { category: "Combinatorics", expression: "nCr(5, 2)", expected: "10", is_substring: false },
    TestCase { category: "Combinatorics", expression: "nPr(5, 2)", expected: "20", is_substring: false },
    TestCase { category: "Functions", expression: "hypot(3, 4)", expected: "5", is_substring: false },
    TestCase { category: "Functions", expression: "sgn(-42)", expected: "-1", is_substring: false },
    TestCase { category: "Functions", expression: "deg(pi)", expected: "180", is_substring: false },
    TestCase { category: "Functions", expression: "sin(pi / 2)", expected: "1", is_substring: false },
    TestCase { category: "Functions", expression: "cos(0)", expected: "1", is_substring: false },
    TestCase { category: "Functions", expression: "ln(e)", expected: "1", is_substring: false },
    TestCase { category: "Functions", expression: "log10(1000)", expected: "3", is_substring: false },
    TestCase { category: "Functions", expression: "log2(1024)", expected: "10", is_substring: false },
    TestCase { category: "Functions", expression: "abs(-42)", expected: "42", is_substring: false },
    TestCase { category: "Functions", expression: "floor(3.7)", expected: "3", is_substring: false },
    TestCase { category: "Functions", expression: "ceil(3.2)", expected: "4", is_substring: false },
    TestCase { category: "Functions", expression: "round(3.5)", expected: "4", is_substring: false },

    // 10. Number Theory, Prime Factorization & Fibonacci
    TestCase { category: "Primes", expression: "primes(28)", expected: "2² × 7", is_substring: false },
    TestCase { category: "Primes", expression: "primes(360)", expected: "2³ × 3² × 5", is_substring: false },
    TestCase { category: "Primes", expression: "primes28", expected: "2² × 7", is_substring: false },
    TestCase { category: "Fibonacci", expression: "fib(10)", expected: "55", is_substring: false },
    TestCase { category: "Fibonacci", expression: "fib10", expected: "55", is_substring: false },
    TestCase { category: "Factorial", expression: "fac(5)", expected: "120", is_substring: false },
    TestCase { category: "Functions", expression: "sqrt64", expected: "8", is_substring: false },
    TestCase { category: "Functions", expression: "sq5", expected: "25", is_substring: false },
    TestCase { category: "Functions", expression: "cb3", expected: "27", is_substring: false },

    // 11. Conversational Height & Flexible Unit Conversions
    TestCase { category: "Units", expression: "5'11 to cm", expected: "180.34", is_substring: false },
    TestCase { category: "Units", expression: "5 11 to cm", expected: "180.34", is_substring: false },
    TestCase { category: "Units", expression: "5.11 to cm", expected: "180.34", is_substring: false },
    TestCase { category: "Units", expression: "6 2 to cm", expected: "187.96", is_substring: false },

    // 12. Percentage & Expressions
    TestCase { category: "Percentage", expression: "50% * 200", expected: "100", is_substring: false },
    TestCase { category: "Percentage", expression: "15% * 80", expected: "12", is_substring: false },

    // 13. De Morgan's Laws & Tautologies
    TestCase { category: "Logic", expression: "!(p & q) <-> (!p | !q)", expected: "Tautology", is_substring: true },
    TestCase { category: "Logic", expression: "!(p | q) <-> (!p & !q)", expected: "Tautology", is_substring: true },

    // 14. Complex Numbers (a + bi)
    TestCase { category: "Complex", expression: "(3 + 4i) * (1 - 2i)", expected: "11 - 2i", is_substring: false },
    TestCase { category: "Complex", expression: "(1 + i)^4", expected: "-4", is_substring: false },
    TestCase { category: "Complex", expression: "sqrt(-16)", expected: "4i", is_substring: false },
    TestCase { category: "Complex", expression: "abs(3 + 4i)", expected: "5", is_substring: false },
    TestCase { category: "Complex", expression: "conj(3 + 4i)", expected: "3 - 4i", is_substring: false },

    // 15. Symbolic Polynomials & Root Finding
    TestCase { category: "CAS", expression: "2x + 3x", expected: "5x", is_substring: false },
    TestCase { category: "CAS", expression: "(x + 1)^2", expected: "x² + 2x + 1", is_substring: false },
    TestCase { category: "CAS", expression: "(x^2 - 4) / (x - 2)", expected: "x + 2", is_substring: false },
    TestCase { category: "CAS", expression: "roots(x^2 - 5x + 6)", expected: "[2, 3]", is_substring: false },

    // 16. Symbolic & Numerical Calculus (Differentiation & Gauss-Legendre Integration)
    TestCase { category: "Calculus", expression: "diff(x^3 + 2x^2 - 5x + 1)", expected: "3x² + 4x - 5", is_substring: false },
    TestCase { category: "Calculus", expression: "diff(sin(x), 0)", expected: "1", is_substring: false },
    TestCase { category: "Calculus", expression: "integrate(3x^2 + 2x)", expected: "x³ + x²", is_substring: false },
    TestCase { category: "Calculus", expression: "integrate(x^2, 0, 3)", expected: "9", is_substring: false },
    TestCase { category: "Calculus", expression: "integrate(sin(x), 0, pi)", expected: "2", is_substring: false },

    // 17. Advanced Matrix Solvers, Eigenvalues & Rank
    TestCase { category: "Matrix", expression: "rank([1, 2; 2, 4])", expected: "1", is_substring: false },
    TestCase { category: "Matrix", expression: "solve([2, 1; 1, 3], [5, 10])", expected: "[1, 3]", is_substring: false },
    TestCase { category: "Matrix", expression: "eig([2, 0; 0, 4])", expected: "[2, 4]", is_substring: false },
];

pub fn run_all_diagnostics() -> DiagnosticsReport {
    run_diagnostics_filtered(None)
}

pub fn run_diagnostics_filtered(category_filter: Option<&str>) -> DiagnosticsReport {
    let t0 = Instant::now();
    let mut results = Vec::with_capacity(TEST_CASES.len());
    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut latencies = Vec::with_capacity(TEST_CASES.len());

    let mut vars = FxHashMap::default();

    for tc in TEST_CASES {
        if let Some(cat) = category_filter {
            if !tc.category.eq_ignore_ascii_case(cat) {
                continue;
            }
        }
        let t_start = Instant::now();
        let eval_res = execute_with_telemetry(tc.expression, &mut vars);
        let elapsed = t_start.elapsed().as_nanos();
        latencies.push(elapsed);

        let (passed, actual) = match eval_res {
            Ok(res) => {
                let s = res.value.to_string();
                let ok = if tc.is_substring {
                    s.contains(tc.expected)
                } else {
                    s == tc.expected
                };
                (ok, s)
            }
            Err(e) => (false, format!("ERROR: {}", e)),
        };

        if passed {
            total_passed += 1;
        } else {
            total_failed += 1;
        }

        results.push(TestResult {
            category: tc.category,
            expression: tc.expression,
            passed,
            actual,
            expected: tc.expected,
            elapsed_nanos: elapsed,
        });
    }

    latencies.sort_unstable();
    let n = latencies.len();
    let min_nanos = *latencies.first().unwrap_or(&0);
    let max_nanos = *latencies.last().unwrap_or(&0);
    let sum_nanos: u128 = latencies.iter().sum();
    let mean_nanos = if n > 0 { sum_nanos / n as u128 } else { 0 };
    let p50_nanos = if n > 0 { latencies[n * 50 / 100] } else { 0 };
    let p90_nanos = if n > 0 { latencies[n * 90 / 100] } else { 0 };
    let p99_nanos = if n > 0 { latencies[(n * 99 / 100).min(n - 1)] } else { 0 };
    let total_duration_ms = t0.elapsed().as_secs_f64() * 1000.0;

    DiagnosticsReport {
        results,
        total_passed,
        total_failed,
        min_nanos,
        mean_nanos,
        p50_nanos,
        p90_nanos,
        p99_nanos,
        max_nanos,
        total_duration_ms,
    }
}

pub fn evaluate_custom_expression(expr: &str) -> Result<(String, String, u128), String> {
    let mut vars = FxHashMap::default();
    let t0 = Instant::now();
    let res = super::execute_settled_with_telemetry(expr, &mut vars)?;
    let elapsed = t0.elapsed().as_nanos();
    let val_str = res.value.to_string();
    Ok((val_str, res.canonical_echo, elapsed))
}

pub fn print_diagnostics_report(report: &DiagnosticsReport) {
    println!("================================================================================");
    println!("             CONTINUUM: SOTA INTERNAL DIAGNOSTICS & BENCHMARK SUITE             ");
    println!("================================================================================");
    println!("{:<14} | {:<32} | {:<6} | {:<10} | {:<15}", "Category", "Expression", "Status", "Latency", "Result");
    println!("--------------------------------------------------------------------------------");

    for r in &report.results {
        let status = if r.passed { "[PASS]" } else { "[FAIL]" };
        let lat_str = format_nanos(r.elapsed_nanos);
        let actual_short = if r.actual.len() > 14 {
            format!("{}...", &r.actual[..11])
        } else {
            r.actual.clone()
        };
        println!("{:<14} | {:<32} | {:<6} | {:<10} | {:<15}", r.category, r.expression, status, lat_str, actual_short);
    }

    println!("================================================================================");
    println!("TOTAL TESTS : {} | PASSED: {} | FAILED: {}", report.results.len(), report.total_passed, report.total_failed);
    println!("TOTAL TIME  : {:.2} ms", report.total_duration_ms);
    println!("LATENCY TELEMETRY (Sub-Microsecond Ground Truth):");
    println!("  Min : {}", format_nanos(report.min_nanos));
    println!("  Mean: {}", format_nanos(report.mean_nanos));
    println!("  p50 : {}", format_nanos(report.p50_nanos));
    println!("  p90 : {}", format_nanos(report.p90_nanos));
    println!("  p99 : {}", format_nanos(report.p99_nanos));
    println!("  Max : {}", format_nanos(report.max_nanos));
    println!("================================================================================");
}

fn format_nanos(nanos: u128) -> String {
    if nanos < 1_000 {
        format!("{} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.1} µs", nanos as f64 / 1_000.0)
    } else {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    }
}
