use super::lexer::Token;
use super::value::CalcValue;

#[derive(Clone, Debug, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Equals,
    // Logic
    LogicAnd,
    LogicOr,
    LogicXor,
    LogicImplies,
    LogicIff,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOp {
    Pos,
    Neg,
    LogicNot,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(CalcValue),
    Var(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    FnCall(String, Vec<Expr>),
    Matrix(usize, usize, Vec<Expr>),
}

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    open_parens: usize,
    open_brackets: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            open_parens: 0,
            open_brackets: 0,
        }
    }

    fn peek(&self) -> &Token {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos]
        } else {
            &Token::Eof
        }
    }

    #[inline(always)]
    fn consume(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    pub fn parse(&mut self) -> Result<Expr, String> {
        if self.tokens.is_empty() || matches!(self.peek(), Token::Eof) {
            return Err("Empty input".to_string());
        }
        self.parse_bp(0)
    }

    fn parse_bp(&mut self, min_bp: u8) -> Result<Expr, String> {
        let mut lhs = match self.peek() {
            Token::Int(n) => {
                let v = *n;
                self.consume();
                Expr::Literal(CalcValue::FastInt(v))
            }
            Token::ExactDecimal(m, s) => {
                let (mantissa, scale) = (*m, *s);
                self.consume();
                Expr::Literal(CalcValue::normalize_decimal(mantissa, scale))
            }
            Token::Number(f) => {
                let v = *f;
                self.consume();
                Expr::Literal(CalcValue::FastFloat(v))
            }
            Token::Plus => {
                self.consume();
                let rhs = self.parse_bp(16)?;
                Expr::Unary(UnaryOp::Pos, Box::new(rhs))
            }
            Token::Minus => {
                self.consume();
                let rhs = self.parse_bp(16)?;
                Expr::Unary(UnaryOp::Neg, Box::new(rhs))
            }
            Token::LogicNot => {
                self.consume();
                let rhs = self.parse_bp(9)?;
                Expr::Unary(UnaryOp::LogicNot, Box::new(rhs))
            }
            Token::LParen => {
                self.consume();
                self.open_parens += 1;
                let expr = self.parse_bp(0)?;
                if matches!(self.peek(), Token::RParen) {
                    self.consume();
                    self.open_parens -= 1;
                }
                // If unclosed at EOF, speculative auto-closure accepts inner expression
                expr
            }
            Token::LBracket => {
                self.consume();
                self.open_brackets += 1;
                let mat = self.parse_matrix()?;
                if matches!(self.peek(), Token::RBracket) {
                    self.consume();
                    self.open_brackets -= 1;
                }
                mat
            }
            Token::Ident(name) => {
                let id = name.clone();
                self.consume();
                if matches!(self.peek(), Token::LParen) {
                    self.consume();
                    self.open_parens += 1;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Token::RParen | Token::Eof) {
                        loop {
                            args.push(self.parse_bp(0)?);
                            if matches!(self.peek(), Token::Comma) {
                                self.consume();
                            } else {
                                break;
                            }
                        }
                    }
                    if matches!(self.peek(), Token::RParen) {
                        self.consume();
                        self.open_parens -= 1;
                    }
                    Expr::FnCall(id, args)
                } else if is_unary_func_name(&id) && matches!(self.peek(), Token::Int(_) | Token::ExactDecimal(_, _) | Token::Number(_) | Token::Ident(_) | Token::Var(_)) {
                    let arg = self.parse_bp(150)?;
                    Expr::FnCall(id, vec![arg])
                } else {
                    Expr::Var(id)
                }
            }
            Token::Var(v) => {
                let name = v.clone();
                self.consume();
                Expr::Var(name)
            }
            _ => return Err("Expected expression".to_string()),
        };

        // Postfix operators (e.g. Factorial !)
        while matches!(self.peek(), Token::Factorial) {
            self.consume();
            lhs = Expr::FnCall("fact".to_string(), vec![lhs]);
        }

        loop {
            let op = match self.peek() {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                Token::Mul => BinaryOp::Mul,
                Token::Div => BinaryOp::Div,
                Token::FloorDiv => BinaryOp::FloorDiv,
                Token::Mod => BinaryOp::Mod,
                Token::Pow => BinaryOp::Pow,
                Token::Equals => BinaryOp::Equals,
                Token::LogicAnd => BinaryOp::LogicAnd,
                Token::LogicOr => BinaryOp::LogicOr,
                Token::LogicXor => BinaryOp::LogicXor,
                Token::LogicImplies => BinaryOp::LogicImplies,
                Token::LogicIff => BinaryOp::LogicIff,
                _ => break,
            };

            let (l_bp, r_bp) = infix_binding_power(&op);
            if l_bp < min_bp {
                break;
            }

            self.consume();
            // Trailing operator recovery: e.g. "12 + " -> Speculatively yield LHS
            let rhs = match self.parse_bp(r_bp) {
                Ok(rhs) => rhs,
                Err(_) => return Ok(lhs),
            };

            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }

        Ok(lhs)
    }

    fn parse_matrix(&mut self) -> Result<Expr, String> {
        let mut rows_list: Vec<Vec<Expr>> = Vec::new();
        let mut current_row: Vec<Expr> = Vec::new();

        while !matches!(self.peek(), Token::RBracket | Token::Eof) {
            let elem = self.parse_bp(0)?;
            current_row.push(elem);

            if matches!(self.peek(), Token::Comma) {
                self.consume();
            } else if matches!(self.peek(), Token::Semicolon) {
                self.consume();
                rows_list.push(std::mem::take(&mut current_row));
            }
        }
        if !current_row.is_empty() {
            rows_list.push(current_row);
        }

        // Support Python/JSON nested brackets: [[1, 2], [3, 4]]
        if rows_list.len() == 1 && rows_list[0].iter().all(|e| matches!(e, Expr::Matrix(1, _, _))) {
            let r = rows_list[0].len();
            let c = match &rows_list[0][0] {
                Expr::Matrix(1, cols, _) => *cols,
                _ => 0,
            };
            let mut flat = Vec::with_capacity(r * c);
            for e in rows_list.swap_remove(0) {
                if let Expr::Matrix(1, _, row_elems) = e {
                    flat.extend(row_elems);
                }
            }
            return Ok(Expr::Matrix(r, c, flat));
        }

        let r = rows_list.len();
        let c = if r > 0 { rows_list[0].len() } else { 0 };
        let mut flat = Vec::with_capacity(r * c);
        for row in rows_list {
            flat.extend(row);
        }
        Ok(Expr::Matrix(r, c, flat))
    }
}

fn infix_binding_power(op: &BinaryOp) -> (u8, u8) {
    match op {
        BinaryOp::LogicIff => (1, 2),
        BinaryOp::LogicImplies => (3, 4),
        BinaryOp::LogicOr => (5, 6),
        BinaryOp::LogicXor => (5, 6),
        BinaryOp::LogicAnd => (7, 8),
        BinaryOp::Equals => (9, 10),
        BinaryOp::Add | BinaryOp::Sub => (11, 12),
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod => (13, 14),
        BinaryOp::Pow => (16, 15), // Right-associative: 2^3^4 = 2^(3^4)
    }
}

impl Expr {
    /// Generates the canonical mathematical string representation ("understood as")
    /// featuring explicit multiplication symbols (×), typographic power superscripts (e.g. 3²×100),
    /// logic glyphs (∧, ∨, ¬, →, ↔, ⊕), and omitting redundant parentheses based on AST precedence.
    pub fn to_canonical_string(&self) -> String {
        self.to_canonical_prec(0, false)
    }

    fn to_canonical_prec(&self, parent_prec: u8, is_right_child: bool) -> String {
        match self {
            Expr::Literal(val) => {
                match val {
                    CalcValue::FastInt(n) => n.to_string(),
                    CalcValue::ExactDecimal { mantissa, scale } => super::bignum::format_exact_decimal(*mantissa, *scale),
                    CalcValue::FastFloat(f) => {
                        if (*f - std::f64::consts::PI).abs() < 1e-12 {
                            "π (≈ 3.14159)".to_string()
                        } else if (*f - std::f64::consts::E).abs() < 1e-12 {
                            "e (≈ 2.71828)".to_string()
                        } else if (*f - std::f64::consts::TAU).abs() < 1e-12 {
                            "τ (≈ 6.28319)".to_string()
                        } else if (*f - 1.618033988749895).abs() < 1e-12 {
                            "φ (≈ 1.61803)".to_string()
                        } else {
                            super::value::format_clean_float(*f)
                        }
                    }
                    CalcValue::InlineBigInt { .. }
                    | CalcValue::HeapBigInt { .. }
                    | CalcValue::BigIntSci { .. } => val.to_string(),
                    CalcValue::Logic { tautology, contradiction, .. } => {
                        if *tautology {
                            "⊤".to_string()
                        } else if *contradiction {
                            "⊥".to_string()
                        } else {
                            "Logic".to_string()
                        }
                    }
                    CalcValue::Matrix { rows, cols, data } => {
                        use std::fmt::Write;
                        let mut s = String::with_capacity(rows * cols * 8 + 4);
                        s.push('[');
                        for r in 0..*rows {
                            if r > 0 { s.push_str("; "); }
                            for c in 0..*cols {
                                if c > 0 { s.push_str(", "); }
                                let val = data[r * cols + c];
                                if val.fract() == 0.0 {
                                    let _ = write!(s, "{:.0}", val);
                                } else {
                                    let _ = write!(s, "{}", val);
                                }
                            }
                        }
                        s.push(']');
                        s
                    }
                    CalcValue::Factored { display, .. } => display.clone(),
                    CalcValue::Complex { .. } | CalcValue::Polynomial { .. } => val.to_string(),
                }
            }
            Expr::Var(name) => {
                if name.eq_ignore_ascii_case("pi") {
                    "π (≈ 3.14159)".to_string()
                } else if name.eq_ignore_ascii_case("e") {
                    "e (≈ 2.71828)".to_string()
                } else if name.eq_ignore_ascii_case("tau") {
                    "τ (≈ 6.28319)".to_string()
                } else if name.eq_ignore_ascii_case("phi") {
                    "φ (≈ 1.61803)".to_string()
                } else {
                    name.clone()
                }
            }
            Expr::Unary(op, inner) => {
                match op {
                    UnaryOp::Pos => inner.to_canonical_prec(parent_prec, is_right_child),
                    UnaryOp::Neg => {
                        let inner_prec = expr_precedence(inner);
                        if inner_prec < 18 {
                            format!("-({})", inner.to_canonical_prec(0, false))
                        } else {
                            format!("-{}", inner.to_canonical_prec(18, false))
                        }
                    }
                    UnaryOp::LogicNot => {
                        let inner_prec = expr_precedence(inner);
                        if inner_prec < 18 {
                            format!("¬({})", inner.to_canonical_prec(0, false))
                        } else {
                            format!("¬{}", inner.to_canonical_prec(18, false))
                        }
                    }
                }
            }
            Expr::Binary(op, left, right) => {
                // Typographic power superscript glowup: e.g. 3^2 -> 3²
                if matches!(op, BinaryOp::Pow) {
                    if let Some(sup) = try_to_superscript(right) {
                        let left_prec = expr_precedence(left);
                        let left_str = if left_prec < 16 {
                            format!("({})", left.to_canonical_prec(0, false))
                        } else {
                            left.to_canonical_prec(16, false)
                        };
                        return format!("{}{}", left_str, sup);
                    }
                }

                let (my_prec, op_symbol, is_right_assoc) = match op {
                    BinaryOp::LogicIff => (1, "\u{2009}↔\u{2009}", false),
                    BinaryOp::LogicImplies => (3, "\u{2009}→\u{2009}", false),
                    BinaryOp::LogicOr => (5, "\u{2009}∨\u{2009}", false),
                    BinaryOp::LogicXor => (5, "\u{2009}⊕\u{2009}", false),
                    BinaryOp::LogicAnd => (7, "\u{2009}∧\u{2009}", false),
                    BinaryOp::Equals => (9, "\u{2009}=\u{2009}", false),
                    BinaryOp::Add => (11, "\u{2009}+\u{2009}", false),
                    BinaryOp::Sub => (11, "\u{2009}-\u{2009}", false),
                    BinaryOp::Mul => (13, "\u{2009}×\u{2009}", false),
                    BinaryOp::Div => (13, "\u{2009}/\u{2009}", false),
                    BinaryOp::FloorDiv => (13, "\u{2009}//\u{2009}", false),
                    BinaryOp::Mod => (13, "\u{2009}%\u{2009}", false),
                    BinaryOp::Pow => (16, "\u{2009}^\u{2009}", true),
                };

                let left_str = left.to_canonical_prec(my_prec, false);
                let right_str = right.to_canonical_prec(my_prec, true);

                let content = format!("{}{}{}", left_str, op_symbol, right_str);

                let need_parens = my_prec < parent_prec
                    || (my_prec == parent_prec && is_right_child && !is_right_assoc && matches!(op, BinaryOp::Sub | BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod))
                    || (my_prec == parent_prec && !is_right_child && is_right_assoc);

                if need_parens {
                    format!("({})", content)
                } else {
                    content
                }
            }
            Expr::FnCall(name, args) => {
                let lower = name.to_lowercase();
                if lower == "fact" && args.len() == 1 {
                    let arg = &args[0];
                    let arg_prec = expr_precedence(arg);
                    if arg_prec < 18 {
                        format!("({})!", arg.to_canonical_prec(0, false))
                    } else {
                        format!("{}!", arg.to_canonical_prec(18, false))
                    }
                } else if lower == "sqrt" && args.len() == 1 {
                    let arg = &args[0];
                    if matches!(arg, Expr::Literal(_) | Expr::Var(_)) {
                        format!("√{}", arg.to_canonical_prec(0, false))
                    } else {
                        format!("√({})", arg.to_canonical_prec(0, false))
                    }
                } else if lower == "cbrt" && args.len() == 1 {
                    let arg = &args[0];
                    if matches!(arg, Expr::Literal(_) | Expr::Var(_)) {
                        format!("∛{}", arg.to_canonical_prec(0, false))
                    } else {
                        format!("∛({})", arg.to_canonical_prec(0, false))
                    }
                } else {
                    let mut out = String::with_capacity(name.len() + args.len() * 8 + 2);
                    out.push_str(name);
                    out.push('(');
                    for (idx, a) in args.iter().enumerate() {
                        if idx > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&a.to_canonical_prec(0, false));
                    }
                    out.push(')');
                    out
                }
            }
            Expr::Matrix(rows, cols, elems) => {
                let mut s = String::from("[");
                for r in 0..*rows {
                    if r > 0 { s.push_str("; "); }
                    for c in 0..*cols {
                        if c > 0 { s.push_str(", "); }
                        s.push_str(&elems[r * cols + c].to_canonical_prec(0, false));
                    }
                }
                s.push(']');
                s
            }
        }
    }
}

fn expr_precedence(expr: &Expr) -> u8 {
    match expr {
        Expr::Literal(_) | Expr::Var(_) | Expr::FnCall(_, _) | Expr::Matrix(_, _, _) => 20,
        Expr::Unary(_, _) => 18,
        Expr::Binary(op, _, _) => match op {
            BinaryOp::LogicIff => 1,
            BinaryOp::LogicImplies => 3,
            BinaryOp::LogicOr | BinaryOp::LogicXor => 5,
            BinaryOp::LogicAnd => 7,
            BinaryOp::Equals => 9,
            BinaryOp::Add | BinaryOp::Sub => 11,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod => 13,
            BinaryOp::Pow => 16,
        },
    }
}

fn try_to_superscript(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(CalcValue::FastInt(n)) => {
            let s = n.to_string();
            let mut out = String::with_capacity(s.len() * 3);
            for c in s.chars() {
                let sup = match c {
                    '0' => '⁰',
                    '1' => '¹',
                    '2' => '²',
                    '3' => '³',
                    '4' => '⁴',
                    '5' => '⁵',
                    '6' => '⁶',
                    '7' => '⁷',
                    '8' => '⁸',
                    '9' => '⁹',
                    '-' => '⁻',
                    '+' => '⁺',
                    _ => return None,
                };
                out.push(sup);
            }
            Some(out)
        }
        Expr::Literal(CalcValue::FastFloat(f)) if f.fract() == 0.0 && f.abs() < 1000.0 => {
            let n = *f as i64;
            let s = n.to_string();
            let mut out = String::with_capacity(s.len() * 3);
            for c in s.chars() {
                let sup = match c {
                    '0' => '⁰',
                    '1' => '¹',
                    '2' => '²',
                    '3' => '³',
                    '4' => '⁴',
                    '5' => '⁵',
                    '6' => '⁶',
                    '7' => '⁷',
                    '8' => '⁸',
                    '9' => '⁹',
                    '-' => '⁻',
                    '+' => '⁺',
                    _ => return None,
                };
                out.push(sup);
            }
            Some(out)
        }
        Expr::Var(name) if name == "n" => Some("ⁿ".to_string()),
        Expr::Var(name) if name == "x" => Some("ˣ".to_string()),
        _ => None,
    }
}

pub fn is_unary_func_name(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
        | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh"
        | "sec" | "csc" | "cot"
        | "ln" | "log10" | "log2" | "log" | "exp"
        | "sqrt" | "cbrt" | "root" | "rt" | "sq" | "square" | "cb" | "cube"
        | "fact" | "fac" | "factorial"
        | "fib" | "fibonacci"
        | "primes" | "prime" | "factors" | "factor"
        | "abs" | "sgn" | "sign" | "floor" | "ceil" | "round" | "deg" | "rad"
        | "popcnt" | "clz" | "ctz" | "bnot"
        | "re" | "im" | "conj" | "arg" | "phase"
        | "rank" | "norm" | "frobenius" | "eig" | "eigenvalues"
        | "roots" | "solve" | "deriv" | "diff" | "integrate" | "int"
    )
}

