#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Number(f64),
    ExactDecimal(i128, u32),
    Int(i64),
    Ident(String),
    Var(String),
    // Arithmetic Operators
    Plus,
    Minus,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Factorial,
    // Parentheses & Brackets
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Equals,
    // Logic Operators
    LogicAnd,
    LogicOr,
    LogicNot,
    LogicXor,
    LogicImplies,
    LogicIff,
    // Speculative / EOF
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionDomain {
    Logic,
    LinearAlgebra,
    Arithmetic,
}

fn parse_human_height_inches(lhs: &str) -> Option<f64> {
    let trimmed = lhs.trim();

    // Case 1: Quotes/apostrophes: 5'11, 5'11", 5' 11", 5'11'', 5′11″
    if let Some((ft_str, in_str)) = trimmed.split_once('\'').or_else(|| trimmed.split_once('′')) {
        let feet: f64 = ft_str.trim().parse().ok()?;
        let in_clean = in_str.trim_matches(|c: char| c == '"' || c == '″' || c == '\'' || c.is_alphabetic() || c.is_whitespace());
        let inches: f64 = if in_clean.is_empty() { 0.0 } else { in_clean.parse().ok()? };
        return Some(feet * 12.0 + inches);
    }

    // Case 2: Feet & inches keywords: 5 ft 11 in, 5ft 11in, 5 ft 11, 5ft11
    let lower = trimmed.to_lowercase();
    if let Some(ft_idx) = lower.find("ft").or_else(|| lower.find("feet")).or_else(|| lower.find("foot")) {
        let feet_part = lower[..ft_idx].trim();
        let feet: f64 = feet_part.parse().ok()?;
        let rem = lower[ft_idx..].trim_start_matches(|c: char| c.is_alphabetic() || c.is_whitespace());
        let in_clean = rem.trim_matches(|c: char| c.is_alphabetic() || c == '"' || c == '\'' || c.is_whitespace());
        let inches: f64 = if in_clean.is_empty() { 0.0 } else { in_clean.parse().ok()? };
        return Some(feet * 12.0 + inches);
    }

    // Case 3: Two space-separated numbers: "5 11", "5  11", "6 2"
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() == 2 {
        if let (Ok(feet), Ok(inches)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
            if (1.0..=8.0).contains(&feet) && (0.0..12.0).contains(&inches) {
                return Some(feet * 12.0 + inches);
            }
        }
    }

    // Case 4: Dot notation in height context: "5.11", "5.2"
    if let Some((ft_str, in_str)) = trimmed.split_once('.') {
        if let (Ok(feet), Ok(inches)) = (ft_str.trim().parse::<f64>(), in_str.trim().parse::<f64>()) {
            if (1.0..=8.0).contains(&feet) && (0.0..12.0).contains(&inches) {
                return Some(feet * 12.0 + inches);
            }
        }
    }

    None
}

#[inline(always)]
fn is_logic_var_byte(b: u8) -> bool {
    matches!(b | 32, b'p' | b'q' | b'r' | b's' | b't' | b'u')
}

#[inline(always)]
fn is_extended_logic_var_byte(b: u8) -> bool {
    matches!(
        b | 32,
        b'p' | b'q' | b'r' | b's' | b't' | b'u' | b'a' | b'b' | b'c' | b'd' | b'e'
    )
}

#[inline]
fn find_unit_keyword_pos(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    for i in (0..=(bytes.len() - 4)).rev() {
        if bytes[i] == b' ' && bytes[i + 3] == b' ' {
            let b1 = bytes[i + 1] | 32;
            let b2 = bytes[i + 2] | 32;
            if (b1 == b't' && b2 == b'o') || (b1 == b'i' && b2 == b'n') {
                return Some(i);
            }
        }
    }
    None
}

pub struct Lexer;

impl Lexer {
    /// Detects whether an expression belongs to Logic, Linear Algebra, or Arithmetic (Zero Heap Allocations)
    pub fn detect_domain(raw: &str) -> ExpressionDomain {
        let trimmed = raw.trim();

        // 1. Linear Algebra Matrix Domain
        if (trimmed.starts_with('[') && trimmed.ends_with(']'))
            || raw.contains(r"\begin{")
            || raw.contains("det(")
            || raw.contains("inv(")
            || raw.contains("trace(")
            || raw.contains("transpose(")
            || raw.contains("eye(")
            || raw.contains("rank(")
            || raw.contains("eig(")
        {
            return ExpressionDomain::LinearAlgebra;
        }

        // 2. Logic Domain Indicators (Fast ASCII & Unicode operator checks)
        if raw.contains("->")
            || raw.contains("<->")
            || raw.contains("<=>")
            || raw.contains('&')
            || raw.contains('|')
            || raw.contains('~')
        {
            return ExpressionDomain::Logic;
        }

        if raw.contains('\\')
            && (raw.contains(r"\lor")
                || raw.contains(r"\vee")
                || raw.contains(r"\land")
                || raw.contains(r"\wedge")
                || raw.contains(r"\neg")
                || raw.contains(r"\lnot")
                || raw.contains(r"\implies")
                || raw.contains(r"\iff")
                || raw.contains(r"\equiv")
                || raw.contains(r"\top")
                || raw.contains(r"\bot"))
        {
            return ExpressionDomain::Logic;
        }

        if !raw.is_ascii()
            && (raw.contains('∧')
                || raw.contains('∨')
                || raw.contains('¬')
                || raw.contains('⊕')
                || raw.contains('→')
                || raw.contains('↔')
                || raw.contains('≡')
                || raw.contains('⊃')
                || raw.contains('⊤')
                || raw.contains('⊥'))
        {
            return ExpressionDomain::Logic;
        }

        // 3. Single-pass zero-allocation scan for logic keywords, negations, and shorthands
        let bytes = raw.as_bytes();
        let len = bytes.len();
        let has_eq = bytes.contains(&b'=');
        let has_plus_or_mul = bytes.contains(&b'+') || bytes.contains(&b'*');
        let has_digits = bytes.iter().any(|b| b.is_ascii_digit());

        let mut i = 0;
        while i < len {
            let b = bytes[i];
            if b.is_ascii_alphabetic() {
                let start = i;
                while i < len && bytes[i].is_ascii_alphanumeric() {
                    i += 1;
                }
                let word = &raw[start..i];
                let w_len = word.len();

                if w_len >= 2
                    && (word.eq_ignore_ascii_case("and")
                        || word.eq_ignore_ascii_case("or")
                        || word.eq_ignore_ascii_case("not")
                        || word.eq_ignore_ascii_case("xor")
                        || word.eq_ignore_ascii_case("implies")
                        || word.eq_ignore_ascii_case("iff")
                        || word.eq_ignore_ascii_case("tautology")
                        || word.eq_ignore_ascii_case("top")
                        || word.eq_ignore_ascii_case("bot"))
                {
                    return ExpressionDomain::Logic;
                }

                // 3-char logic shorthand: pvq, qvp, etc.
                if w_len == 3 {
                    let wb = word.as_bytes();
                    if is_logic_var_byte(wb[0]) && (wb[1] | 32) == b'v' && is_logic_var_byte(wb[2]) {
                        return ExpressionDomain::Logic;
                    }
                }

                // Isolated 1-char logic variable: p, q, r, s, t, u
                if w_len == 1 && is_logic_var_byte(word.as_bytes()[0]) {
                    let prev_b = if start > 0 { bytes[start - 1] } else { b' ' };
                    let next_b = if i < len { bytes[i] } else { b' ' };
                    if matches!(prev_b, b'\'' | b'"' | b'`' | b'!' | b'~')
                        || matches!(next_b, b'\'' | b'"' | b'`')
                    {
                        return ExpressionDomain::Logic;
                    }
                    if has_eq || (has_plus_or_mul && !has_digits) {
                        return ExpressionDomain::Logic;
                    }
                }
            } else {
                i += 1;
            }
        }

        ExpressionDomain::Arithmetic
    }

    /// Desugars raw user input into canonical format before tokenizing
    pub fn desugar(raw: &str) -> String {
        let domain = Self::detect_domain(raw);
        let mut s = raw.to_string();

        // Fast-path guard: only run KaTeX / LaTeX backslash desugaring when '\' is present
        if s.contains('\\') {
            let mut scratch = String::with_capacity(s.len() + 16);
            // 1. KaTeX & LaTeX Matrix Environments (\begin{pmatrix} ... \end{pmatrix}, etc.)
            for &(begin_tag, end_tag, is_det) in &[
                (r"\begin{pmatrix}", r"\end{pmatrix}", false),
                (r"\begin{bmatrix}", r"\end{bmatrix}", false),
                (r"\begin{matrix}", r"\end{matrix}", false),
                (r"\begin{vmatrix}", r"\end{vmatrix}", true),
            ] {
                while let Some(start) = s.find(begin_tag) {
                    if let Some(end_rel) = s[start + begin_tag.len()..].find(end_tag) {
                        let inner_start = start + begin_tag.len();
                        let inner_end = inner_start + end_rel;
                        let inner_norm = s[inner_start..inner_end].replace(r"\\", "; ").replace('&', ", ");
                        scratch.clear();
                        scratch.push_str(&s[..start]);
                        if is_det {
                            scratch.push_str("det([");
                        } else {
                            scratch.push('[');
                        }
                        scratch.push_str(inner_norm.trim());
                        if is_det {
                            scratch.push_str("])");
                        } else {
                            scratch.push(']');
                        }
                        scratch.push_str(&s[inner_end + end_tag.len()..]);
                        std::mem::swap(&mut s, &mut scratch);
                    } else {
                        break;
                    }
                }
            }

            // 2. KaTeX Parentheses, Brackets, Norms
            s = s.replace(r"\left(", "(")
                 .replace(r"\right)", ")")
                 .replace(r"\left[", "[")
                 .replace(r"\right]", "]")
                 .replace(r"\{", "[")
                 .replace(r"\}", "]");

            // 3. KaTeX Logic Operators & Truth Constants
            s = s.replace(r"\lor", " | ")
                 .replace(r"\vee", " | ")
                 .replace(r"\land", " & ")
                 .replace(r"\wedge", " & ")
                 .replace(r"\neg", " ! ")
                 .replace(r"\lnot", " ! ")
                 .replace(r"\implies", " -> ")
                 .replace(r"\Rightarrow", " -> ")
                 .replace(r"\iff", " <-> ")
                 .replace(r"\Leftrightarrow", " <-> ")
                 .replace(r"\equiv", " <-> ")
                 .replace(r"\oplus", " ^ ")
                 .replace(r"\top", " 1 ")
                 .replace(r"\bot", " 0 ")
                 .replace(r"\to", " -> ");

            // 4. KaTeX Math Operators & Functions
            s = s.replace(r"\operatorname{det}", " det ")
                 .replace(r"\det", " det ")
                 .replace(r"\operatorname{trace}", " trace ")
                 .replace(r"\operatorname{tr}", " trace ")
                 .replace(r"\trace", " trace ")
                 .replace(r"\operatorname{inv}", " inv ")
                 .replace(r"\inv", " inv ")
                 .replace(r"\operatorname{transpose}", " transpose ")
                 .replace(r"\cdot", " * ")
                 .replace(r"\times", " * ")
                 .replace(r"\div", " / ")
                 .replace(r"\pm", " ± ")
                 .replace(r"\mp", " ∓ ")
                 .replace(r"\pi", " pi ")
                 .replace(r"\tau", " (2*pi) ")
                 .replace(r"\sin", " sin ")
                 .replace(r"\cos", " cos ")
                 .replace(r"\tan", " tan ")
                 .replace(r"\ln", " ln ")
                 .replace(r"\log", " log ")
                 .replace(r"\exp", " exp ");

            // 5. Fractions: \frac{a}{b} -> ((a)/(b))
            while let Some(idx) = s.find(r"\frac{") {
                if let Some(first_close) = s[idx + 6..].find('}') {
                    let num_end = idx + 6 + first_close;
                    let num = &s[idx + 6..num_end];
                    if s[num_end..].starts_with("}{") {
                        if let Some(second_close) = s[num_end + 2..].find('}') {
                            let den_end = num_end + 2 + second_close;
                            let den = &s[num_end + 2..den_end];
                            let repl = format!("(({})/({}))", num, den);
                            s.replace_range(idx..=den_end, &repl); // parallel-ok
                            continue;
                        }
                    }
                }
                break;
            }

            // 6. Roots: \sqrt[n]{a} -> ((a)^(1/(n))) and \sqrt{a} -> sqrt(a)
            while let Some(idx) = s.find(r"\sqrt[") {
                if let Some(bracket_close) = s[idx + 6..].find(']') {
                    let deg_end = idx + 6 + bracket_close;
                    let degree = &s[idx + 6..deg_end];
                    if s[deg_end..].starts_with("]{") {
                        if let Some(brace_close) = s[deg_end + 2..].find('}') {
                            let rad_end = deg_end + 2 + brace_close;
                            let radicand = &s[deg_end + 2..rad_end];
                            let repl = format!("(({})^(1/({})))", radicand, degree);
                            s.replace_range(idx..=rad_end, &repl); // parallel-ok
                            continue;
                        }
                    }
                }
                break;
            }

            while let Some(idx) = s.find(r"\sqrt{") {
                if let Some(close) = s[idx + 6..].find('}') {
                    let end = idx + 6 + close;
                    let inner = &s[idx + 6..end];
                    let repl = format!("sqrt({})", inner);
                    s.replace_range(idx..=end, &repl); // parallel-ok
                    continue;
                }
                break;
            }
        }

        // Desugar Unicode root symbols: √ and ∛ (only when non-ASCII)
        if !s.is_ascii() {
            if s.contains('√') {
                s = desugar_root_char(s, '√', "sqrt");
            }
            if s.contains('∛') {
                s = desugar_root_char(s, '∛', "cbrt");
            }
        }

        // Percentage handling: "50% of 200" -> "((50 / 100) * (200))", and "50%" -> "(50 / 100)"
        if s.contains('%') {
            s = desugar_percentages(s);
        }

        // 7. Conversational Unit Conversion: "10 km to m", "100 c to f", "5 lb in kg", "5'11 to cm", "5 11 to cm"
        if let Some(to_pos) = find_unit_keyword_pos(&s) {
            let kw_len = 4;
            let lhs = s[..to_pos].trim();
            let rhs = s[to_pos + kw_len..].trim();
            if let Some(target_u) = crate::engine::units::find_unit(rhs) {
                if target_u.family == crate::engine::units::UnitFamily::Length {
                    if let Some(total_in) = parse_human_height_inches(lhs) {
                        s = format!("convert({}, \"in\", \"{}\")", total_in, target_u.name);
                    } else {
                        let mut split_byte = lhs.len();
                        for (idx, ch) in lhs.char_indices().rev() {
                            if ch.is_alphabetic() || ch == '°' || ch == 'µ' {
                                split_byte = idx;
                            } else {
                                break;
                            }
                        }
                        let from_unit_raw = &lhs[split_byte..];
                        let val_expr_trimmed = lhs[..split_byte].trim();
                        if let Some(from_u) = crate::engine::units::find_unit(from_unit_raw) {
                            if from_u.family == target_u.family && !val_expr_trimmed.is_empty() {
                                s = format!("convert({}, \"{}\", \"{}\")", val_expr_trimmed, from_u.name, target_u.name);
                            }
                        }
                    }
                } else {
                    let mut split_byte = lhs.len();
                    for (idx, ch) in lhs.char_indices().rev() {
                        if ch.is_alphabetic() || ch == '°' || ch == 'µ' {
                            split_byte = idx;
                        } else {
                            break;
                        }
                    }
                    let from_unit_raw = &lhs[split_byte..];
                    let val_expr_trimmed = lhs[..split_byte].trim();
                    if let Some(from_u) = crate::engine::units::find_unit(from_unit_raw) {
                        if from_u.family == target_u.family && !val_expr_trimmed.is_empty() {
                            s = format!("convert({}, \"{}\", \"{}\")", val_expr_trimmed, from_u.name, target_u.name);
                        }
                    }
                }
            }
        }

        // 8 & 9. Single-pass Unicode Superscripts & Common Unicode Math Substitutions (only when non-ASCII)
        if !s.is_ascii() {
            let mut sup_out = String::with_capacity(s.len() + 16);
            let mut sup_buf = String::new();
            for ch in s.chars() {
                let mapped_sup = match ch {
                    '⁰' => Some('0'),
                    '¹' => Some('1'),
                    '²' => Some('2'),
                    '³' => Some('3'),
                    '⁴' => Some('4'),
                    '⁵' => Some('5'),
                    '⁶' => Some('6'),
                    '⁷' => Some('7'),
                    '⁸' => Some('8'),
                    '⁹' => Some('9'),
                    '⁻' => Some('-'),
                    '⁺' => Some('+'),
                    _ => None,
                };
                if let Some(digit) = mapped_sup {
                    sup_buf.push(digit);
                } else {
                    if !sup_buf.is_empty() {
                        sup_out.push_str("^(");
                        sup_out.push_str(&sup_buf);
                        sup_out.push(')');
                        sup_buf.clear();
                    }
                    match ch {
                        '·' | '×' => sup_out.push('*'),
                        '÷' => sup_out.push('/'),
                        '∧' => sup_out.push('&'),
                        '∨' => sup_out.push('|'),
                        '¬' => sup_out.push('!'),
                        '⊕' => sup_out.push('^'),
                        '→' | '⇒' | '⊃' => sup_out.push_str("->"),
                        '↔' | '⇔' | '≡' => sup_out.push_str("<->"),
                        'π' => sup_out.push_str("pi"),
                        'τ' => sup_out.push_str("tau"),
                        'φ' => sup_out.push_str("phi"),
                        _ => sup_out.push(ch),
                    }
                }
            }
            if !sup_buf.is_empty() {
                sup_out.push_str("^(");
                sup_out.push_str(&sup_buf);
                sup_out.push(')');
            }
            s = sup_out;
        }

        // 10. Domain-Specific Operator Overloading (Single-pass zero-redundancy transform)
        match domain {
            ExpressionDomain::Logic => {
                // Handle whole-expression quotes if wrapped: e.g. 'p=p' -> p=p
                let trimmed = s.trim();
                let active_slice = if (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 3 && !trimmed[1..trimmed.len() - 1].contains('\''))
                    || (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 3 && !trimmed[1..trimmed.len() - 1].contains('"'))
                    || (trimmed.starts_with('`') && trimmed.ends_with('`') && trimmed.len() >= 3 && !trimmed[1..trimmed.len() - 1].contains('`'))
                {
                    &trimmed[1..trimmed.len() - 1]
                } else {
                    s.as_str()
                };

                let bytes = active_slice.as_bytes();
                let len = bytes.len();
                let mut out = String::with_capacity(len + 16);
                let mut i = 0;

                while i < len {
                    let b = bytes[i];
                    // Prefix negation: 'p, "p, `p, ~p -> (!p)
                    if matches!(b, b'\'' | b'"' | b'`' | b'~')
                        && i + 1 < len
                        && is_extended_logic_var_byte(bytes[i + 1])
                        && (i + 2 == len || !bytes[i + 2].is_ascii_alphanumeric())
                    {
                        out.push_str("(!");
                        out.push(bytes[i + 1] as char);
                        out.push(')');
                        i += 2;
                        continue;
                    }

                    // 3-char logic shorthand: pvq, pVq -> p | q
                    if is_extended_logic_var_byte(b)
                        && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                        && i + 2 < len
                        && (bytes[i + 1] | 32) == b'v'
                        && is_extended_logic_var_byte(bytes[i + 2])
                        && (i + 3 == len || !bytes[i + 3].is_ascii_alphanumeric())
                    {
                        out.push(b as char);
                        out.push_str(" | ");
                        out.push(bytes[i + 2] as char);
                        i += 3;
                        continue;
                    }

                    // Postfix negation: p', p", p` -> (!p)
                    if is_extended_logic_var_byte(b)
                        && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                        && i + 1 < len
                        && matches!(bytes[i + 1], b'\'' | b'"' | b'`')
                    {
                        out.push_str("(!");
                        out.push(b as char);
                        out.push(')');
                        i += 2;
                        continue;
                    }

                    // Overload '==' and '=' to Bi-implication '<->' in Logic
                    if b == b'=' {
                        if i + 1 < len && bytes[i + 1] == b'=' {
                            out.push_str(" <-> ");
                            i += 2;
                            continue;
                        }
                        let prev = if i > 0 { bytes[i - 1] } else { b' ' };
                        let next = if i + 1 < len { bytes[i + 1] } else { b' ' };
                        if prev != b'<' && prev != b'-' && prev != b'>' && prev != b'!' && next != b'>' && next != b'=' {
                            out.push_str(" <-> ");
                            i += 1;
                            continue;
                        }
                    }

                    // Overload '+' to OR (' | ') and '*' to AND (' & ') in Boolean logic
                    if b == b'+' {
                        out.push_str(" | ");
                        i += 1;
                        continue;
                    }
                    if b == b'*' {
                        out.push_str(" & ");
                        i += 1;
                        continue;
                    }

                    // Copy UTF-8 character safely
                    let ch = active_slice[i..].chars().next().unwrap();
                    out.push(ch);
                    i += ch.len_utf8();
                }
                s = out;
            }
            ExpressionDomain::LinearAlgebra => {
                // Preserves matrix multiplications and linear combinations
            }
            ExpressionDomain::Arithmetic => {
                // If user wrote ==, normalize to = for equality testing
                if s.contains("==") {
                    s = s.replace("==", "=");
                }
            }
        }

        s
    }

    /// Tokenizes an expression with the Deterministic 'x' Disambiguation Matrix
    /// and Implied Multiplication emission
    pub fn tokenize(input: &str) -> (Vec<Token>, String) {
        let desugared = Self::desugar(input);
        let chars: Vec<char> = desugared.chars().collect();
        let len = chars.len();
        let mut i = 0;
        let mut tokens = Vec::with_capacity(32);
        let mut canonical_echo = String::with_capacity(64);

        while i < len {
            let ch = chars[i];

            // Skip whitespace
            if ch.is_whitespace() {
                i += 1;
                continue;
            }

            // Hex literal: 0x...
            if ch == '0' && i + 1 < len && (chars[i + 1] == 'x' || chars[i + 1] == 'X') {
                let start = i;
                i += 2;
                while i < len && (chars[i].is_ascii_hexdigit() || chars[i] == '_') {
                    i += 1;
                }
                let hex_str: String = chars[start + 2..i].iter().filter(|&&c| c != '_').collect();
                if let Ok(val) = i64::from_str_radix(&hex_str, 16) {
                    Self::push_token_with_implied(&mut tokens, Token::Int(val));
                    canonical_echo.push_str(&val.to_string());
                } else if let Ok(val) = i128::from_str_radix(&hex_str, 16) {
                    Self::push_token_with_implied(&mut tokens, Token::ExactDecimal(val, 0));
                    canonical_echo.push_str(&val.to_string());
                }
                continue;
            }

            // Binary literal: 0b...
            if ch == '0' && i + 1 < len && (chars[i + 1] == 'b' || chars[i + 1] == 'B') {
                let start = i;
                i += 2;
                while i < len && (chars[i] == '0' || chars[i] == '1' || chars[i] == '_') {
                    i += 1;
                }
                let bin_str: String = chars[start + 2..i].iter().filter(|&&c| c != '_').collect();
                if let Ok(val) = i64::from_str_radix(&bin_str, 2) {
                    Self::push_token_with_implied(&mut tokens, Token::Int(val));
                    canonical_echo.push_str(&val.to_string());
                } else if let Ok(val) = i128::from_str_radix(&bin_str, 2) {
                    Self::push_token_with_implied(&mut tokens, Token::ExactDecimal(val, 0));
                    canonical_echo.push_str(&val.to_string());
                }
                continue;
            }

            // Octal literal: 0o...
            if ch == '0' && i + 1 < len && (chars[i + 1] == 'o' || chars[i + 1] == 'O') {
                let start = i;
                i += 2;
                while i < len && ((chars[i] >= '0' && chars[i] <= '7') || chars[i] == '_') {
                    i += 1;
                }
                let oct_str: String = chars[start + 2..i].iter().filter(|&&c| c != '_').collect();
                if let Ok(val) = i64::from_str_radix(&oct_str, 8) {
                    Self::push_token_with_implied(&mut tokens, Token::Int(val));
                    canonical_echo.push_str(&val.to_string());
                } else if let Ok(val) = i128::from_str_radix(&oct_str, 8) {
                    Self::push_token_with_implied(&mut tokens, Token::ExactDecimal(val, 0));
                    canonical_echo.push_str(&val.to_string());
                }
                continue;
            }

            // Numbers: integer, float, bignum with flexible Anglo-American (1,000.0) vs Continental European (1.000,0) detection
            if ch.is_ascii_digit() || ((ch == '.' || ch == ',') && i + 1 < len && chars[i + 1].is_ascii_digit()) {
                let start = i;
                i += 1;
                while i < len {
                    let c = chars[i];
                    if c.is_ascii_digit() || c == '_' {
                        i += 1;
                    } else if (c == '.' || c == ',') && i + 1 < len && chars[i + 1].is_ascii_digit() {
                        // Flanked punctuation mark between digits (e.g. 1.000, 1,000, 1,5, 1.5)
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Check for 'e' or 'E' scientific notation
                if i + 1 < len && (chars[i] == 'e' || chars[i] == 'E') && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '+' || chars[i + 1] == '-') {
                    i += 2;
                    while i < len && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }

                let num_raw: String = chars[start..i].iter().filter(|&&c| c != '_').collect();
                let normalized = Self::normalize_number_punctuation(&num_raw);

                if normalized.contains('.') || normalized.contains('e') || normalized.contains('E') {
                    if let Some((m, s)) = super::bignum::parse_exact_decimal(&normalized) {
                        Self::push_token_with_implied(&mut tokens, Token::ExactDecimal(m, s));
                        canonical_echo.push_str(&normalized);
                    } else if let Ok(f) = normalized.parse::<f64>() {
                        Self::push_token_with_implied(&mut tokens, Token::Number(f));
                        canonical_echo.push_str(&normalized);
                    }
                } else if let Ok(n) = normalized.parse::<i64>() {
                    Self::push_token_with_implied(&mut tokens, Token::Int(n));
                    canonical_echo.push_str(&normalized);
                } else if let Some((m, s)) = super::bignum::parse_exact_decimal(&normalized) {
                    Self::push_token_with_implied(&mut tokens, Token::ExactDecimal(m, s));
                    canonical_echo.push_str(&normalized);
                } else if let Ok(f) = normalized.parse::<f64>() {
                    Self::push_token_with_implied(&mut tokens, Token::Number(f));
                    canonical_echo.push_str(&normalized);
                }

                // Check for percentage: e.g. 50%
                if i < len && chars[i] == '%' {
                    i += 1;
                    // Tokenize % as divide by 100
                    tokens.push(Token::Div);
                    tokens.push(Token::ExactDecimal(100, 0));
                    canonical_echo.push_str(" / 100");
                }
                continue;
            }

            // Exponentiation: ** or ^
            if ch == '*' && i + 1 < len && chars[i + 1] == '*' {
                tokens.push(Token::Pow);
                canonical_echo.push_str("\u{2009}^\u{2009}");
                i += 2;
                continue;
            }

            // Integer Floor Div: //
            if ch == '/' && i + 1 < len && chars[i + 1] == '/' {
                tokens.push(Token::FloorDiv);
                canonical_echo.push_str("\u{2009}//\u{2009}");
                i += 2;
                continue;
            }

            // Logic Implies: -> or =>
            if (ch == '-' || ch == '=') && i + 1 < len && chars[i + 1] == '>' {
                tokens.push(Token::LogicImplies);
                canonical_echo.push_str("\u{2009}->\u{2009}");
                i += 2;
                continue;
            }

            // Logic Iff: <-> or <=>
            if ch == '<' && i + 2 < len && (chars[i + 1] == '-' || chars[i + 1] == '=') && chars[i + 2] == '>' {
                tokens.push(Token::LogicIff);
                canonical_echo.push_str("\u{2009}<->\u{2009}");
                i += 3;
                continue;
            }

            // Consecutive Sign Collapsing: ++ or -- or + - or - -
            if ch == '+' {
                if i + 1 < len && chars[i + 1] == '+' {
                    tokens.push(Token::Plus);
                    canonical_echo.push_str("\u{2009}+\u{2009}");
                    i += 2;
                    continue;
                }
                if i + 1 < len && chars[i + 1] == '-' {
                    // + - resolves to -
                    tokens.push(Token::Minus);
                    canonical_echo.push_str("\u{2009}-\u{2009}");
                    i += 2;
                    continue;
                }
                tokens.push(Token::Plus);
                canonical_echo.push_str("\u{2009}+\u{2009}");
                i += 1;
                continue;
            }

            if ch == '-' {
                if i + 1 < len && chars[i + 1] == '-' {
                    // - - resolves to +
                    tokens.push(Token::Plus);
                    canonical_echo.push_str("\u{2009}+\u{2009}");
                    i += 2;
                    continue;
                }
                tokens.push(Token::Minus);
                canonical_echo.push_str("\u{2009}-\u{2009}");
                i += 1;
                continue;
            }

            // Deterministic 'x' Disambiguation Matrix
            if (ch == 'x' || ch == 'X') && (i + 1 >= len || (!chars[i + 1].is_alphabetic() && chars[i + 1] != '_')) {
                let prev_is_num = tokens.last().map_or(false, |t| matches!(t, Token::Int(_) | Token::ExactDecimal(_, _) | Token::Number(_) | Token::RParen | Token::RBracket));
                let next_is_digit = i + 1 < len && chars[i + 1].is_ascii_digit();
                let next_is_space = i + 1 < len && chars[i + 1].is_whitespace();
                let prev_is_space = i > 0 && chars[i - 1].is_whitespace();

                // Case A: 3x5 or 3 x 5 -> multiplication
                if prev_is_num && ((!prev_is_space && next_is_digit) || (prev_is_space && (next_is_space || next_is_digit))) {
                    tokens.push(Token::Mul);
                    canonical_echo.push_str("\u{2009}*\u{2009}");
                    i += 1;
                    continue;
                }

                // Case B: 3x + 5 or 2x^2 -> 3 * x
                if prev_is_num && !prev_is_space && !next_is_digit {
                    tokens.push(Token::Mul);
                    tokens.push(Token::Var("x".to_string()));
                    canonical_echo.push_str("\u{2009}*\u{2009}x");
                    i += 1;
                    continue;
                }
            }

            // Logic characters
            match ch {
                '&' => { tokens.push(Token::LogicAnd); canonical_echo.push_str("\u{2009}&\u{2009}"); i += 1; continue; }
                '|' => { tokens.push(Token::LogicOr); canonical_echo.push_str("\u{2009}|\u{2009}"); i += 1; continue; }
                '!' => {
                    let prev_is_operand = tokens.last().map_or(false, |t| matches!(t, Token::Int(_) | Token::ExactDecimal(_, _) | Token::Number(_) | Token::RParen | Token::RBracket | Token::Var(_) | Token::Ident(_) | Token::Factorial));
                    if prev_is_operand {
                        tokens.push(Token::Factorial);
                        canonical_echo.push('!');
                    } else {
                        tokens.push(Token::LogicNot);
                        canonical_echo.push('!');
                    }
                    i += 1;
                    continue;
                }
                '~' => { tokens.push(Token::LogicNot); canonical_echo.push('!'); i += 1; continue; }
                '*' => { tokens.push(Token::Mul); canonical_echo.push_str("\u{2009}*\u{2009}"); i += 1; continue; }
                '/' | '\\' => { tokens.push(Token::Div); canonical_echo.push_str("\u{2009}/\u{2009}"); i += 1; continue; }
                '%' => { tokens.push(Token::Mod); canonical_echo.push_str("\u{2009}%\u{2009}"); i += 1; continue; }
                '^' => { tokens.push(Token::Pow); canonical_echo.push_str("\u{2009}^\u{2009}"); i += 1; continue; }
                '(' => {
                    Self::push_token_with_implied(&mut tokens, Token::LParen);
                    canonical_echo.push('(');
                    i += 1;
                    continue;
                }
                ')' => { tokens.push(Token::RParen); canonical_echo.push(')'); i += 1; continue; }
                '[' => {
                    Self::push_token_with_implied(&mut tokens, Token::LBracket);
                    canonical_echo.push('[');
                    i += 1;
                    continue;
                }
                ']' => { tokens.push(Token::RBracket); canonical_echo.push(']'); i += 1; continue; }
                ',' => { tokens.push(Token::Comma); canonical_echo.push_str(",\u{2009}"); i += 1; continue; }
                ';' => { tokens.push(Token::Semicolon); canonical_echo.push_str(";\u{2009}"); i += 1; continue; }
                '=' => { tokens.push(Token::Equals); canonical_echo.push_str("\u{2009}=\u{2009}"); i += 1; continue; }
                _ => {}
            }

            // Identifiers, Constants, Functions, or Variables
            if ch.is_alphabetic() || ch == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Logic keywords
                match word.to_lowercase().as_str() {
                    "v" => { tokens.push(Token::LogicOr); canonical_echo.push_str("\u{2009}|\u{2009}"); continue; }
                    "and" => { tokens.push(Token::LogicAnd); canonical_echo.push_str("\u{2009}&\u{2009}"); continue; }
                    "or" => { tokens.push(Token::LogicOr); canonical_echo.push_str("\u{2009}|\u{2009}"); continue; }
                    "not" => { tokens.push(Token::LogicNot); canonical_echo.push('!'); continue; }
                    "xor" => { tokens.push(Token::LogicXor); canonical_echo.push_str("\u{2009}^\u{2009}"); continue; }
                    "mod" => { tokens.push(Token::Mod); canonical_echo.push_str("\u{2009}mod\u{2009}"); continue; }
                    "pi" => {
                        Self::push_token_with_implied(&mut tokens, Token::Number(std::f64::consts::PI));
                        canonical_echo.push_str("π (≈ 3.14159)");
                        continue;
                    }
                    "e" => {
                        Self::push_token_with_implied(&mut tokens, Token::Number(std::f64::consts::E));
                        canonical_echo.push_str("e (≈ 2.71828)");
                        continue;
                    }
                    "tau" => {
                        Self::push_token_with_implied(&mut tokens, Token::Number(std::f64::consts::TAU));
                        canonical_echo.push_str("τ (≈ 6.28319)");
                        continue;
                    }
                    "phi" => {
                        Self::push_token_with_implied(&mut tokens, Token::Number(1.618033988749895));
                        canonical_echo.push_str("φ (≈ 1.61803)");
                        continue;
                    }
                    _ => {
                        let word_lower = word.to_lowercase();
                        if word_lower == "atan2" || word_lower == "log10" || word_lower == "log2" {
                            Self::push_token_with_implied(&mut tokens, Token::Ident(word.clone()));
                            canonical_echo.push_str(&word);
                            continue;
                        }

                        // Check if word is an unparenthesized unary function attached to digits or constants
                        // Zero-relocation 1-byte bucketed CSR LUT (`344 B` L1d, `0` PE `.reloc` entries, `O(1)` first-byte dispatch)
                        const PREFIX_FUNCS_BLOB: &[u8] = concat!(
                            "arcsinh", "arccosh", "arctanh", "arcsin", "arccos", "arctan", "asin", "acos", "atan", "abs",
                            "bnot",
                            "cosh", "cbrt", "ceil", "clz", "ctz", "cos", "csc", "cot", "cb",
                            "deg",
                            "exp",
                            "fibonacci", "factorial", "factors", "factor", "floor", "fact", "fib", "fac",
                            "hypot",
                            "log10", "log2", "log", "ln",
                            "primes", "popcnt", "prime",
                            "round", "root", "rad", "rt",
                            "sinh", "sqrt", "sin", "sec", "sgn", "sq",
                            "tanh", "tan"
                        ).as_bytes();
                        const PREFIX_FUNCS_OFFSETS: [u16; 51] = [
                            0, 7, 14, 21, 27, 33, 39, 43, 47, 51,
                            54,
                            58, 62, 66, 70, 73, 76, 79, 82, 85,
                            87,
                            90,
                            93, 102, 111, 118, 124, 129, 133, 136,
                            139,
                            144, 149, 153, 156,
                            158, 164, 170,
                            175, 180, 184, 187,
                            189, 193, 197, 200, 203, 206,
                            208, 212, 215,
                        ];
                        const PREFIX_FUNCS_BUCKET: [u8; 27] = [
                            0,  // a (0..10)
                            10, // b (10..11)
                            11, // c (11..20)
                            20, // d (20..21)
                            21, // e (21..22)
                            22, // f (22..30)
                            30, // g (empty)
                            30, // h (30..31)
                            31, // i (empty)
                            31, // j (empty)
                            31, // k (empty)
                            31, // l (31..35)
                            35, // m (empty)
                            35, // n (empty)
                            35, // o (empty)
                            35, // p (35..38)
                            38, // q (empty)
                            38, // r (38..42)
                            42, // s (42..48)
                            48, // t (48..50)
                            50, 50, 50, 50, 50, 50, 50, // u..z + sentinel
                        ];

                        let mut handled_attached = false;
                        let wb = word_lower.as_bytes();
                        if let Some(&b0) = wb.first() && b0.is_ascii_lowercase() {
                            let bucket_i = (b0 - b'a') as usize;
                            let p_start = PREFIX_FUNCS_BUCKET[bucket_i] as usize;
                            let p_end = PREFIX_FUNCS_BUCKET[bucket_i + 1] as usize;
                            for p_idx in p_start..p_end {
                                let o0 = PREFIX_FUNCS_OFFSETS[p_idx] as usize;
                                let o1 = PREFIX_FUNCS_OFFSETS[p_idx + 1] as usize;
                                let pf_bytes = &PREFIX_FUNCS_BLOB[o0..o1];
                                if wb.len() > pf_bytes.len() && wb.starts_with(pf_bytes) {
                                    let pf = unsafe { core::str::from_utf8_unchecked(pf_bytes) };
                                    let rem = &word[pf.len()..];
                                    if rem.eq_ignore_ascii_case("pi") {
                                        Self::push_token_with_implied(&mut tokens, Token::Ident(pf.to_string()));
                                        tokens.push(Token::LParen);
                                        tokens.push(Token::Number(std::f64::consts::PI));
                                        tokens.push(Token::RParen);
                                        canonical_echo.push_str(pf);
                                        canonical_echo.push_str("(π (≈ 3.14159))");
                                        handled_attached = true;
                                        break;
                                    } else if rem.eq_ignore_ascii_case("e") {
                                        Self::push_token_with_implied(&mut tokens, Token::Ident(pf.to_string()));
                                        tokens.push(Token::LParen);
                                        tokens.push(Token::Number(std::f64::consts::E));
                                        tokens.push(Token::RParen);
                                        canonical_echo.push_str(pf);
                                        canonical_echo.push_str("(e (≈ 2.71828))");
                                        handled_attached = true;
                                        break;
                                    } else if rem.bytes().all(|c| c.is_ascii_digit() || c == b'.' || c == b'_') {
                                        Self::push_token_with_implied(&mut tokens, Token::Ident(pf.to_string()));
                                        tokens.push(Token::LParen);
                                        let norm = Self::normalize_number_punctuation(rem);
                                        if let Ok(n) = norm.parse::<i64>() {
                                            tokens.push(Token::Int(n));
                                        } else if let Some((m, s)) = super::bignum::parse_exact_decimal(&norm) {
                                            tokens.push(Token::ExactDecimal(m, s));
                                        } else if let Ok(f) = norm.parse::<f64>() {
                                            tokens.push(Token::Number(f));
                                        }
                                        tokens.push(Token::RParen);
                                        canonical_echo.push_str(pf);
                                        canonical_echo.push('(');
                                        canonical_echo.push_str(rem);
                                        canonical_echo.push(')');
                                        handled_attached = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if handled_attached {
                            continue;
                        }

                        // Function or Variable with implied multiplication
                        Self::push_token_with_implied(&mut tokens, Token::Ident(word.clone()));
                        canonical_echo.push_str(&word);
                        continue;
                    }
                }
            }

            i += 1;
        }

        tokens.push(Token::Eof);
        (tokens, canonical_echo)
    }

    /// Emits implied multiplication if preceding token was an operand/closer
    fn push_token_with_implied(tokens: &mut Vec<Token>, next: Token) {
        if let Some(prev) = tokens.last() {
            let prev_triggers = matches!(
                prev,
                Token::Int(_) | Token::ExactDecimal(_, _) | Token::Number(_) | Token::RParen | Token::RBracket | Token::Var(_)
            );
            let next_triggers = matches!(
                next,
                Token::Int(_) | Token::ExactDecimal(_, _) | Token::Number(_) | Token::LParen | Token::LBracket | Token::Ident(_) | Token::Var(_)
            );
            if prev_triggers && next_triggers {
                tokens.push(Token::Mul);
            }
        }
        tokens.push(next);
    }

    /// Smart Deterministic Number Punctuation Normalizer:
    /// Disambiguates Anglo-American (1,000.0) vs Continental European (1.000,0) by counting
    /// separator digit spans and relative ordering.
    pub fn normalize_number_punctuation(s: &str) -> String {
        let has_dot = s.contains('.');
        let has_comma = s.contains(',');

        if has_dot && has_comma {
            let first_dot = s.find('.').unwrap();
            let first_comma = s.find(',').unwrap();

            if first_comma < first_dot {
                // Anglo-American: 1,000.0 or 1,234,567.89 -> strip commas
                s.replace(',', "")
            } else {
                // Continental European: 1.000,0 or 1.234.567,89 -> strip dots, comma becomes dot
                s.replace('.', "").replace(',', ".")
            }
        } else if has_comma && !has_dot {
            let commas_count = s.chars().filter(|&c| c == ',').count();
            if commas_count > 1 {
                // Multiple commas: e.g. 1,000,000
                let last_comma = s.rfind(',').unwrap();
                let after_last = &s[last_comma + 1..];
                if after_last.len() == 3 {
                    // Standard thousands: 1,000,000 -> 1000000
                    s.replace(',', "")
                } else {
                    // Hybrid: 1,000,5 -> 1000.5
                    let prefix = &s[..last_comma].replace(',', "");
                    format!("{}.{}", prefix, after_last)
                }
            } else {
                // Single comma: 1,5 vs 1,000
                let comma_idx = s.find(',').unwrap();
                let after = &s[comma_idx + 1..];
                let before = &s[..comma_idx];
                // Exactly 3 digits after comma with non-zero prefix <= 3 digits: thousands grouping
                if after.len() == 3 && before.len() <= 3 && before != "0" {
                    s.replace(',', "")
                } else {
                    // Decimal comma: 1,5 -> 1.5; 0,25 -> 0.25; 3,14159 -> 3.14159
                    s.replace(',', ".")
                }
            }
        } else if has_dot && !has_comma {
            let dots_count = s.chars().filter(|&c| c == '.').count();
            if dots_count > 1 {
                // European thousands: 1.000.000
                let last_dot = s.rfind('.').unwrap();
                let after_last = &s[last_dot + 1..];
                if after_last.len() == 3 {
                    s.replace('.', "")
                } else {
                    let prefix = &s[..last_dot].replace('.', "");
                    format!("{}.{}", prefix, after_last)
                }
            } else {
                // Standard single decimal dot: 1.5, 3.14
                s.to_string()
            }
        } else {
            s.to_string()
        }
    }
}

fn desugar_root_char(s: String, sym: char, fn_name: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut cursor = 0;
    let sym_len = sym.len_utf8();
    while let Some(rel_idx) = s[cursor..].find(sym) {
        let idx = cursor + rel_idx;
        out.push_str(&s[cursor..idx]);
        let after = &s[idx + sym_len..];
        let trimmed_after = after.trim_start();
        let leading_spaces = after.len() - trimmed_after.len();
        let op_start = idx + sym_len + leading_spaces;

        if trimmed_after.starts_with('(') {
            let mut depth = 0;
            let mut close_idx = None;
            for (ci, ch) in trimmed_after.char_indices() {
                if ch == '(' {
                    depth += 1;
                } else if ch == ')' {
                    depth -= 1;
                    if depth == 0 {
                        close_idx = Some(ci);
                        break;
                    }
                }
            }
            if let Some(ci) = close_idx {
                let inner = &trimmed_after[1..ci];
                out.push_str(fn_name);
                out.push('(');
                out.push_str(inner);
                out.push(')');
                cursor = op_start + ci + 1;
                continue;
            }
        } else if trimmed_after.starts_with('{') {
            if let Some(ci) = trimmed_after.find('}') {
                let inner = &trimmed_after[1..ci];
                out.push_str(fn_name);
                out.push('(');
                out.push_str(inner);
                out.push(')');
                cursor = op_start + ci + 1;
                continue;
            }
        } else {
            let mut op_len = 0;
            for ch in trimmed_after.chars() {
                if ch.is_alphanumeric() || ch == '.' || ch == '_' {
                    op_len += ch.len_utf8();
                } else {
                    break;
                }
            }
            if op_len > 0 {
                let operand = &trimmed_after[..op_len];
                out.push_str(fn_name);
                out.push('(');
                out.push_str(operand);
                out.push(')');
                cursor = op_start + op_len;
                continue;
            }
        }

        out.push(' ');
        out.push_str(fn_name);
        out.push(' ');
        cursor = idx + sym_len;
    }
    out.push_str(&s[cursor..]);
    out
}

fn desugar_percentages(mut s: String) -> String {
    // 1. "X% of Y" -> "((X / 100) * (Y))"
    while let Some(of_pos) = s.to_lowercase().find(" of ") {
        let lhs = s[..of_pos].trim();
        let rhs = s[of_pos + 4..].trim();
        if lhs.ends_with('%') {
            let pct_val = lhs[..lhs.len() - 1].trim();
            s = format!("(({} / 100) * ({}))", pct_val, rhs);
            continue;
        }
        break;
    }

    // 2. Trailing/standalone "X%" -> "(X / 100)" when not used as binary modulo
    let mut out = String::with_capacity(s.len() + 16);
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if chars[i] == '%' {
            let prev_non_space = chars[..i].iter().rev().find(|c| !c.is_whitespace()).copied();
            let next_non_space = chars[i + 1..].iter().find(|c| !c.is_whitespace()).copied();

            let is_postfix_pct = match next_non_space {
                None => true,
                Some('+' | '-' | '*' | '/' | ')' | ',' | ';' | '=' | '>' | '<' | '&' | '|' | '^') => true,
                _ => false,
            };

            let is_preceded_by_operand = match prev_non_space {
                Some(c) => c.is_alphanumeric() || c == ')' || c == ']',
                None => false,
            };

            if is_postfix_pct && is_preceded_by_operand {
                out.push_str(" / 100");
                i += 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}
