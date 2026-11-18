//! High-performance unit conversion engine with Banking-Tier Exact Rational (`i128 / i128`) SI/NIST dispatch.
//! Supports Length, Mass, Temperature, Data, Time, Angle, Speed, Pressure, Energy, and Power.

use crate::engine::bignum::{decimal_add, decimal_div, decimal_mul, decimal_sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitFamily {
    Length,
    Mass,
    Temperature,
    Data,
    Time,
    Angle,
    Speed,
    Pressure,
    Energy,
    Power,
}

pub struct UnitDef {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub family: UnitFamily,
    pub to_base: f64,
    /// Exact rational `(num, den)` multiplier to the family SI base unit (`None` only for transcendental `π` rad/deg).
    pub exact_ratio: Option<(i128, i128)>,
}

const UNITS: &[UnitDef] = &[
    // --- Length (base: meter) ---
    UnitDef { name: "m", aliases: &["meter", "meters", "metre", "metres"], family: UnitFamily::Length, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "km", aliases: &["kilometer", "kilometers", "kilometre", "kilometres"], family: UnitFamily::Length, to_base: 1_000.0, exact_ratio: Some((1_000, 1)) },
    UnitDef { name: "cm", aliases: &["centimeter", "centimeters", "centimetre", "centimetres"], family: UnitFamily::Length, to_base: 0.01, exact_ratio: Some((1, 100)) },
    UnitDef { name: "mm", aliases: &["millimeter", "millimeters", "millimetre", "millimetres"], family: UnitFamily::Length, to_base: 0.001, exact_ratio: Some((1, 1_000)) },
    UnitDef { name: "um", aliases: &["micrometer", "micrometers", "micron", "microns", "µm"], family: UnitFamily::Length, to_base: 1e-6, exact_ratio: Some((1, 1_000_000)) },
    UnitDef { name: "nm", aliases: &["nanometer", "nanometers"], family: UnitFamily::Length, to_base: 1e-9, exact_ratio: Some((1, 1_000_000_000)) },
    UnitDef { name: "in", aliases: &["inch", "inches"], family: UnitFamily::Length, to_base: 0.0254, exact_ratio: Some((254, 10_000)) },
    UnitDef { name: "ft", aliases: &["foot", "feet"], family: UnitFamily::Length, to_base: 0.3048, exact_ratio: Some((3_048, 10_000)) },
    UnitDef { name: "yd", aliases: &["yard", "yards"], family: UnitFamily::Length, to_base: 0.9144, exact_ratio: Some((9_144, 10_000)) },
    UnitDef { name: "mi", aliases: &["mile", "miles"], family: UnitFamily::Length, to_base: 1609.344, exact_ratio: Some((1_609_344, 1_000)) },
    UnitDef { name: "nmi", aliases: &["nauticalmile", "nauticalmiles"], family: UnitFamily::Length, to_base: 1852.0, exact_ratio: Some((1_852, 1)) },
    UnitDef { name: "ly", aliases: &["lightyear", "lightyears"], family: UnitFamily::Length, to_base: 9.4607304725808e15, exact_ratio: Some((9_460_730_472_580_800, 1)) },

    // --- Mass / Weight (base: kilogram) ---
    UnitDef { name: "kg", aliases: &["kilogram", "kilograms", "kilo", "kilos"], family: UnitFamily::Mass, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "g", aliases: &["gram", "grams"], family: UnitFamily::Mass, to_base: 0.001, exact_ratio: Some((1, 1_000)) },
    UnitDef { name: "mg", aliases: &["milligram", "milligrams"], family: UnitFamily::Mass, to_base: 1e-6, exact_ratio: Some((1, 1_000_000)) },
    UnitDef { name: "ug", aliases: &["microgram", "micrograms", "µg"], family: UnitFamily::Mass, to_base: 1e-9, exact_ratio: Some((1, 1_000_000_000)) },
    UnitDef { name: "ton", aliases: &["tonne", "tonnes", "tons", "metricton"], family: UnitFamily::Mass, to_base: 1_000.0, exact_ratio: Some((1_000, 1)) },
    UnitDef { name: "lb", aliases: &["lbs", "pound", "pounds"], family: UnitFamily::Mass, to_base: 0.45359237, exact_ratio: Some((45_359_237, 100_000_000)) },
    UnitDef { name: "oz", aliases: &["ounce", "ounces"], family: UnitFamily::Mass, to_base: 0.028349523125, exact_ratio: Some((45_359_237, 1_600_000_000)) },

    // --- Temperature (handled via exact affine transformations) ---
    UnitDef { name: "c", aliases: &["degc", "celsius", "centigrade", "°c"], family: UnitFamily::Temperature, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "f", aliases: &["degf", "fahrenheit", "°f"], family: UnitFamily::Temperature, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "k", aliases: &["kelvin", "degk", "°k"], family: UnitFamily::Temperature, to_base: 1.0, exact_ratio: Some((1, 1)) },

    // --- Digital Storage / Data (base: bytes) ---
    UnitDef { name: "b", aliases: &["byte", "bytes"], family: UnitFamily::Data, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "kb", aliases: &["kilobyte", "kilobytes"], family: UnitFamily::Data, to_base: 1_000.0, exact_ratio: Some((1_000, 1)) },
    UnitDef { name: "mb", aliases: &["megabyte", "megabytes"], family: UnitFamily::Data, to_base: 1_000_000.0, exact_ratio: Some((1_000_000, 1)) },
    UnitDef { name: "gb", aliases: &["gigabyte", "gigabytes"], family: UnitFamily::Data, to_base: 1_000_000_000.0, exact_ratio: Some((1_000_000_000, 1)) },
    UnitDef { name: "tb", aliases: &["terabyte", "terabytes"], family: UnitFamily::Data, to_base: 1_000_000_000_000.0, exact_ratio: Some((1_000_000_000_000, 1)) },
    UnitDef { name: "pb", aliases: &["petabyte", "petabytes"], family: UnitFamily::Data, to_base: 1_000_000_000_000_000.0, exact_ratio: Some((1_000_000_000_000_000, 1)) },
    UnitDef { name: "kib", aliases: &["kibibyte", "kibibytes"], family: UnitFamily::Data, to_base: 1024.0, exact_ratio: Some((1_024, 1)) },
    UnitDef { name: "mib", aliases: &["mebibyte", "mebibytes"], family: UnitFamily::Data, to_base: 1048576.0, exact_ratio: Some((1_048_576, 1)) },
    UnitDef { name: "gib", aliases: &["gibibyte", "gibibytes"], family: UnitFamily::Data, to_base: 1073741824.0, exact_ratio: Some((1_073_741_824, 1)) },
    UnitDef { name: "tib", aliases: &["tebibyte", "tebibytes"], family: UnitFamily::Data, to_base: 1099511627776.0, exact_ratio: Some((1_099_511_627_776, 1)) },

    // --- Time (base: second) ---
    UnitDef { name: "s", aliases: &["sec", "secs", "second", "seconds"], family: UnitFamily::Time, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "ms", aliases: &["millisecond", "milliseconds"], family: UnitFamily::Time, to_base: 0.001, exact_ratio: Some((1, 1_000)) },
    UnitDef { name: "us", aliases: &["microsecond", "microseconds", "µs"], family: UnitFamily::Time, to_base: 1e-6, exact_ratio: Some((1, 1_000_000)) },
    UnitDef { name: "ns", aliases: &["nanosecond", "nanoseconds"], family: UnitFamily::Time, to_base: 1e-9, exact_ratio: Some((1, 1_000_000_000)) },
    UnitDef { name: "min", aliases: &["minute", "minutes"], family: UnitFamily::Time, to_base: 60.0, exact_ratio: Some((60, 1)) },
    UnitDef { name: "h", aliases: &["hr", "hrs", "hour", "hours"], family: UnitFamily::Time, to_base: 3600.0, exact_ratio: Some((3_600, 1)) },
    UnitDef { name: "d", aliases: &["day", "days"], family: UnitFamily::Time, to_base: 86400.0, exact_ratio: Some((86_400, 1)) },
    UnitDef { name: "week", aliases: &["weeks", "wk", "wks"], family: UnitFamily::Time, to_base: 604800.0, exact_ratio: Some((604_800, 1)) },
    UnitDef { name: "yr", aliases: &["year", "years"], family: UnitFamily::Time, to_base: 31536000.0, exact_ratio: Some((31_536_000, 1)) },

    // --- Angle (base: radian; deg <-> grad is exact 10/9, rad involves transcendental π) ---
    UnitDef { name: "rad", aliases: &["radian", "radians"], family: UnitFamily::Angle, to_base: 1.0, exact_ratio: None },
    UnitDef { name: "deg", aliases: &["degree", "degrees", "°"], family: UnitFamily::Angle, to_base: std::f64::consts::PI / 180.0, exact_ratio: None },
    UnitDef { name: "grad", aliases: &["gradian", "gradians", "gon"], family: UnitFamily::Angle, to_base: std::f64::consts::PI / 200.0, exact_ratio: None },

    // --- Speed (base: m/s) ---
    UnitDef { name: "mps", aliases: &["m/s", "meterpersecond"], family: UnitFamily::Speed, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "kph", aliases: &["km/h", "kmh", "kilometerperhour"], family: UnitFamily::Speed, to_base: 1.0 / 3.6, exact_ratio: Some((5, 18)) },
    UnitDef { name: "mph", aliases: &["mi/h", "mileperhour"], family: UnitFamily::Speed, to_base: 0.44704, exact_ratio: Some((44_704, 100_000)) },
    UnitDef { name: "knot", aliases: &["knots", "kt", "kts"], family: UnitFamily::Speed, to_base: 1852.0 / 3600.0, exact_ratio: Some((463, 900)) },

    // --- Pressure (base: Pascal) ---
    UnitDef { name: "pa", aliases: &["pascal", "pascals"], family: UnitFamily::Pressure, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "kpa", aliases: &["kilopascal", "kilopascals"], family: UnitFamily::Pressure, to_base: 1000.0, exact_ratio: Some((1_000, 1)) },
    UnitDef { name: "bar", aliases: &["bars"], family: UnitFamily::Pressure, to_base: 100000.0, exact_ratio: Some((100_000, 1)) },
    UnitDef { name: "psi", aliases: &["poundsperforce"], family: UnitFamily::Pressure, to_base: 6894.757293168, exact_ratio: None },
    UnitDef { name: "atm", aliases: &["atmosphere", "atmospheres"], family: UnitFamily::Pressure, to_base: 101325.0, exact_ratio: Some((101_325, 1)) },
    UnitDef { name: "torr", aliases: &["mmhg"], family: UnitFamily::Pressure, to_base: 133.322368421, exact_ratio: Some((101_325, 760)) },

    // --- Energy (base: Joule) ---
    UnitDef { name: "j", aliases: &["joule", "joules"], family: UnitFamily::Energy, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "kj", aliases: &["kilojoule", "kilojoules"], family: UnitFamily::Energy, to_base: 1_000.0, exact_ratio: Some((1_000, 1)) },
    UnitDef { name: "cal", aliases: &["calorie", "calories"], family: UnitFamily::Energy, to_base: 4.184, exact_ratio: Some((4_184, 1_000)) },
    UnitDef { name: "kcal", aliases: &["kilocalorie", "kilocalories"], family: UnitFamily::Energy, to_base: 4184.0, exact_ratio: Some((4_184, 1)) },
    UnitDef { name: "wh", aliases: &["watthour", "watthours"], family: UnitFamily::Energy, to_base: 3600.0, exact_ratio: Some((3_600, 1)) },
    UnitDef { name: "kwh", aliases: &["kilowatthour", "kilowatthours"], family: UnitFamily::Energy, to_base: 3.6e6, exact_ratio: Some((3_600_000, 1)) },
    UnitDef { name: "btu", aliases: &["btus"], family: UnitFamily::Energy, to_base: 1055.056, exact_ratio: Some((1_055_056, 1_000)) },
    UnitDef { name: "ev", aliases: &["electronvolt", "electronvolts"], family: UnitFamily::Energy, to_base: 1.602176634e-19, exact_ratio: None },

    // --- Power (base: Watt) ---
    UnitDef { name: "w", aliases: &["watt", "watts"], family: UnitFamily::Power, to_base: 1.0, exact_ratio: Some((1, 1)) },
    UnitDef { name: "kw", aliases: &["kilowatt", "kilowatts"], family: UnitFamily::Power, to_base: 1_000.0, exact_ratio: Some((1_000, 1)) },
    UnitDef { name: "mw", aliases: &["megawatt", "megawatts"], family: UnitFamily::Power, to_base: 1_000_000.0, exact_ratio: Some((1_000_000, 1)) },
    UnitDef { name: "hp", aliases: &["horsepower", "mechanicalhorsepower"], family: UnitFamily::Power, to_base: 745.699872, exact_ratio: Some((745_699_872, 1_000_000)) },
];

pub fn find_unit(raw: &str) -> Option<&'static UnitDef> {
    let lower = raw.trim().to_lowercase();
    for u in UNITS {
        if u.name == lower || u.aliases.iter().any(|&a| a == lower) {
            return Some(u);
        }
    }
    None
}

/// Banking-Tier Exact Rational (`i128 / i128`) unit conversion (`0` IEEE-754 binary float drift).
/// Returns `Ok(Some((mantissa, scale)))` whenever both units have exact rational definitions and
/// do not overflow `i128`; returns `Ok(None)` when transcendental (`rad <-> deg`) or exceeding `i128`.
pub fn convert_units_exact(
    mantissa: i128,
    scale: u32,
    from_str: &str,
    to_str: &str,
) -> Result<Option<(i128, u32)>, String> {
    let u_from = find_unit(from_str).ok_or_else(|| format!("Unknown unit: '{}'", from_str))?;
    let u_to = find_unit(to_str).ok_or_else(|| format!("Unknown unit: '{}'", to_str))?;

    if u_from.family != u_to.family {
        return Err(format!(
            "Incompatible unit families: cannot convert {:?} ({}) to {:?} ({})",
            u_from.family, from_str, u_to.family, to_str
        ));
    }
    if u_from.name == u_to.name {
        return Ok(Some((mantissa, scale)));
    }

    if u_from.family == UnitFamily::Temperature {
        let (c_m, c_s) = match u_from.name {
            "c" => (mantissa, scale),
            "f" => {
                let Ok((sub_m, sub_s)) = decimal_sub(mantissa, scale, 32, 0) else { return Ok(None); };
                let Ok((mul_m, mul_s)) = decimal_mul(sub_m, sub_s, 5, 0) else { return Ok(None); };
                let Ok(res) = decimal_div(mul_m, mul_s, 9, 0) else { return Ok(None); };
                res
            }
            "k" => {
                let Ok(res) = decimal_sub(mantissa, scale, 27_315, 2) else { return Ok(None); };
                res
            }
            _ => (mantissa, scale),
        };
        let out = match u_to.name {
            "c" => (c_m, c_s),
            "f" => {
                let Ok((mul_m, mul_s)) = decimal_mul(c_m, c_s, 9, 0) else { return Ok(None); };
                let Ok((div_m, div_s)) = decimal_div(mul_m, mul_s, 5, 0) else { return Ok(None); };
                let Ok(res) = decimal_add(div_m, div_s, 32, 0) else { return Ok(None); };
                res
            }
            "k" => {
                let Ok(res) = decimal_add(c_m, c_s, 27_315, 2) else { return Ok(None); };
                res
            }
            _ => (c_m, c_s),
        };
        return Ok(Some(out));
    }

    // Exact deg <-> grad conversion (180 deg == 200 grad => 10/9) without π!
    if u_from.family == UnitFamily::Angle {
        if u_from.name == "deg" && u_to.name == "grad" {
            if let Ok((m1, s1)) = decimal_mul(mantissa, scale, 10, 0) {
                if let Ok(res) = decimal_div(m1, s1, 9, 0) {
                    return Ok(Some(res));
                }
            }
        } else if u_from.name == "grad" && u_to.name == "deg" {
            if let Ok((m1, s1)) = decimal_mul(mantissa, scale, 9, 0) {
                if let Ok(res) = decimal_div(m1, s1, 10, 0) {
                    return Ok(Some(res));
                }
            }
        }
        return Ok(None);
    }

    let (Some((n1, d1)), Some((n2, d2))) = (u_from.exact_ratio, u_to.exact_ratio) else {
        return Ok(None);
    };
    let Some(num_ratio) = n1.checked_mul(d2) else { return Ok(None); };
    let Some(den_ratio) = d1.checked_mul(n2) else { return Ok(None); };
    let Ok((m_num, s_num)) = decimal_mul(mantissa, scale, num_ratio, 0) else { return Ok(None); };
    let Ok(res) = decimal_div(m_num, s_num, den_ratio, 0) else { return Ok(None); };
    Ok(Some(res))
}

pub fn convert_units(val: f64, from_str: &str, to_str: &str) -> Result<f64, String> {
    let u_from = find_unit(from_str).ok_or_else(|| format!("Unknown unit: '{}'", from_str))?;
    let u_to = find_unit(to_str).ok_or_else(|| format!("Unknown unit: '{}'", to_str))?;

    if u_from.family != u_to.family {
        return Err(format!(
            "Incompatible unit families: cannot convert {:?} ({}) to {:?} ({})",
            u_from.family, from_str, u_to.family, to_str
        ));
    }

    if u_from.family == UnitFamily::Temperature {
        let c_val = match u_from.name {
            "c" => val,
            "f" => (val - 32.0) * (5.0 / 9.0),
            "k" => val - 273.15,
            _ => val,
        };
        let target_val = match u_to.name {
            "c" => c_val,
            "f" => (c_val * (9.0 / 5.0)) + 32.0,
            "k" => c_val + 273.15,
            _ => c_val,
        };
        return Ok(target_val);
    }

    let base_val = val * u_from.to_base;
    let target_val = base_val / u_to.to_base;
    Ok(target_val)
}
