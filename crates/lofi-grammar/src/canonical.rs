//! The canonical JSON representation shared with the Python boundary.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::ser::Formatter;

use crate::{GrammarIr, LanguageIr};

/// Serialize a language IR using Python's compact ASCII canonical spelling.
///
/// # Errors
///
/// Returns a JSON error if an interior declaration value cannot be encoded.
pub fn language_bytes(value: &LanguageIr) -> Result<Vec<u8>, serde_json::Error> {
    to_vec(value)
}

/// Serialize a grammar IR using Python's compact ASCII canonical spelling.
///
/// # Errors
///
/// Returns a JSON error if an interior declaration value cannot be encoded.
pub fn grammar_bytes(value: &GrammarIr) -> Result<Vec<u8>, serde_json::Error> {
    to_vec(value)
}

/// Serialize an interior value with the declaration boundary's canonical writer.
pub(crate) fn value_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    to_vec(value)
}

fn to_vec<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let sorted = serde_json::to_value(value)?;
    let mut output = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut output, AsciiFormatter);
    sorted.serialize(&mut serializer)?;
    Ok(output)
}

struct AsciiFormatter;

impl Formatter for AsciiFormatter {
    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        for character in fragment.chars() {
            let point = u32::from(character);
            if point <= 0x7f {
                writer.write_all(&[u8::try_from(point)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?])?;
            } else if point <= 0xffff {
                write!(writer, "\\u{point:04x}")?;
            } else {
                let scalar = point - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                write!(writer, "\\u{high:04x}\\u{low:04x}")?;
            }
        }
        Ok(())
    }

    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "canonical JSON requires a finite float",
            ));
        }

        let mut buffer = ryu::Buffer::new();
        let spelling = buffer.format_finite(value);
        write_python_float(writer, spelling)
    }
}

fn write_python_float<W>(writer: &mut W, spelling: &str) -> io::Result<()>
where
    W: ?Sized + Write,
{
    if matches!(spelling, "0.0" | "-0.0") {
        return writer.write_all(spelling.as_bytes());
    }

    let (sign, unsigned) = spelling
        .strip_prefix('-')
        .map_or(("", spelling), |unsigned| ("-", unsigned));
    let (mantissa, explicit_exponent) = if let Some((mantissa, exponent)) = unsigned.split_once('e')
    {
        let exponent = exponent
            .parse::<i32>()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        (mantissa, exponent)
    } else {
        (unsigned, 0)
    };
    let decimal = mantissa.find('.').unwrap_or(mantissa.len());
    let first_nonzero = mantissa
        .bytes()
        .position(|byte| byte != b'0' && byte != b'.')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "float has no digit"))?;
    let decimal = i32::try_from(decimal)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let first_nonzero = i32::try_from(first_nonzero)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let correction = if first_nonzero < decimal {
        decimal - first_nonzero - 1
    } else {
        decimal - first_nonzero
    };
    let exponent = explicit_exponent + correction;
    let digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let trimmed = digits.trim_start_matches('0').trim_end_matches('0');
    let digits = if trimmed.is_empty() { "0" } else { trimmed };

    let mut python = String::from(sign);
    if (-4..16).contains(&exponent) {
        let decimal = exponent + 1;
        if decimal <= 0 {
            python.push_str("0.");
            for _ in decimal..0 {
                python.push('0');
            }
            python.push_str(digits);
        } else {
            let decimal = usize::try_from(decimal)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            if decimal >= digits.len() {
                python.push_str(digits);
                for _ in digits.len()..decimal {
                    python.push('0');
                }
                python.push_str(".0");
            } else {
                python.push_str(&digits[..decimal]);
                python.push('.');
                python.push_str(&digits[decimal..]);
            }
        }
    } else {
        let mut characters = digits.chars();
        let first = characters
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "float has no digit"))?;
        python.push(first);
        let remainder: String = characters.collect();
        if !remainder.is_empty() {
            python.push('.');
            python.push_str(&remainder);
        }
        python.push('e');
        python.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            python.push('0');
        }
        python.push_str(&magnitude.to_string());
    }
    writer.write_all(python.as_bytes())
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use indexmap::IndexMap;
    use proptest::prelude::*;
    use serde::Serialize;
    use serde_json::ser::CompactFormatter;

    use super::to_vec;

    #[derive(Serialize)]
    struct CanonicalWitness<'a> {
        schema: &'a str,
        labels: IndexMap<&'a str, f64>,
    }

    #[test]
    fn canonical_json_is_compact_ascii_with_sorted_keys() -> Result<(), Box<dyn Error>> {
        let labels = IndexMap::from([("zeta", 1.0), ("snowman-☃", 1.0e-5), ("face-😀", 1.0e20)]);
        let encoded = to_vec(&CanonicalWitness {
            schema: "decl-😀",
            labels,
        })?;

        assert_eq!(
            encoded,
            br#"{"labels":{"face-\ud83d\ude00":1e+20,"snowman-\u2603":1e-05,"zeta":1.0},"schema":"decl-\ud83d\ude00"}"#
        );
        Ok(())
    }

    #[test]
    fn finite_float_spellings_match_python_json() -> Result<(), Box<dyn Error>> {
        let values = [
            0.0, -0.0, 1.0, 1.5, 1.0e-5, 1.0e-4, 1.0e15, 1.0e16, 1.2345e16, 1.2e9, 1.2e-7,
        ];

        assert_eq!(
            to_vec(&values)?,
            b"[0.0,-0.0,1.0,1.5,1e-05,0.0001,1000000000000000.0,1e+16,1.2345e+16,1200000000.0,1.2e-07]"
        );
        Ok(())
    }

    proptest! {
        #[test]
        fn canonical_strings_are_ascii(value in any::<String>()) {
            let encoded = to_vec(&value);
            prop_assert!(encoded.is_ok(), "every Rust string must be JSON text");
            if let Ok(encoded) = encoded {
                prop_assert!(encoded.is_ascii());
                let decoded = serde_json::from_slice::<String>(&encoded);
                prop_assert!(decoded.is_ok(), "canonical text must decode");
                if let Ok(decoded) = decoded {
                    prop_assert_eq!(decoded, value);
                }
            }
        }

        #[test]
        fn finite_floats_round_trip(value in any::<f64>().prop_filter(
            "canonical JSON carries only finite floats",
            |value| value.is_finite(),
        )) {
            let encoded = to_vec(&value);
            prop_assert!(encoded.is_ok(), "a finite float must be encodable");
            if let Ok(encoded) = encoded {
                let decoded = serde_json::from_slice::<f64>(&encoded);
                prop_assert!(decoded.is_ok(), "canonical float spelling must be JSON");
                if let Ok(decoded) = decoded {
                    prop_assert_eq!(decoded.to_bits(), value.to_bits());
                }
            }
        }
    }

    #[test]
    fn compact_formatter_remains_compact_for_ascii_scalars() -> Result<(), Box<dyn Error>> {
        let mut output = Vec::new();
        let mut serializer = serde_json::Serializer::with_formatter(&mut output, CompactFormatter);
        CanonicalWitness {
            schema: "decl-1",
            labels: IndexMap::new(),
        }
        .serialize(&mut serializer)?;

        assert_eq!(output, br#"{"schema":"decl-1","labels":{}}"#);
        Ok(())
    }
}
