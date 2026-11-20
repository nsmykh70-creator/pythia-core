//! Explicit data-domain contracts for reversible models.
use crate::UmceError;

/// Semantic representation expected at a model boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Domain {
    /// Arbitrary bytes with no additional interpretation.
    Bytes = 0,
    /// A byte stream interpreted as unsigned scalar samples.
    U8Samples = 1,
    /// A byte stream interpreted as signed 8-bit scalar samples.
    I8Samples = 2,
    /// A byte stream containing little-endian 16-bit words.
    U16LeWords = 3,
    /// A byte stream containing little-endian 32-bit words.
    U32LeWords = 4,
    /// A byte stream representing a sequence of fixed-width records.
    FixedRecords = 5,
}

impl Domain {
    pub fn from_tag(tag: u8) -> Result<Self, UmceError> {
        match tag {
            0 => Ok(Self::Bytes), 1 => Ok(Self::U8Samples), 2 => Ok(Self::I8Samples),
            3 => Ok(Self::U16LeWords), 4 => Ok(Self::U32LeWords), 5 => Ok(Self::FixedRecords),
            _ => Err(UmceError::InvalidDescriptor("unknown domain tag")),
        }
    }

    pub fn tag(self) -> u8 { self as u8 }
}

/// A checked edge in a transform graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainContract {
    pub input: Domain,
    pub output: Domain,
}

impl DomainContract {
    pub const fn new(input: Domain, output: Domain) -> Self { Self { input, output } }

    pub fn ensure_follows(self, previous: Self) -> Result<(), UmceError> {
        if previous.output == self.input { Ok(()) }
        else { Err(UmceError::DomainMismatch { expected: domain_name(self.input), actual: domain_name(previous.output) }) }
    }
}

pub const fn domain_name(d: Domain) -> &'static str {
    match d {
        Domain::Bytes => "bytes", Domain::U8Samples => "u8_samples", Domain::I8Samples => "i8_samples",
        Domain::U16LeWords => "u16le_words", Domain::U32LeWords => "u32le_words", Domain::FixedRecords => "fixed_records",
    }
}
