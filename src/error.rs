use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UmceError {
    UnknownModel(u32),
    InvalidParameters(&'static str),
    InvalidAuxiliary(&'static str),
    DomainMismatch { expected: &'static str, actual: &'static str },
    LengthMismatch { expected: usize, actual: usize },
    LengthOverflow,
    InvalidDescriptor(&'static str),
}

impl fmt::Display for UmceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownModel(id) => write!(f, "unknown model id: {id}"),
            Self::InvalidParameters(s) => write!(f, "invalid parameters: {s}"),
            Self::InvalidAuxiliary(s) => write!(f, "invalid auxiliary data: {s}"),
            Self::DomainMismatch { expected, actual } => {
                write!(f, "domain mismatch: expected {expected}, got {actual}")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
            Self::LengthOverflow => write!(f, "length does not fit platform usize"),
            Self::InvalidDescriptor(s) => write!(f, "invalid chain descriptor: {s}"),
        }
    }
}

impl std::error::Error for UmceError {}
