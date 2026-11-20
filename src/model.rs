use crate::{domain::Domain, UmceError};

pub type ModelId = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformOutput {
    pub data: Vec<u8>,
    /// Per-invocation side information required by inverse(). Empty for PS6.1 models.
    pub auxiliary: Vec<u8>,
}

/// Contract for a deterministic, byte-domain reversible transform.
///
/// Implementations must not depend on platform endianness, randomized state,
/// unrecorded global state, or floating-point behavior.
pub trait ReversibleModel: Send + Sync {
    fn id(&self) -> ModelId;
    fn version(&self) -> u16 { 1 }
    fn name(&self) -> &'static str;
    fn input_domain(&self) -> Domain { Domain::Bytes }
    fn output_domain(&self) -> Domain { Domain::Bytes }
    fn validate_parameters(&self, params: &[u8]) -> Result<(), UmceError>;
    fn forward(&self, input: &[u8], params: &[u8]) -> Result<TransformOutput, UmceError>;
    fn inverse(
        &self,
        output: &[u8],
        params: &[u8],
        auxiliary: &[u8],
    ) -> Result<Vec<u8>, UmceError>;
}

pub(crate) fn require_empty_params(params: &[u8]) -> Result<(), UmceError> {
    if params.is_empty() { Ok(()) } else { Err(UmceError::InvalidParameters("model takes no parameters")) }
}

pub(crate) fn require_empty_aux(aux: &[u8]) -> Result<(), UmceError> {
    if aux.is_empty() { Ok(()) } else { Err(UmceError::InvalidAuxiliary("unexpected auxiliary bytes")) }
}
