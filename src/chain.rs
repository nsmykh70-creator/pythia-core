use crate::{domain::Domain, model::{ReversibleModel, TransformOutput}, transforms::builtin_model, ModelId, UmceError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub model_id: ModelId,
    pub model_version: u16,
    /// Canonical serialized parameters. Empty for the initial built-ins.
    pub parameters: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainResult {
    pub data: Vec<u8>,
    /// Auxiliary data aligned one-to-one with stages in forward order.
    pub auxiliary: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelChain {
    stages: Vec<Stage>,
}

impl ModelChain {
    pub fn new(stages: Vec<Stage>) -> Self { Self { stages } }
    pub fn stages(&self) -> &[Stage] { &self.stages }
    pub fn is_empty(&self) -> bool { self.stages.is_empty() }

    /// Validate every model boundary and the externally declared chain domains.
    pub fn validate_domains(&self, input: Domain, output: Domain) -> Result<(), UmceError> {
        let mut current = input;
        for stage in &self.stages {
            let model = resolve(stage)?;
            if model.input_domain() != current {
                return Err(UmceError::DomainMismatch {
                    expected: crate::domain::domain_name(model.input_domain()),
                    actual: crate::domain::domain_name(current),
                });
            }
            current = model.output_domain();
        }
        if current != output {
            return Err(UmceError::DomainMismatch {
                expected: crate::domain::domain_name(output),
                actual: crate::domain::domain_name(current),
            });
        }
        Ok(())
    }

    pub fn forward(&self, input: &[u8]) -> Result<ChainResult, UmceError> {
        let mut data = input.to_vec();
        let mut auxiliary = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let model = resolve(stage)?;
            let result = model.forward(&data, &stage.parameters)?;
            data = result.data;
            auxiliary.push(result.auxiliary);
        }
        Ok(ChainResult { data, auxiliary })
    }

    pub fn inverse(&self, result: &ChainResult) -> Result<Vec<u8>, UmceError> {
        if result.auxiliary.len() != self.stages.len() {
            return Err(UmceError::LengthMismatch { expected: self.stages.len(), actual: result.auxiliary.len() });
        }
        let mut data = result.data.clone();
        for (stage, auxiliary) in self.stages.iter().zip(result.auxiliary.iter()).rev() {
            let model = resolve(stage)?;
            data = model.inverse(&data, &stage.parameters, auxiliary)?;
        }
        Ok(data)
    }
}

fn resolve(stage: &Stage) -> Result<Box<dyn ReversibleModel>, UmceError> {
    let model = builtin_model(stage.model_id).ok_or(UmceError::UnknownModel(stage.model_id))?;
    if model.version() != stage.model_version {
        return Err(UmceError::InvalidParameters("unsupported model version"));
    }
    Ok(model)
}

impl From<TransformOutput> for ChainResult {
    fn from(value: TransformOutput) -> Self { Self { data: value.data, auxiliary: vec![value.auxiliary] } }
}
