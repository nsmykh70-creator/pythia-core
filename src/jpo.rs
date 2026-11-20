//! Joint Parameter Optimization (JPO), PS6.4.
//!
//! Deterministic coordinate descent over caller-supplied canonical parameter
//! candidates. Every candidate is evaluated using exact complete-archive cost.
//! This is a bounded optimizer, not a proof of global optimality.

use crate::{chain::Stage, cost::ArchiveCost, UmceError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpoMetrics {
    pub archive: ArchiveCost,
    pub encode_time_ns: u64,
    pub peak_memory_bytes: u64,
}

impl JpoMetrics {
    pub fn bytes(self) -> Result<u64, UmceError> { self.archive.total() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterDomain {
    /// Candidate canonical byte encodings for this stage's parameters.
    /// The current parameter value should be included if it must remain eligible.
    pub candidates: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpoLimits {
    /// Maximum complete coordinate-descent passes.
    pub max_passes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpoResult {
    pub stages: Vec<Stage>,
    pub metrics: JpoMetrics,
    pub evaluations: usize,
    pub passes: usize,
    pub converged: bool,
}

/// Optimize stage parameters with exact evaluation after each coordinate change.
/// The evaluator must include transformed payload, entropy coding, descriptors,
/// auxiliary data, model bank, framing, checksums and index in `archive`.
pub fn optimize_parameters<F>(
    initial: &[Stage],
    domains: &[ParameterDomain],
    limits: JpoLimits,
    mut evaluate: F,
) -> Result<JpoResult, UmceError>
where
    F: FnMut(&[Stage]) -> Result<JpoMetrics, UmceError>,
{
    if initial.len() != domains.len() {
        return Err(UmceError::LengthMismatch { expected: initial.len(), actual: domains.len() });
    }
    if limits.max_passes == 0 { return Err(UmceError::InvalidParameters("JPO max_passes must be non-zero")); }
    if domains.iter().any(|d| d.candidates.is_empty()) {
        return Err(UmceError::InvalidParameters("each JPO parameter domain must contain at least one candidate"));
    }

    let mut stages = initial.to_vec();
    let mut metrics = evaluate(&stages)?;
    let mut evaluations = 1usize;
    let mut passes = 0usize;
    let mut converged = false;

    for pass in 0..limits.max_passes {
        let mut changed = false;
        for index in 0..stages.len() {
            let mut best_params = stages[index].parameters.clone();
            let mut best_metrics = metrics;
            for candidate in &domains[index].candidates {
                let mut trial = stages.clone();
                trial[index].parameters = candidate.clone();
                let trial_metrics = evaluate(&trial)?;
                evaluations = evaluations.checked_add(1).ok_or(UmceError::LengthOverflow)?;
                if metric_key(trial_metrics)? < metric_key(best_metrics)? {
                    best_params = candidate.clone();
                    best_metrics = trial_metrics;
                }
            }
            if best_params != stages[index].parameters {
                stages[index].parameters = best_params;
                metrics = best_metrics;
                changed = true;
            }
        }
        passes = pass + 1;
        if !changed { converged = true; break; }
    }
    Ok(JpoResult { stages, metrics, evaluations, passes, converged })
}

/// Exact size is primary. Time and RAM break ties only at equal byte size.
fn metric_key(m: JpoMetrics) -> Result<(u64, u64, u64), UmceError> {
    Ok((m.bytes()?, m.encode_time_ns, m.peak_memory_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(id: u32, p: u8) -> Stage { Stage { model_id: id, model_version: 1, parameters: vec![p] } }
    fn m(n: u64) -> JpoMetrics { JpoMetrics { archive: ArchiveCost { payload: n, ..ArchiveCost::default() }, encode_time_ns: n, peak_memory_bytes: n } }

    #[test]
    fn coordinate_descent_uses_exact_archive_size() {
        let initial = [s(1, 0), s(2, 0)];
        let domains = [ParameterDomain { candidates: vec![vec![0], vec![1]] }, ParameterDomain { candidates: vec![vec![0], vec![1]] }];
        let result = optimize_parameters(&initial, &domains, JpoLimits { max_passes: 5 }, |stages| {
            let a = stages[0].parameters[0]; let b = stages[1].parameters[0];
            Ok(m(match (a, b) { (1, 1) => 50, (1, 0) | (0, 1) => 70, _ => 100 }))
        }).unwrap();
        assert_eq!(result.metrics.bytes().unwrap(), 50);
        assert_eq!(result.stages[0].parameters, vec![1]);
        assert_eq!(result.stages[1].parameters, vec![1]);
    }

    #[test]
    fn exact_size_beats_faster_but_larger_candidate() {
        let initial = [s(1, 0)];
        let domains = [ParameterDomain { candidates: vec![vec![0], vec![1]] }];
        let result = optimize_parameters(&initial, &domains, JpoLimits { max_passes: 2 }, |stages| {
            Ok(if stages[0].parameters == [0] {
                JpoMetrics { archive: ArchiveCost { payload: 80, ..ArchiveCost::default() }, encode_time_ns: 1, peak_memory_bytes: 1 }
            } else { JpoMetrics { archive: ArchiveCost { payload: 70, ..ArchiveCost::default() }, encode_time_ns: 100, peak_memory_bytes: 100 } })
        }).unwrap();
        assert_eq!(result.stages[0].parameters, vec![1]);
    }

    #[test]
    fn rejects_parameter_domain_shape_mismatch() {
        assert!(optimize_parameters(&[s(1, 0)], &[], JpoLimits { max_passes: 1 }, |_| Ok(m(1))).is_err());
    }
}
