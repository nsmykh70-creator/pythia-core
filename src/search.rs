//! Bounded, deterministic composition search (PS6.3).
//!
//! The caller supplies an evaluator that MUST return the exact serialized archive
//! cost for the proposed chain. Search heuristics only limit exploration; they do
//! not estimate compression ratios or replace final exact evaluation.
use std::collections::BTreeSet;

use crate::{cost::ArchiveCost, domain::Domain, chain::Stage, UmceError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub model_id: u32,
    pub model_version: u16,
    pub input_domain: Domain,
    pub output_domain: Domain,
    /// Canonical parameter bytes for this search candidate.
    pub parameters: Vec<u8>,
}

impl ModelSpec {
    pub fn stage(&self) -> Stage {
        Stage { model_id: self.model_id, model_version: self.model_version, parameters: self.parameters.clone() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchLimits {
    pub max_depth: usize,
    pub beam_width: usize,
    /// Candidates within this many parts per million of the smallest archive
    /// may be ranked by encode time, then peak memory, then exact size.
    pub size_slack_ppm: u32,
}

impl Default for SearchLimits {
    fn default() -> Self { Self { max_depth: 4, beam_width: 16, size_slack_ppm: 2_000 } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateMetrics {
    pub archive: ArchiveCost,
    pub encode_time_ns: u64,
    pub peak_memory_bytes: u64,
}

impl CandidateMetrics {
    pub fn archive_bytes(self) -> Result<u64, UmceError> { self.archive.total() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredChain {
    pub stages: Vec<Stage>,
    pub metrics: CandidateMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub winner: ScoredChain,
    /// All complete candidates encountered, sorted by the configured ranking.
    pub finalists: Vec<ScoredChain>,
    pub expanded_states: u64,
    pub evaluated_chains: u64,
}

#[derive(Debug, Clone)]
struct State {
    stages: Vec<Stage>,
    domain: Domain,
    metrics: CandidateMetrics,
}

/// Search model chains from `input_domain` to `target_domain`.
///
/// `evaluate` must run the real transform + entropy coder + canonical metadata
/// serializer (or an exact equivalent) and return the full archive byte count.
/// The evaluator is called for every expansion retained by the bounded search.
pub fn search_compositions<F>(
    catalog: &[ModelSpec],
    input_domain: Domain,
    target_domain: Domain,
    limits: SearchLimits,
    mut evaluate: F,
) -> Result<SearchResult, UmceError>
where
    F: FnMut(&[Stage]) -> Result<CandidateMetrics, UmceError>,
{
    if limits.max_depth == 0 || limits.beam_width == 0 {
        return Err(UmceError::InvalidParameters("search depth and beam width must be non-zero"));
    }
    if catalog.is_empty() {
        return Err(UmceError::InvalidParameters("model catalog is empty"));
    }

    let empty = Vec::<Stage>::new();
    let base_metrics = evaluate(&empty)?;
    let mut beam = vec![State { stages: empty, domain: input_domain, metrics: base_metrics }];
    let mut finalists = Vec::new();
    let mut expanded_states = 0u64;
    let mut evaluated_chains = 1u64;
    let mut seen = BTreeSet::<Vec<(u32, u16, Vec<u8>)>>::new();
    seen.insert(Vec::new());

    for _depth in 1..=limits.max_depth {
        let mut next = Vec::<State>::new();
        for state in &beam {
            for model in catalog.iter().filter(|m| m.input_domain == state.domain) {
                let mut stages = state.stages.clone();
                stages.push(model.stage());
                let signature = signature(&stages);
                if !seen.insert(signature) { continue; }
                expanded_states = expanded_states.checked_add(1).ok_or(UmceError::LengthOverflow)?;
                let metrics = evaluate(&stages)?;
                evaluated_chains = evaluated_chains.checked_add(1).ok_or(UmceError::LengthOverflow)?;
                let candidate = State { stages, domain: model.output_domain, metrics };
                if candidate.domain == target_domain {
                    finalists.push(ScoredChain { stages: candidate.stages.clone(), metrics });
                }
                next.push(candidate);
            }
        }
        if next.is_empty() { break; }
        rank_states(&mut next, limits.size_slack_ppm)?;
        next.truncate(limits.beam_width);
        beam = next;
    }

    if finalists.is_empty() {
        return Err(UmceError::InvalidParameters("no chain reaches the requested output domain within max_depth"));
    }
    rank_finalists(&mut finalists, limits.size_slack_ppm)?;
    let winner = finalists[0].clone();
    Ok(SearchResult { winner, finalists, expanded_states, evaluated_chains })
}

fn signature(stages: &[Stage]) -> Vec<(u32, u16, Vec<u8>)> {
    stages.iter().map(|s| (s.model_id, s.model_version, s.parameters.clone())).collect()
}

fn size_threshold(minimum: u64, ppm: u32) -> Result<u64, UmceError> {
    let extra = (minimum as u128).checked_mul(ppm as u128).ok_or(UmceError::LengthOverflow)?;
    let rounded = extra.checked_add(999_999).ok_or(UmceError::LengthOverflow)? / 1_000_000;
    let threshold = (minimum as u128).checked_add(rounded).ok_or(UmceError::LengthOverflow)?;
    u64::try_from(threshold).map_err(|_| UmceError::LengthOverflow)
}

fn rank_states(states: &mut [State], ppm: u32) -> Result<(), UmceError> {
    if states.is_empty() { return Ok(()); }
    let minimum = states.iter().map(|s| s.metrics.archive_bytes()).collect::<Result<Vec<_>, _>>()?.into_iter().min().unwrap_or(0);
    let threshold = size_threshold(minimum, ppm)?;
    states.sort_by_key(|s| {
        let bytes = s.metrics.archive.total().unwrap_or(u64::MAX);
        if bytes <= threshold {
            (0u8, s.metrics.encode_time_ns, s.metrics.peak_memory_bytes, bytes)
        } else {
            (1u8, bytes, s.metrics.encode_time_ns, s.metrics.peak_memory_bytes)
        }
    });
    Ok(())
}

fn rank_finalists(candidates: &mut [ScoredChain], ppm: u32) -> Result<(), UmceError> {
    if candidates.is_empty() { return Ok(()); }
    let minimum = candidates.iter().map(|s| s.metrics.archive_bytes()).collect::<Result<Vec<_>, _>>()?.into_iter().min().unwrap_or(0);
    let threshold = size_threshold(minimum, ppm)?;
    candidates.sort_by_key(|s| {
        let bytes = s.metrics.archive.total().unwrap_or(u64::MAX);
        if bytes <= threshold {
            (0u8, s.metrics.encode_time_ns, s.metrics.peak_memory_bytes, bytes)
        } else {
            (1u8, bytes, s.metrics.encode_time_ns, s.metrics.peak_memory_bytes)
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: u32, input: Domain, output: Domain) -> ModelSpec {
        ModelSpec { model_id: id, model_version: 1, input_domain: input, output_domain: output, parameters: vec![] }
    }

    fn metrics(bytes: u64, time: u64, ram: u64) -> CandidateMetrics {
        CandidateMetrics { archive: ArchiveCost { payload: bytes, ..ArchiveCost::default() }, encode_time_ns: time, peak_memory_bytes: ram }
    }

    #[test]
    fn finds_chain_with_matching_domain_edges() {
        let catalog = [spec(1, Domain::Bytes, Domain::U8Samples), spec(2, Domain::U8Samples, Domain::Bytes)];
        let result = search_compositions(&catalog, Domain::Bytes, Domain::Bytes,
            SearchLimits { max_depth: 2, beam_width: 8, size_slack_ppm: 0 },
            |stages| Ok(metrics(100 - stages.len() as u64, 10, 20))).unwrap();
        assert_eq!(result.winner.stages.len(), 2);
        assert_eq!(result.winner.stages[0].model_id, 1);
        assert_eq!(result.winner.stages[1].model_id, 2);
    }

    #[test]
    fn exact_size_precedes_speed_outside_slack() {
        let catalog = [spec(1, Domain::Bytes, Domain::Bytes), spec(2, Domain::Bytes, Domain::Bytes)];
        let result = search_compositions(&catalog, Domain::Bytes, Domain::Bytes,
            SearchLimits { max_depth: 1, beam_width: 8, size_slack_ppm: 0 },
            |stages| Ok(if stages.is_empty() { metrics(120, 0, 0) } else if stages[0].model_id == 1 { metrics(90, 1, 1) } else { metrics(80, 100, 100) })).unwrap();
        assert_eq!(result.winner.stages[0].model_id, 2);
    }

    #[test]
    fn speed_breaks_ties_inside_size_slack() {
        let catalog = [spec(1, Domain::Bytes, Domain::Bytes), spec(2, Domain::Bytes, Domain::Bytes)];
        let result = search_compositions(&catalog, Domain::Bytes, Domain::Bytes,
            SearchLimits { max_depth: 1, beam_width: 8, size_slack_ppm: 20_000 },
            |stages| Ok(if stages.is_empty() { metrics(120, 0, 0) } else if stages[0].model_id == 1 { metrics(100, 90, 10) } else { metrics(101, 10, 50) })).unwrap();
        assert_eq!(result.winner.stages[0].model_id, 2);
    }

    #[test]
    fn rejects_zero_limits_and_unreachable_domain() {
        let catalog = [spec(1, Domain::Bytes, Domain::U8Samples)];
        assert!(search_compositions(&catalog, Domain::Bytes, Domain::Bytes,
            SearchLimits { max_depth: 0, beam_width: 1, size_slack_ppm: 0 }, |_| Ok(metrics(1, 1, 1))).is_err());
        assert!(search_compositions(&catalog, Domain::Bytes, Domain::FixedRecords,
            SearchLimits { max_depth: 1, beam_width: 2, size_slack_ppm: 0 }, |_| Ok(metrics(1, 1, 1))).is_err());
    }
}
