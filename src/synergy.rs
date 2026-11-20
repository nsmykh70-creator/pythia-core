//! Transform Synergy Analysis (TSA), PS6.4.
//!
//! TSA compares exact complete-archive costs for baseline, individual models,
//! and ordered compositions. A positive saving means the composition costs
//! fewer bytes than the stated independent baseline. Order is never normalized.

use crate::{chain::Stage, cost::ArchiveCost, UmceError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evaluation {
    pub archive: ArchiveCost,
    pub encode_time_ns: u64,
    pub peak_memory_bytes: u64,
}

impl Evaluation {
    pub fn bytes(self) -> Result<u64, UmceError> { self.archive.total() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynergyResult {
    pub stages: Vec<Stage>,
    pub measured: Evaluation,
    /// baseline_bytes - measured_bytes; positive is beneficial.
    pub saving_bytes: i128,
    /// Interaction gain. For pairs: L(A)+L(B)-L(0)-L(AB).
    /// For triples: the third-order inclusion-exclusion interaction term.
    pub synergy_bytes: Option<i128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TsaReport {
    pub baseline: Evaluation,
    pub singles: Vec<SynergyResult>,
    /// Ordered pairs: A->B and B->A are separate results.
    pub ordered_pairs: Vec<SynergyResult>,
    /// Ordered triples are evaluated only when explicitly requested.
    pub ordered_triples: Vec<SynergyResult>,
}

/// Evaluate exact baseline/single/pair/triple archive costs.
/// `evaluate` must serialize and entropy-code each candidate and include every
/// archive component. The function does not estimate compression ratios.
pub fn analyze_synergy<F>(
    models: &[Stage],
    include_triples: bool,
    mut evaluate: F,
) -> Result<TsaReport, UmceError>
where
    F: FnMut(&[Stage]) -> Result<Evaluation, UmceError>,
{
    let baseline = evaluate(&[])?;
    let baseline_bytes = baseline.bytes()?;
    let mut singles = Vec::new();
    let mut ordered_pairs = Vec::new();
    let mut ordered_triples = Vec::new();
    let mut single_sizes = vec![0u64; models.len()];
    let mut pair_sizes = vec![vec![None; models.len()]; models.len()];

    // Measure every singleton first: pair synergy depends on both values.
    for a in 0..models.len() {
        let one = vec![models[a].clone()];
        let single_eval = evaluate(&one)?;
        single_sizes[a] = single_eval.bytes()?;
        singles.push(result(one, single_eval, baseline_bytes, None)?);
    }
    // Ordered pairs are distinct experiments; A→B is not B→A.
    for a in 0..models.len() {
        for b in 0..models.len() {
            if a == b { continue; }
            let pair = vec![models[a].clone(), models[b].clone()];
            let pair_eval = evaluate(&pair)?;
            let pair_bytes = pair_eval.bytes()?;
            pair_sizes[a][b] = Some(pair_bytes);
            let pair_synergy = i128::from(single_sizes[a]) + i128::from(single_sizes[b])
                - i128::from(baseline_bytes) - i128::from(pair_bytes);
            ordered_pairs.push(result(pair, pair_eval, baseline_bytes, Some(pair_synergy))?);
        }
    }
    if include_triples {
        for a in 0..models.len() {
            for b in 0..models.len() {
                if b == a { continue; }
                for c in 0..models.len() {
                    if c == a || c == b { continue; }
                    let triple = vec![models[a].clone(), models[b].clone(), models[c].clone()];
                    let triple_eval = evaluate(&triple)?;
                    let ab = pair_sizes[a][b].ok_or(UmceError::InvalidParameters("missing TSA pair measurement"))?;
                    let ac = pair_sizes[a][c].ok_or(UmceError::InvalidParameters("missing TSA pair measurement"))?;
                    let bc = pair_sizes[b][c].ok_or(UmceError::InvalidParameters("missing TSA pair measurement"))?;
                    let third_order = i128::from(ab) + i128::from(ac) + i128::from(bc)
                        - i128::from(single_sizes[a]) - i128::from(single_sizes[b])
                        - i128::from(single_sizes[c]) + i128::from(baseline_bytes)
                        - i128::from(triple_eval.bytes()?);
                    ordered_triples.push(result(triple.clone(), triple_eval, baseline_bytes, Some(third_order))?);
                }
            }
        }
    }
    singles.sort_by_key(result_order);
    ordered_pairs.sort_by_key(result_order);
    ordered_triples.sort_by_key(result_order);
    Ok(TsaReport { baseline, singles, ordered_pairs, ordered_triples })
}

fn result(stages: Vec<Stage>, measured: Evaluation, baseline_bytes: u64, synergy_bytes: Option<i128>) -> Result<SynergyResult, UmceError> {
    let bytes = measured.bytes()?;
    let saving = i128::from(baseline_bytes) - i128::from(bytes);
    Ok(SynergyResult { stages, measured, saving_bytes: saving, synergy_bytes })
}

fn result_order(r: &SynergyResult) -> (u64, u64, u64, Vec<(u32, u16, Vec<u8>)>) {
    (r.measured.archive.total().unwrap_or(u64::MAX), r.measured.encode_time_ns,
     r.measured.peak_memory_bytes,
     r.stages.iter().map(|s| (s.model_id, s.model_version, s.parameters.clone())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(id: u32) -> Stage { Stage { model_id: id, model_version: 1, parameters: vec![] } }
    fn e(n: u64) -> Evaluation { Evaluation { archive: ArchiveCost { payload: n, ..ArchiveCost::default() }, encode_time_ns: n, peak_memory_bytes: n } }

    #[test]
    fn evaluates_both_pair_orders_and_reports_real_savings() {
        let models = [s(1), s(2)];
        let report = analyze_synergy(&models, false, |chain| Ok(match chain {
            [] => e(100), [a] if a.model_id == 1 => e(95), [a] if a.model_id == 2 => e(90),
            [a, b] if a.model_id == 1 && b.model_id == 2 => e(70), _ => e(110),
        })).unwrap();
        assert_eq!(report.ordered_pairs.len(), 2);
        let ab = report.ordered_pairs.iter().find(|x| x.stages[0].model_id == 1).unwrap();
        let ba = report.ordered_pairs.iter().find(|x| x.stages[0].model_id == 2).unwrap();
        assert_eq!(ab.saving_bytes, 30);
        assert_eq!(ab.synergy_bytes, Some(15));
        assert_eq!(ba.saving_bytes, -10);
        assert_eq!(ba.synergy_bytes, Some(-25));
    }

    #[test]
    fn triple_search_is_opt_in_and_ordered() {
        let models = [s(1), s(2), s(3)];
        let report = analyze_synergy(&models, true, |_| Ok(e(100))).unwrap();
        assert_eq!(report.ordered_triples.len(), 6);
    }
}
