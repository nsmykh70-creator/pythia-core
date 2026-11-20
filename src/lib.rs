//! PYTHIA — reversible transform composition, adaptive entropy coding and versioned single-file and multi-file streaming containers.

pub mod bank;
pub mod chain;
pub mod cost;
pub mod descriptor;
pub mod domain;
pub mod entropy;
pub mod error;
pub mod model;
pub mod jpo;
pub mod search;
pub mod synergy;
pub mod transforms;
pub mod algebraic_search;
pub mod lz77;

pub use bank::{BankEntryId, ModelBank, ModelBankEntry};
pub use chain::{ChainResult, ModelChain, Stage};
pub use cost::{ArchiveCost, BankEconomics};
pub use descriptor::{ChainDescriptor, DESCRIPTOR_VERSION};
pub use domain::{Domain, DomainContract};
pub use entropy::{
    add_best_to_archive_cost as entropy_add_best_to_archive_cost,
    add_selection_to_archive_cost as entropy_add_selection_to_archive_cost,
    add_to_archive_cost as entropy_add_to_archive_cost,
    decode as entropy_decode,
    decode_best as entropy_decode_best,
    encode as entropy_encode,
    encode_best as entropy_encode_best,
    encoded_size as entropy_encoded_size,
    EntropyCandidate, EntropyCodec, EntropySelection,
};
pub use error::UmceError;
pub use model::{ModelId, ReversibleModel, TransformOutput};
pub use jpo::{optimize_parameters, JpoLimits, JpoMetrics, JpoResult, ParameterDomain};
pub use synergy::{analyze_synergy, Evaluation as TsaEvaluation, SynergyResult, TsaReport};
pub use search::{search_compositions, CandidateMetrics, ModelSpec, ScoredChain, SearchLimits, SearchResult};
pub use transforms::{AdaptivePredictor, Bitplane8, Delta2, Delta3, Delta8, XorPrevious, ZigZag8, MODEL_ADAPTIVE_PREDICTOR, MODEL_DELTA2, MODEL_DELTA3, MODEL_LZ77, WeightedMedoid, ProbabilisticMixture, MODEL_WEIGHTED_MEDOID, MODEL_PROBABILISTIC_MIXTURE, AlgebraicRecurrence, MODEL_ALGEBRAIC_RECURRENCE, fit_algebraic_recurrence, LaneTranspose, MODEL_LANE_TRANSPOSE, detect_record_stride};
pub use lz77::Lz77;

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(id: u32) -> Stage {
        Stage { model_id: id, model_version: 1, parameters: Vec::new() }
    }

    #[test]
    fn every_builtin_is_byte_exact_for_all_single_byte_values() {
        let input: Vec<u8> = (0..=255).collect();
        for id in 1..=8 {
            let chain = ModelChain::new(vec![stage(id)]);
            let encoded = chain.forward(&input).unwrap();
            assert_eq!(chain.inverse(&encoded).unwrap(), input, "model {id}");
        }
    }

    #[test]
    fn chains_roundtrip_empty_short_odd_and_large_inputs() {
        let chain = ModelChain::new(vec![stage(1), stage(4), stage(2), stage(3)]);
        for len in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 1024, 65537] {
            let input: Vec<u8> = (0..len).map(|i| ((i * 73 + i / 7) & 255) as u8).collect();
            let encoded = chain.forward(&input).unwrap();
            assert_eq!(chain.inverse(&encoded).unwrap(), input, "length {len}");
        }
    }

    #[test]
    fn lz77_model_roundtrips_repetitive_records() {
        let input = b"record:alpha|record:beta|record:alpha|record:beta|".repeat(4096);
        let chain = ModelChain::new(vec![stage(5)]);
        let encoded = chain.forward(&input).unwrap();
        assert!(encoded.data.len() < input.len());
        assert_eq!(chain.inverse(&encoded).unwrap(), input);
    }

    #[test]
    fn deterministic_forward_output() {
        let input = b"PYTHIA deterministic composition test";
        let chain = ModelChain::new(vec![stage(2), stage(1), stage(4)]);
        assert_eq!(chain.forward(input).unwrap(), chain.forward(input).unwrap());
    }

    #[test]
    fn algebraic_recurrence_is_exact_and_parameterized() {
        let input: Vec<u8> = (0..8192).map(|i| ((i * 17 + (i / 7) * 31) & 255) as u8).collect();
        let params = crate::fit_algebraic_recurrence(&input);
        let chain = ModelChain::new(vec![Stage { model_id: crate::MODEL_ALGEBRAIC_RECURRENCE, model_version: 1, parameters: params.to_vec() }]);
        let transformed = chain.forward(&input).unwrap();
        assert_eq!(chain.inverse(&transformed).unwrap(), input);
        assert_eq!(chain.forward(&input).unwrap(), transformed);
    }

    #[test]
    fn algebraic_recurrence_rejects_invalid_parameter_length() {
        let model = crate::AlgebraicRecurrence;
        assert!(model.forward(b"abc", &[1, 2, 3]).is_err());
        assert!(model.inverse(b"abc", &[1, 2, 3, 4], &[0]).is_err());
    }

    #[test]
    fn rejects_unknown_model() {
        let chain = ModelChain::new(vec![stage(999)]);
        assert_eq!(chain.forward(b"x"), Err(UmceError::UnknownModel(999)));
    }

    #[test]
    fn rejects_wrong_auxiliary_count() {
        let chain = ModelChain::new(vec![stage(1)]);
        let result = ChainResult { data: vec![1], auxiliary: vec![] };
        assert!(matches!(chain.inverse(&result), Err(UmceError::LengthMismatch { .. })));
    }
}


#[cfg(test)]
mod ps62_tests {
    use super::*;

    fn stage(id: u32) -> Stage { Stage { model_id: id, model_version: 1, parameters: vec![] } }

    #[test]
    fn descriptor_roundtrip_is_canonical() {
        let d = ChainDescriptor { input_domain: Domain::Bytes, output_domain: Domain::Bytes, stages: vec![stage(1), stage(4)] };
        let bytes = d.encode().unwrap();
        assert_eq!(ChainDescriptor::decode(&bytes, 32, 4096).unwrap(), d);
        assert_eq!(ChainDescriptor::decode(&bytes, 32, 4096).unwrap().encode().unwrap(), bytes);
    }

    #[test]
    fn descriptor_rejects_trailing_and_noncanonical_varints() {
        let d = ChainDescriptor { input_domain: Domain::Bytes, output_domain: Domain::Bytes, stages: vec![] };
        let mut bytes = d.encode().unwrap(); bytes.push(0);
        assert!(ChainDescriptor::decode(&bytes, 32, 4096).is_err());
        let mut bad = b"PY6D\x01\x00\x00\x80\x00".to_vec();
        assert!(ChainDescriptor::decode(&bad, 32, 4096).is_err());
        bad.clear();
    }

    #[test]
    fn model_bank_deduplicates_only_identical_versioned_instances() {
        let mut bank = ModelBank::new();
        let a = bank.intern(7, 1, &[1, 2]).unwrap();
        let b = bank.intern(7, 1, &[1, 2]).unwrap();
        let c = bank.intern(7, 2, &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(bank.entries().count(), 2);
    }

    #[test]
    fn bank_economics_uses_full_amortized_cost() {
        let good = BankEconomics { standalone_descriptor_bytes: 12, bank_entry_bytes: 20, reference_bytes_per_segment: 2, segment_count: 10 };
        let bad = BankEconomics { standalone_descriptor_bytes: 4, bank_entry_bytes: 20, reference_bytes_per_segment: 2, segment_count: 2 };
        assert!(good.is_profitable().unwrap());
        assert!(!bad.is_profitable().unwrap());
    }

    #[test]
    fn chain_domain_validation_checks_declared_boundaries() {
        let chain = ModelChain::new(vec![stage(1), stage(4)]);
        assert!(chain.validate_domains(Domain::Bytes, Domain::Bytes).is_ok());
        assert!(chain.validate_domains(Domain::I8Samples, Domain::Bytes).is_err());
    }

    #[test]
    fn archive_cost_detects_overflow() {
        let c = ArchiveCost { payload: u64::MAX, header: 1, ..ArchiveCost::default() };
        assert!(c.total().is_err());
    }
}

#[cfg(test)]
#[path = "../tests.rs"]
mod regression_tests;

/// Versioned, checksummed streaming archive container.
pub mod container;

#[cfg(test)]
mod ps87_predictive_residual_tests {
    use super::*;

    fn one_stage(id: u32) -> ModelChain {
        ModelChain::new(vec![Stage { model_id: id, model_version: 1, parameters: Vec::new() }])
    }

    #[test]
    fn delta2_and_delta3_roundtrip_edge_lengths_and_all_byte_values() {
        let mut cases = vec![Vec::new(), (0..=255).collect()];
        for len in [1usize, 2, 3, 7, 8, 9, 255, 256, 257, 4096, 65537] {
            cases.push((0..len).map(|i| ((i * 73 + i / 5 + 19) & 255) as u8).collect());
        }
        for id in [crate::transforms::MODEL_DELTA2, crate::transforms::MODEL_DELTA3] {
            let chain = one_stage(id);
            for input in &cases {
                let encoded = chain.forward(input).unwrap();
                assert_eq!(chain.inverse(&encoded).unwrap(), *input, "model {id}, len {}", input.len());
            }
        }
    }

    #[test]
    fn higher_order_differences_annihilate_low_degree_byte_polynomials() {
        let linear: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(17) & 255) as u8).collect();
        let quadratic: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(i).wrapping_add(3*i).wrapping_add(7) & 255) as u8).collect();
        let d2 = one_stage(crate::transforms::MODEL_DELTA2).forward(&linear).unwrap().data;
        let d3 = one_stage(crate::transforms::MODEL_DELTA3).forward(&quadratic).unwrap().data;
        assert!(d2.iter().skip(2).all(|&x| x == 0));
        assert!(d3.iter().skip(3).all(|&x| x == 0));
    }

    #[test]
    fn predictive_models_reject_unexpected_parameters_and_auxiliary_data() {
        let model = crate::transforms::Delta2;
        assert!(model.forward(&[1, 2, 3], &[0]).is_err());
        assert!(model.inverse(&[1, 2, 3], &[], &[0]).is_err());
    }
}


#[cfg(test)]
mod ps88_adaptive_predictor_tests {
    use super::*;

    fn chain() -> ModelChain {
        ModelChain::new(vec![Stage { model_id: crate::transforms::MODEL_ADAPTIVE_PREDICTOR,
            model_version: 1, parameters: Vec::new() }])
    }

    #[test]
    fn adaptive_predictor_roundtrips_edge_lengths_and_byte_patterns() {
        let mut cases = vec![Vec::new(), vec![0], vec![255], (0..=255).collect()];
        for len in [2usize, 3, 4, 7, 8, 9, 255, 256, 257, 4096, 65537] {
            cases.push((0..len).map(|i| ((i * 31 + i / 7 + 91) & 255) as u8).collect());
            cases.push((0..len).map(|i| ((i * i + 13 * i + 5) & 255) as u8).collect());
        }
        cases.push(b"PYTHIA adaptive expert predictor ".repeat(4096));
        let chain = chain();
        for input in cases {
            let encoded = chain.forward(&input).unwrap();
            assert_eq!(chain.inverse(&encoded).unwrap(), input);
        }
    }

    #[test]
    fn adaptive_predictor_is_deterministic_and_rejects_metadata() {
        let model = crate::transforms::AdaptivePredictor;
        let input = b"adaptive prediction must be deterministic";
        assert_eq!(model.forward(input, &[]).unwrap(), model.forward(input, &[]).unwrap());
        assert!(model.forward(input, &[1]).is_err());
        assert!(model.inverse(input, &[], &[1]).is_err());
    }

    #[test]
    fn adaptive_predictor_tracks_polynomial_sequences() {
        let linear: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(17) & 255) as u8).collect();
        let quadratic: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(i).wrapping_add(3*i).wrapping_add(7) & 255) as u8).collect();
        let model = crate::transforms::AdaptivePredictor;
        for input in [linear, quadratic] {
            let residual = model.forward(&input, &[]).unwrap().data;
            assert!(residual.iter().skip(4).filter(|&&x| x == 0).count() > input.len() / 2);
        }
    }
}


#[cfg(test)]
mod ps89_ensemble_tests {
    use super::*;

    fn chain(id: u32) -> ModelChain {
        ModelChain::new(vec![Stage { model_id: id, model_version: 1, parameters: Vec::new() }])
    }

    fn patterns() -> Vec<Vec<u8>> {
        let mut cases = vec![Vec::new(), vec![0], vec![255], (0..=255).collect(), b"PYTHIA PS8.9 ensemble ".repeat(2048)];
        for len in [2usize, 3, 4, 7, 8, 9, 255, 256, 257, 4096, 65537] {
            cases.push((0..len).map(|i| ((i * 31 + i / 7 + 91) & 255) as u8).collect());
            cases.push((0..len).map(|i| ((i * i + 13 * i + 5) & 255) as u8).collect());
            cases.push((0..len).map(|i| ((i * 17 + 23) & 255) as u8).collect());
        }
        let mut state = 0x9e37_79b9u32;
        cases.push((0..131_071).map(|_| { state ^= state << 13; state ^= state >> 17; state ^= state << 5; state as u8 }).collect());
        cases
    }

    #[test]
    fn both_ps89_ensembles_roundtrip_all_patterns_and_edge_lengths() {
        for id in [crate::transforms::MODEL_WEIGHTED_MEDOID, crate::transforms::MODEL_PROBABILISTIC_MIXTURE] {
            let c = chain(id);
            for input in patterns() {
                let encoded = c.forward(&input).unwrap();
                assert_eq!(c.inverse(&encoded).unwrap(), input, "model {id}, length {}", input.len());
            }
        }
    }

    #[test]
    fn ps89_models_are_deterministic_and_reject_metadata() {
        let weighted = crate::transforms::WeightedMedoid;
        let probabilistic = crate::transforms::ProbabilisticMixture;
        let models: [&dyn ReversibleModel; 2] = [&weighted, &probabilistic];
        for model in models {
            let input = b"fixed point ensemble determinism".repeat(128);
            assert_eq!(model.forward(&input, &[]).unwrap(), model.forward(&input, &[]).unwrap());
            assert!(model.forward(&input, &[1]).is_err());
            assert!(model.inverse(&input, &[], &[1]).is_err());
        }
    }
}


