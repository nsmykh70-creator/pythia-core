//! Initial reversible byte transforms. All arithmetic is explicitly modulo 256
//! where appropriate. IDs are reserved by the PS6.1 prototype.

use crate::{model::{require_empty_aux, require_empty_params, ReversibleModel, TransformOutput}, ModelId, UmceError};

pub const MODEL_DELTA8: ModelId = 1;
pub const MODEL_XOR_PREVIOUS: ModelId = 2;
pub const MODEL_ZIGZAG8: ModelId = 3;
pub const MODEL_BITPLANE8: ModelId = 4;
pub const MODEL_LZ77: ModelId = crate::lz77::MODEL_LZ77;
pub const MODEL_DELTA2: ModelId = 6;
pub const MODEL_DELTA3: ModelId = 7;
pub const MODEL_ADAPTIVE_PREDICTOR: ModelId = 8;
pub const MODEL_WEIGHTED_MEDOID: ModelId = 9;
pub const MODEL_PROBABILISTIC_MIXTURE: ModelId = 10;
pub const MODEL_ALGEBRAIC_RECURRENCE: ModelId = 11;
pub const MODEL_LANE_TRANSPOSE: ModelId = 12;

#[derive(Debug, Default, Clone, Copy)]
pub struct Delta8;
impl ReversibleModel for Delta8 {
    fn id(&self) -> ModelId { MODEL_DELTA8 }
    fn name(&self) -> &'static str { "delta8" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = 0u8;
        for &x in input {
            out.push(x.wrapping_sub(prev));
            prev = x;
        }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = 0u8;
        for &d in input {
            let x = d.wrapping_add(prev);
            out.push(x);
            prev = x;
        }
        Ok(out)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct XorPrevious;
impl ReversibleModel for XorPrevious {
    fn id(&self) -> ModelId { MODEL_XOR_PREVIOUS }
    fn name(&self) -> &'static str { "xor_previous" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = 0u8;
        for &x in input { out.push(x ^ prev); prev = x; }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = 0u8;
        for &d in input { let x = d ^ prev; out.push(x); prev = x; }
        Ok(out)
    }
}

/// Bijective mapping of signed i8 interpretation to unsigned ZigZag byte.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZigZag8;
impl ReversibleModel for ZigZag8 {
    fn id(&self) -> ModelId { MODEL_ZIGZAG8 }
    fn name(&self) -> &'static str { "zigzag8" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let data = input.iter().map(|&b| {
            let x = b as i8 as i16;
            ((x << 1) ^ (x >> 7)) as u8
        }).collect();
        Ok(TransformOutput { data, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let data = input.iter().map(|&u| {
            let z = u as i16;
            ((z >> 1) ^ -(z & 1)) as i8 as u8
        }).collect();
        Ok(data)
    }
}

/// Transposes each complete group of eight bytes into eight bitplanes.
/// A final group shorter than eight bytes is copied unchanged. This rule is
/// deterministic and length-preserving; inverse applies the exact same grouping.
#[derive(Debug, Default, Clone, Copy)]
pub struct Bitplane8;
impl ReversibleModel for Bitplane8 {
    fn id(&self) -> ModelId { MODEL_BITPLANE8 }
    fn name(&self) -> &'static str { "bitplane8" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks_exact(8) {
            for bit in 0..8 {
                let mut plane = 0u8;
                for (i, &byte) in chunk.iter().enumerate() {
                    plane |= ((byte >> bit) & 1) << i;
                }
                out.push(plane);
            }
        }
        out.extend_from_slice(input.chunks_exact(8).remainder());
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let mut out = Vec::with_capacity(input.len());
        let complete = input.len() / 8 * 8;
        for planes in input[..complete].chunks_exact(8) {
            let mut bytes = [0u8; 8];
            for (bit, &plane) in planes.iter().enumerate() {
                for i in 0..8 { bytes[i] |= ((plane >> i) & 1) << bit; }
            }
            out.extend_from_slice(&bytes);
        }
        out.extend_from_slice(&input[complete..]);
        Ok(out)
    }
}

/// Fixed registry for the first prototype. Later PS6 stages replace this with
/// a versioned registry populated by PS1–PS5 model providers.
pub fn builtin_model(id: ModelId) -> Option<Box<dyn ReversibleModel>> {
    match id {
        MODEL_DELTA8 => Some(Box::new(Delta8)),
        MODEL_XOR_PREVIOUS => Some(Box::new(XorPrevious)),
        MODEL_ZIGZAG8 => Some(Box::new(ZigZag8)),
        MODEL_BITPLANE8 => Some(Box::new(Bitplane8)),
        MODEL_LZ77 => Some(Box::new(crate::lz77::Lz77)),
        MODEL_DELTA2 => Some(Box::new(Delta2)),
        MODEL_DELTA3 => Some(Box::new(Delta3)),
        MODEL_ADAPTIVE_PREDICTOR => Some(Box::new(AdaptivePredictor)),
        MODEL_WEIGHTED_MEDOID => Some(Box::new(WeightedMedoid)),
        MODEL_PROBABILISTIC_MIXTURE => Some(Box::new(ProbabilisticMixture)),
        MODEL_ALGEBRAIC_RECURRENCE => Some(Box::new(AlgebraicRecurrence)),
        MODEL_LANE_TRANSPOSE => Some(Box::new(LaneTranspose)),
        _ => None,
    }
}

/// Second-order finite-difference residual modulo 256:
/// e[i] = x[i] - 2*x[i-1] + x[i-2]. The two initial history bytes are zero.
/// This is exactly reversible for arbitrary byte streams and makes linear
/// byte ramps constant-zero after their first two symbols.
#[derive(Debug, Default, Clone, Copy)]
pub struct Delta2;
impl ReversibleModel for Delta2 {
    fn id(&self) -> ModelId { MODEL_DELTA2 }
    fn name(&self) -> &'static str { "delta2_mod256" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let (mut p1, mut p2) = (0u8, 0u8);
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            out.push(x.wrapping_sub(p1.wrapping_mul(2)).wrapping_add(p2));
            p2 = p1;
            p1 = x;
        }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let (mut p1, mut p2) = (0u8, 0u8);
        let mut out = Vec::with_capacity(input.len());
        for &e in input {
            let x = e.wrapping_add(p1.wrapping_mul(2)).wrapping_sub(p2);
            out.push(x);
            p2 = p1;
            p1 = x;
        }
        Ok(out)
    }
}

/// Third-order finite-difference residual modulo 256:
/// e[i] = x[i] - 3*x[i-1] + 3*x[i-2] - x[i-3].
/// The three initial history bytes are zero.
#[derive(Debug, Default, Clone, Copy)]
pub struct Delta3;
impl ReversibleModel for Delta3 {
    fn id(&self) -> ModelId { MODEL_DELTA3 }
    fn name(&self) -> &'static str { "delta3_mod256" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            out.push(x.wrapping_sub(p1.wrapping_mul(3)).wrapping_add(p2.wrapping_mul(3)).wrapping_sub(p3));
            p3 = p2; p2 = p1; p1 = x;
        }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
        let mut out = Vec::with_capacity(input.len());
        for &e in input {
            let x = e.wrapping_add(p1.wrapping_mul(3)).wrapping_sub(p2.wrapping_mul(3)).wrapping_add(p3);
            out.push(x);
            p3 = p2; p2 = p1; p1 = x;
        }
        Ok(out)
    }
}


/// Online mixture-of-experts predictor for byte streams (PS8.8).
///
/// Five deterministic experts predict the next byte from already decoded
/// history: zero, previous byte, linear extrapolation, quadratic
/// extrapolation, and XOR of the two previous bytes. At each position the
/// expert with the lowest cumulative circular-byte prediction error is used.
/// All arithmetic is modulo 256; cumulative losses saturate at u64::MAX.
/// The decoder can reproduce the same expert choice from reconstructed
/// history, so no side channel or per-byte model decisions are serialized.
#[derive(Debug, Default, Clone, Copy)]
pub struct AdaptivePredictor;

impl ReversibleModel for AdaptivePredictor {
    fn id(&self) -> ModelId { MODEL_ADAPTIVE_PREDICTOR }
    fn name(&self) -> &'static str { "adaptive_predictor_moe5" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }

    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let mut losses = [0u64; 5];
        let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            let predictions = predictor_set(p1, p2, p3);
            let mut best = 0usize;
            for i in 1..predictions.len() {
                if losses[i] < losses[best] { best = i; }
            }
            let prediction = predictions[best];
            out.push(x.wrapping_sub(prediction));
            for i in 0..predictions.len() {
                losses[i] = losses[i].saturating_add(byte_distance(x, predictions[i]) as u64);
            }
            p3 = p2; p2 = p1; p1 = x;
        }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }

    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?;
        require_empty_aux(a)?;
        let mut losses = [0u64; 5];
        let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
        let mut out = Vec::with_capacity(input.len());
        for &residual in input {
            let predictions = predictor_set(p1, p2, p3);
            let mut best = 0usize;
            for i in 1..predictions.len() {
                if losses[i] < losses[best] { best = i; }
            }
            let x = residual.wrapping_add(predictions[best]);
            out.push(x);
            for i in 0..predictions.len() {
                losses[i] = losses[i].saturating_add(byte_distance(x, predictions[i]) as u64);
            }
            p3 = p2; p2 = p1; p1 = x;
        }
        Ok(out)
    }
}

#[inline]
fn predictor_set(p1: u8, p2: u8, p3: u8) -> [u8; 5] {
    [0, p1, p1.wrapping_mul(2).wrapping_sub(p2),
     p1.wrapping_mul(3).wrapping_sub(p2.wrapping_mul(3)).wrapping_add(p3),
     p1 ^ p2]
}

#[inline]
fn byte_distance(a: u8, b: u8) -> u8 {
    let d = a.wrapping_sub(b);
    d.min(0u8.wrapping_sub(d))
}


/// PS8.9 weighted-medoid mixture. Each expert receives a weight derived from
/// its cumulative circular-byte loss. The selected prediction is the expert
/// candidate minimizing the weighted sum of circular distances. This is a
/// deterministic robust ensemble and requires no serialized side information.
#[derive(Debug, Default, Clone, Copy)]
pub struct WeightedMedoid;

impl ReversibleModel for WeightedMedoid {
    fn id(&self) -> ModelId { MODEL_WEIGHTED_MEDOID }
    fn name(&self) -> &'static str { "weighted_medoid_moe5" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }

    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        Ok(TransformOutput { data: encode_weighted_medoid(input), auxiliary: Vec::new() })
    }

    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?;
        require_empty_aux(a)?;
        Ok(decode_weighted_medoid(input))
    }
}

fn encode_weighted_medoid(input: &[u8]) -> Vec<u8> {
    let mut losses = [0u64; 5];
    let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
    let mut residuals = Vec::with_capacity(input.len());
    for &x in input {
        let preds = predictor_set(p1, p2, p3);
        let weights = losses.map(|loss| 1u64 + (1u64 << 20) / loss.saturating_add(1));
        let prediction = weighted_medoid_candidate(&preds, &weights);
        residuals.push(x.wrapping_sub(prediction));
        for k in 0..5 { losses[k] = losses[k].saturating_add(byte_distance(x, preds[k]) as u64); }
        p3 = p2; p2 = p1; p1 = x;
    }
    residuals
}

fn decode_weighted_medoid(input: &[u8]) -> Vec<u8> {
    let mut losses = [0u64; 5];
    let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
    let mut output = Vec::with_capacity(input.len());
    for &r in input {
        let preds = predictor_set(p1, p2, p3);
        let weights = losses.map(|loss| 1u64 + (1u64 << 20) / loss.saturating_add(1));
        let x = r.wrapping_add(weighted_medoid_candidate(&preds, &weights));
        output.push(x);
        for k in 0..5 { losses[k] = losses[k].saturating_add(byte_distance(x, preds[k]) as u64); }
        p3 = p2; p2 = p1; p1 = x;
    }
    output
}

#[inline]
fn weighted_medoid_candidate(preds: &[u8; 5], weights: &[u64; 5]) -> u8 {
    let mut best = 0usize;
    let mut best_cost = u128::MAX;
    for candidate in 0..5 {
        let mut cost = 0u128;
        for expert in 0..5 {
            cost += (weights[expert] as u128) * (byte_distance(preds[candidate], preds[expert]) as u128);
        }
        if cost < best_cost { best_cost = cost; best = candidate; }
    }
    preds[best]
}

/// PS8.9 probabilistic mixture over five expert-centered discrete kernels.
/// Each expert assigns likelihood proportional to 1/(1+circular_distance).
/// The mixture MAP is searched over the five expert predictions (the support
/// candidates), then Bayesian-like integer weights are updated from the
/// observed symbol. Fixed-point arithmetic makes encoder/decoder identical.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbabilisticMixture;

impl ReversibleModel for ProbabilisticMixture {
    fn id(&self) -> ModelId { MODEL_PROBABILISTIC_MIXTURE }
    fn name(&self) -> &'static str { "probabilistic_mixture_moe5" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }

    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        Ok(TransformOutput { data: encode_probabilistic_mixture(input), auxiliary: Vec::new() })
    }

    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?;
        require_empty_aux(a)?;
        Ok(decode_probabilistic_mixture(input))
    }
}

const MIXTURE_WEIGHT_TOTAL: u64 = 1 << 24;
const MIXTURE_LIKELIHOOD_SCALE: u64 = 1 << 16;

#[inline]
fn expert_likelihood(distance: u8) -> u64 {
    (MIXTURE_LIKELIHOOD_SCALE / (1 + distance as u64)).max(1)
}

fn mixture_prediction(preds: &[u8; 5], weights: &[u64; 5]) -> u8 {
    let mut best = 0usize;
    let mut best_score = 0u128;
    for candidate in 0..5 {
        let mut score = 0u128;
        for expert in 0..5 {
            score += (weights[expert] as u128)
                * (expert_likelihood(byte_distance(preds[candidate], preds[expert])) as u128);
        }
        if score > best_score { best_score = score; best = candidate; }
    }
    preds[best]
}

fn update_mixture_weights(weights: &mut [u64; 5], actual: u8, preds: &[u8; 5]) {
    let mut raw = [0u64; 5];
    let mut total = 0u128;
    for k in 0..5 {
        raw[k] = weights[k].saturating_mul(expert_likelihood(byte_distance(actual, preds[k]))).max(1);
        total += raw[k] as u128;
    }
    if total == 0 { *weights = [MIXTURE_WEIGHT_TOTAL / 5; 5]; return; }
    let mut normalized = [0u64; 5];
    let mut assigned = 0u64;
    for k in 0..5 {
        normalized[k] = (((raw[k] as u128) * (MIXTURE_WEIGHT_TOTAL as u128)) / total) as u64;
        assigned += normalized[k];
    }
    // Assign integer-rounding remainder to the strongest posterior expert.
    let mut strongest = 0usize;
    for k in 1..5 {
        if raw[k] > raw[strongest] { strongest = k; }
    }
    normalized[strongest] += MIXTURE_WEIGHT_TOTAL.saturating_sub(assigned);
    *weights = normalized;
}

fn encode_probabilistic_mixture(input: &[u8]) -> Vec<u8> {
    let mut weights = [MIXTURE_WEIGHT_TOTAL / 5; 5];
    let initial_remainder = MIXTURE_WEIGHT_TOTAL - weights.iter().sum::<u64>();
    weights[0] += initial_remainder;
    let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
    let mut residuals = Vec::with_capacity(input.len());
    for &x in input {
        let preds = predictor_set(p1, p2, p3);
        let prediction = mixture_prediction(&preds, &weights);
        residuals.push(x.wrapping_sub(prediction));
        update_mixture_weights(&mut weights, x, &preds);
        p3 = p2; p2 = p1; p1 = x;
    }
    residuals
}

fn decode_probabilistic_mixture(input: &[u8]) -> Vec<u8> {
    let mut weights = [MIXTURE_WEIGHT_TOTAL / 5; 5];
    let initial_remainder = MIXTURE_WEIGHT_TOTAL - weights.iter().sum::<u64>();
    weights[0] += initial_remainder;
    let (mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8);
    let mut output = Vec::with_capacity(input.len());
    for &r in input {
        let preds = predictor_set(p1, p2, p3);
        let x = r.wrapping_add(mixture_prediction(&preds, &weights));
        output.push(x);
        update_mixture_weights(&mut weights, x, &preds);
        p3 = p2; p2 = p1; p1 = x;
    }
    output
}


/// PS9.3: reversible fourth-order recurrence over the ring Z/256Z.
/// Parameters are four causal coefficients a1..a4. Residual:
/// e[i] = x[i] - sum(a[j] * x[i-j-1]) mod 256.
/// The inverse is causal and exact for every byte sequence, regardless of
/// whether the recurrence is a good statistical model for that sequence.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlgebraicRecurrence;

impl ReversibleModel for AlgebraicRecurrence {
    fn id(&self) -> ModelId { MODEL_ALGEBRAIC_RECURRENCE }
    fn name(&self) -> &'static str { "algebraic_recurrence_z256_order4" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> {
        if p.len() == 4 { Ok(()) } else { Err(UmceError::InvalidParameters("algebraic recurrence requires four coefficients")) }
    }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        let mut out = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            let mut prediction = 0u8;
            for j in 0..4 {
                if i > j { prediction = prediction.wrapping_add(p[j].wrapping_mul(input[i-j-1])); }
            }
            out.push(input[i].wrapping_sub(prediction));
        }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?; require_empty_aux(a)?;
        let mut out = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            let mut prediction = 0u8;
            for j in 0..4 {
                if i > j { prediction = prediction.wrapping_add(p[j].wrapping_mul(out[i-j-1])); }
            }
            out.push(input[i].wrapping_add(prediction));
        }
        Ok(out)
    }
}

/// Fit recurrence coefficients by coordinate ascent on the order-2 Renyi
/// concentration of the residual histogram: maximize sum(count[s]^2).
/// This objective is an integer, deterministic proxy for reducing entropy.
/// The final archive selector still measures the exact encoded frame cost.
pub fn fit_algebraic_recurrence(sample: &[u8]) -> [u8; 4] {
    // Coefficient fitting is intentionally performed on a bounded prefix;
    // the exact full-block archive-cost comparison decides whether to keep it.
    let sample = &sample[..sample.len().min(4096)];
    let mut coeff = [0u8; 4];
    if sample.len() < 2 { return coeff; }
    let score = |a: &[u8; 4]| -> u128 {
        let mut hist = [0u64; 256];
        for i in 0..sample.len() {
            let mut pred = 0u8;
            for j in 0..4 { if i > j { pred = pred.wrapping_add(a[j].wrapping_mul(sample[i-j-1])); } }
            hist[sample[i].wrapping_sub(pred) as usize] += 1;
        }
        hist.iter().map(|&n| (n as u128) * (n as u128)).sum()
    };
    let mut best = score(&coeff);
    // Two deterministic coordinate passes allow coefficients to interact.
    for _ in 0..2 {
        for coordinate in 0..4 {
            let mut chosen = coeff[coordinate];
            let mut chosen_score = best;
            for candidate in 0u16..=255 {
                let mut trial = coeff;
                trial[coordinate] = candidate as u8;
                let value = score(&trial);
                if value > chosen_score || (value == chosen_score && (candidate as u8) < chosen) {
                    chosen = candidate as u8;
                    chosen_score = value;
                }
            }
            coeff[coordinate] = chosen;
            best = chosen_score;
        }
    }
    // PS9.8: coordinate ascent can miss exact recurrences (Fibonacci mod 256 is
    // one). Lifting finds them; it is accepted only if it concentrates the full
    // residual histogram strictly better, so noisy data keeps the old result.
    let lifted = lift_algebraic_recurrence(sample);
    if recurrence_concentration(sample, &lifted) > best { return lifted; }
    coeff
}

/// Order-2 Renyi concentration (sum of squared residual counts) of a recurrence.
fn recurrence_concentration(sample: &[u8], a: &[u8; 4]) -> u128 {
    let mut hist = [0u64; 256];
    for i in 0..sample.len() {
        let mut pred = 0u8;
        for j in 0..4 { if i > j { pred = pred.wrapping_add(a[j].wrapping_mul(sample[i - j - 1])); } }
        hist[sample[i].wrapping_sub(pred) as usize] += 1;
    }
    hist.iter().map(|&n| (n as u128) * (n as u128)).sum()
}

/// Hensel-style lifting over Z/256Z. The residual modulo 2^k depends only on the
/// coefficients modulo 2^k, so the coefficients are decided one bit level at a
/// time (beam search, scored by residual zeros modulo 2^k) instead of searching
/// 256^4 combinations. Deterministic: ties are broken by the coefficient array.
fn lift_algebraic_recurrence(sample: &[u8]) -> [u8; 4] {
    const LIFT_SAMPLE: usize = 1024;
    const BEAM: usize = 16;
    let search = &sample[..sample.len().min(LIFT_SAMPLE)];
    let mut beam: Vec<[u8; 4]> = vec![[0u8; 4]];
    for level in 1..=8u32 {
        let mask = ((1u16 << level) - 1) as u8;
        let bit = 1u8 << (level - 1);
        let mut scored: Vec<(usize, [u8; 4])> = Vec::with_capacity(beam.len() * 16);
        for base in &beam {
            for add in 0u8..16 {
                let mut c = *base;
                for j in 0..4 { if add & (1 << j) != 0 { c[j] |= bit; } }
                let mut zeros = 0usize;
                for i in 0..search.len() {
                    let mut pred = 0u8;
                    for j in 0..4 { if i > j { pred = pred.wrapping_add(c[j].wrapping_mul(search[i - j - 1])); } }
                    if (search[i].wrapping_sub(pred) & mask) == 0 { zeros += 1; }
                }
                scored.push((zeros, c));
            }
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(BEAM);
        beam = scored.into_iter().map(|(_, c)| c).collect();
    }
    let mut best: Option<([u8; 4], u128)> = None;
    for c in beam {
        let s = recurrence_concentration(sample, &c);
        if best.map_or(true, |(_, b)| s > b) { best = Some((c, s)); }
    }
    best.map(|(c, _)| c).unwrap_or([0u8; 4])
}

/// PS9.8: byte-lane transposition for fixed-size records. One parameter byte
/// holds the record stride (>= 2). Output is lane 0 (bytes 0, S, 2S, ...), then
/// lane 1, and so on, so every record field becomes one contiguous stream.
/// Length-preserving and exactly reversible for every input length.
#[derive(Debug, Default, Clone, Copy)]
pub struct LaneTranspose;

fn lane_stride(p: &[u8]) -> Result<usize, UmceError> {
    match p {
        [stride] if *stride >= 2 => Ok(*stride as usize),
        _ => Err(UmceError::InvalidParameters("lane transpose requires one stride byte >= 2")),
    }
}

impl ReversibleModel for LaneTranspose {
    fn id(&self) -> ModelId { MODEL_LANE_TRANSPOSE }
    fn name(&self) -> &'static str { "lane_transpose" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { lane_stride(p).map(|_| ()) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        let stride = lane_stride(p)?;
        let mut out = Vec::with_capacity(input.len());
        for lane in 0..stride { out.extend(input.iter().skip(lane).step_by(stride)); }
        Ok(TransformOutput { data: out, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        let stride = lane_stride(p)?;
        require_empty_aux(a)?;
        let mut out = vec![0u8; input.len()];
        let mut source = input.iter();
        for lane in 0..stride {
            for slot in out.iter_mut().skip(lane).step_by(stride) {
                if let Some(&byte) = source.next() { *slot = byte; }
            }
        }
        Ok(out)
    }
}

/// Detects a fixed record stride in 2..=64 from a prefix sample. For every lag
/// the histogram concentration of x[i] - x[i-lag] is computed (fixed point, 2^32
/// = all mass in one value). The smallest near-best lag wins (a later lag must
/// beat the current best by more than 1/16, so multiples of the true stride
/// lose), and it must at least double the better of the raw and lag-1 scores.
/// Encoder-side heuristic only: the archive never depends on it for decoding.
pub fn detect_record_stride(sample: &[u8]) -> Option<u8> {
    const MIN_SAMPLE: usize = 4096;
    const MAX_SAMPLE: usize = 65_536;
    const MAX_STRIDE: usize = 64;
    let sample = &sample[..sample.len().min(MAX_SAMPLE)];
    if sample.len() < MIN_SAMPLE { return None; }
    let n = sample.len();
    let concentration = |hist: &[u64; 256], m: usize| -> u128 {
        let squares: u128 = hist.iter().map(|&c| (c as u128) * (c as u128)).sum();
        (squares << 32) / ((m as u128) * (m as u128))
    };
    let mut raw_hist = [0u64; 256];
    for &b in sample { raw_hist[b as usize] += 1; }
    let lag_score = |lag: usize| -> u128 {
        let mut hist = [0u64; 256];
        for i in lag..n { hist[sample[i].wrapping_sub(sample[i - lag]) as usize] += 1; }
        concentration(&hist, n - lag)
    };
    let baseline = concentration(&raw_hist, n).max(lag_score(1));
    let mut best_score = 0u128;
    let mut best_stride = 0usize;
    for stride in 2..=MAX_STRIDE {
        let score = lag_score(stride);
        if best_stride == 0 || score.saturating_mul(16) > best_score.saturating_mul(17) {
            best_score = score;
            best_stride = stride;
        }
    }
    if best_stride != 0 && best_score >= baseline.saturating_mul(2) { Some(best_stride as u8) } else { None }
}

#[cfg(test)]
mod ps98_record_stride_tests {
    use super::*;

    fn xorshift(mut x: u32) -> impl FnMut() -> u32 {
        move || { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x }
    }

    fn synthetic_records(count: usize) -> Vec<u8> {
        let mut next = xorshift(0x1234_5678);
        let mut out = Vec::with_capacity(count * 16);
        for id in 0..count as u32 {
            out.extend_from_slice(&id.to_le_bytes());
            out.push(0xA5);
            out.push(((next() >> 24) as u8) & 3);
            out.extend_from_slice(&(((next() >> 16) as u16) % 500).to_le_bytes());
            out.extend_from_slice(&1_700_000_000u32.wrapping_add(id.wrapping_mul(3)).to_le_bytes());
            out.extend_from_slice(&next().to_le_bytes());
        }
        out
    }

    #[test]
    fn lane_transpose_layout_is_column_major() {
        let input: Vec<u8> = (0u8..11).collect();
        let out = LaneTranspose.forward(&input, &[4]).unwrap().data;
        assert_eq!(out, vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7]);
    }

    #[test]
    fn lane_transpose_roundtrips_all_lengths_and_strides() {
        for len in [0usize, 1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 1000, 4099] {
            let input: Vec<u8> = (0..len).map(|i| ((i * 37 + i / 3) & 255) as u8).collect();
            for stride in [2u8, 3, 4, 16, 64, 255] {
                let forward = LaneTranspose.forward(&input, &[stride]).unwrap();
                assert_eq!(forward.data.len(), input.len());
                assert_eq!(LaneTranspose.inverse(&forward.data, &[stride], &[]).unwrap(), input, "len {len} stride {stride}");
            }
        }
    }

    #[test]
    fn lane_transpose_rejects_bad_parameters_and_auxiliary_data() {
        for bad in [&[][..], &[0][..], &[1][..], &[2, 3][..]] {
            assert!(LaneTranspose.forward(&[1, 2, 3], bad).is_err());
            assert!(LaneTranspose.inverse(&[1, 2, 3], bad, &[]).is_err());
        }
        assert!(LaneTranspose.inverse(&[1, 2, 3], &[2], &[1]).is_err());
    }

    #[test]
    fn stride_detector_finds_record_size_and_rejects_unstructured_data() {
        assert_eq!(detect_record_stride(&synthetic_records(4096)), Some(16));
        let mut next = xorshift(2_463_534_242);
        let random: Vec<u8> = (0..65_536).map(|_| (next() >> 24) as u8).collect();
        assert_eq!(detect_record_stride(&random), None);
        assert_eq!(detect_record_stride(&vec![0u8; 65_536]), None);
        assert_eq!(detect_record_stride(&synthetic_records(62)), None);
    }

    #[test]
    fn lifting_finds_exact_recurrences_that_coordinate_ascent_misses() {
        let mut fib = vec![0u8, 1];
        while fib.len() < 8192 { let n = fib.len(); fib.push(fib[n - 1].wrapping_add(fib[n - 2])); }
        assert_eq!(fit_algebraic_recurrence(&fib), [1, 1, 0, 0]);

        let mut rec = vec![1u8, 5];
        while rec.len() < 8192 {
            let n = rec.len();
            rec.push(rec[n - 1].wrapping_mul(2).wrapping_add(rec[n - 2].wrapping_mul(3)));
        }
        assert_eq!(fit_algebraic_recurrence(&rec), [2, 3, 0, 0]);

        let mut geo = vec![1u8];
        while geo.len() < 8192 { let n = geo.len(); geo.push(geo[n - 1].wrapping_mul(3)); }
        assert_eq!(fit_algebraic_recurrence(&geo), [3, 0, 0, 0]);
    }
}
