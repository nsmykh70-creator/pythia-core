//! Shared block representation search used by both single-file and multi-file containers.
use std::io;
use crate::{
    chain::{ModelChain, Stage},
    descriptor::ChainDescriptor,
    domain::Domain,
    entropy::encode_best,
    transforms::{MODEL_BITPLANE8, MODEL_DELTA8, MODEL_DELTA2, MODEL_DELTA3, MODEL_ADAPTIVE_PREDICTOR, MODEL_WEIGHTED_MEDOID, MODEL_PROBABILISTIC_MIXTURE, MODEL_LZ77, MODEL_XOR_PREVIOUS, MODEL_ZIGZAG8, MODEL_ALGEBRAIC_RECURRENCE, fit_algebraic_recurrence, MODEL_LANE_TRANSPOSE, detect_record_stride},
};
use super::format::{max_transformed_block_len, MAX_DESCRIPTOR_LEN};

/// Uses a deterministic 64 KiB prefix sample to rank transform candidates,
/// then fully encodes the identity candidate and the four best transforms.
/// Final selection always compares exact descriptor + complete entropy-frame
/// lengths on the full block. The sample is a search heuristic, not a size claim.
pub(crate) fn select_block_representation(input: &[u8]) -> io::Result<(Vec<u8>, Vec<u8>)> {
    const SAMPLE_LIMIT: usize = 64 * 1024;
    const TRANSFORMS_TO_FULLY_ENCODE: usize = 4;

    let sample = &input[..input.len().min(SAMPLE_LIMIT)];
    let model_ids = [MODEL_DELTA8, MODEL_XOR_PREVIOUS, MODEL_ZIGZAG8, MODEL_BITPLANE8, MODEL_DELTA2, MODEL_DELTA3, MODEL_ADAPTIVE_PREDICTOR, MODEL_WEIGHTED_MEDOID, MODEL_PROBABILISTIC_MIXTURE];
    let mut ranked: Vec<(usize, usize, Vec<Stage>)> = Vec::with_capacity(16);
    let mut ordinal = 0usize;

    for depth in 1..=2 {
        let combinations = model_ids.len().pow(depth as u32);
        for encoded_ordinal in 0..combinations {
            let mut n = encoded_ordinal;
            let mut stages = Vec::with_capacity(depth);
            for _ in 0..depth {
                let id = model_ids[n % model_ids.len()];
                n /= model_ids.len();
                stages.push(Stage { model_id: id, model_version: 1, parameters: Vec::new() });
            }
            // Adjacent repetitions add cost without introducing a distinct
            // model interaction; exclude them from the bounded search space.
            if stages.windows(2).any(|pair| pair[0].model_id == pair[1].model_id) {
                ordinal += 1;
                continue;
            }
            let chain = ModelChain::new(stages.clone());
            let transformed_sample = chain.forward(sample).map_err(model_error)?;
            let descriptor = ChainDescriptor::from_chain(&chain, Domain::Bytes, Domain::Bytes)
                .encode().map_err(model_error)?;
            if descriptor.len() > MAX_DESCRIPTOR_LEN as usize {
                ordinal += 1;
                continue;
            }
            let sample_frame = encode_best(&transformed_sample.data).map_err(entropy_error)?;
            let score = descriptor.len().checked_add(sample_frame.archive.len())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "sample candidate size overflow"))?;
            ranked.push((score, ordinal, stages));
            ordinal += 1;
        }
    }

    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Identity is always a full-data candidate and guarantees that transform
    // search cannot force a larger representation than the baseline.
    let baseline = encode_best(input).map_err(entropy_error)?;
    let mut best_descriptor = Vec::new();
    let mut best_frame = baseline.archive;
    let mut best_cost = best_frame.len();

    // PS8.0: always test the standalone LZ77 candidate on the full block.
    // A bounded prefix can miss repeated records or dictionaries later in a block.
    let lz_chain = ModelChain::new(vec![Stage { model_id: MODEL_LZ77, model_version: 1, parameters: Vec::new() }]);
    let lz_transformed = lz_chain.forward(input).map_err(model_error)?;
    let lz_descriptor = ChainDescriptor::from_chain(&lz_chain, Domain::Bytes, Domain::Bytes)
        .encode().map_err(model_error)?;
    // The encoder does not guarantee the nominal LZ77 worst case: a 4-byte match
    // at distance >= 16384 costs 4 bytes and splits literal runs. Readers reject
    // transformed streams longer than max_transformed_block_len, so the writer
    // applies the very same bound instead of trusting the encoder.
    let transformed_limit = max_transformed_block_len(u32::try_from(input.len()).unwrap_or(u32::MAX));
    if lz_descriptor.len() <= MAX_DESCRIPTOR_LEN as usize && lz_transformed.data.len() <= transformed_limit {
        let lz_encoded = encode_best(&lz_transformed.data).map_err(entropy_error)?;
        let lz_cost = lz_descriptor.len().checked_add(lz_encoded.archive.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "LZ77 candidate size overflow"))?;
        if lz_cost < best_cost {
            best_cost = lz_cost;
            best_descriptor = lz_descriptor;
            best_frame = lz_encoded.archive;
        }
    }

    // PS9.3: always evaluate the fitted algebraic recurrence on the complete
    // block. A prefix ranking must not hide a model that captures later data.
    let algebraic_params = fit_algebraic_recurrence(sample).to_vec();
    let algebraic_chain = ModelChain::new(vec![Stage { model_id: MODEL_ALGEBRAIC_RECURRENCE, model_version: 1, parameters: algebraic_params }]);
    let algebraic_transformed = algebraic_chain.forward(input).map_err(model_error)?;
    let algebraic_descriptor = ChainDescriptor::from_chain(&algebraic_chain, Domain::Bytes, Domain::Bytes)
        .encode().map_err(model_error)?;
    if algebraic_descriptor.len() <= MAX_DESCRIPTOR_LEN as usize {
        let algebraic_encoded = encode_best(&algebraic_transformed.data).map_err(entropy_error)?;
        let algebraic_cost = algebraic_descriptor.len().checked_add(algebraic_encoded.archive.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "algebraic candidate size overflow"))?;
        if algebraic_cost < best_cost {
            best_cost = algebraic_cost;
            best_descriptor = algebraic_descriptor;
            best_frame = algebraic_encoded.archive;
        }
    }

    // PS9.8: fixed-size record data. Transpose bytes into per-field lanes (and
    // optionally difference inside each lane); exact full-block cost decides.
    if let Some(stride) = detect_record_stride(sample) {
        for with_delta in [false, true] {
            let mut stages = vec![Stage { model_id: MODEL_LANE_TRANSPOSE, model_version: 1, parameters: vec![stride] }];
            if with_delta {
                stages.push(Stage { model_id: MODEL_DELTA8, model_version: 1, parameters: Vec::new() });
            }
            if let Some((cost, descriptor, frame)) = evaluate_full_candidate(input, stages)? {
                if cost < best_cost {
                    best_cost = cost;
                    best_descriptor = descriptor;
                    best_frame = frame;
                }
            }
        }
    }

    for (_, _, stages) in ranked.into_iter().take(TRANSFORMS_TO_FULLY_ENCODE) {
        let chain = ModelChain::new(stages);
        let transformed = chain.forward(input).map_err(model_error)?;
        let descriptor = ChainDescriptor::from_chain(&chain, Domain::Bytes, Domain::Bytes)
            .encode().map_err(model_error)?;
        if descriptor.len() > MAX_DESCRIPTOR_LEN as usize { continue; }
        let encoded = encode_best(&transformed.data).map_err(entropy_error)?;
        let candidate_cost = descriptor.len().checked_add(encoded.archive.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "candidate size overflow"))?;
        if candidate_cost < best_cost {
            best_cost = candidate_cost;
            best_descriptor = descriptor;
            best_frame = encoded.archive;
        }
    }
    Ok((best_descriptor, best_frame))
}

/// Compresses independent blocks concurrently while preserving input order.
/// At most `workers` raw blocks and their results are resident at once.
pub(crate) fn select_block_representations_parallel(
    blocks: Vec<Vec<u8>>,
    workers: usize,
) -> io::Result<Vec<(Vec<u8>, (Vec<u8>, Vec<u8>))>> {
    if workers == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "worker count must be nonzero"));
    }
    let mut pending = blocks.into_iter();
    let mut output = Vec::new();
    loop {
        let batch: Vec<Vec<u8>> = pending.by_ref().take(workers).collect();
        if batch.is_empty() { break; }
        let mut results = Vec::with_capacity(batch.len());
        std::thread::scope(|scope| {
            let handles: Vec<_> = batch.into_iter().map(|block| {
                scope.spawn(move || {
                    let encoded = select_block_representation(&block)?;
                    Ok::<_, io::Error>((block, encoded))
                })
            }).collect();
            for handle in handles {
                let result = handle.join().map_err(|_| io::Error::new(io::ErrorKind::Other, "compression worker panicked"))?;
                results.push(result?);
            }
            Ok::<(), io::Error>(())
        })?;
        output.extend(results);
    }
    Ok(output)
}

/// Selects a per-file block size from a bounded prefix sample.
/// Low-entropy inputs use larger blocks; high-entropy inputs use smaller
/// blocks to cap transient memory and reduce wasted transform search.
pub(crate) fn adaptive_block_size(sample: &[u8], configured: usize) -> usize {
    let minimum = 256 * 1024usize;
    let maximum = 16 * 1024 * 1024usize;
    let base = configured.clamp(1, maximum);
    if sample.is_empty() { return base; }
    let mut histogram = [0u32; 256];
    for &byte in sample { histogram[byte as usize] += 1; }
    let n = sample.len() as f64;
    let entropy = histogram.iter().filter(|&&count| count != 0).map(|&count| {
        let p = count as f64 / n;
        -p * p.log2()
    }).sum::<f64>();
    if entropy < 2.0 { base.max(4 * 1024 * 1024).min(maximum) }
    else if entropy < 6.5 { base }
    else { base.min(512 * 1024).max(minimum.min(base)) }
}


/// Fully evaluates one transform chain on the complete block: exact descriptor
/// plus entropy-frame size, or None when the chain cannot be represented or
/// could not be read back (descriptor or transformed-stream bound).
fn evaluate_full_candidate(input: &[u8], stages: Vec<Stage>) -> io::Result<Option<(usize, Vec<u8>, Vec<u8>)>> {
    let chain = ModelChain::new(stages);
    let transformed = chain.forward(input).map_err(model_error)?;
    let descriptor = ChainDescriptor::from_chain(&chain, Domain::Bytes, Domain::Bytes)
        .encode().map_err(model_error)?;
    let limit = max_transformed_block_len(u32::try_from(input.len()).unwrap_or(u32::MAX));
    if descriptor.len() > MAX_DESCRIPTOR_LEN as usize || transformed.data.len() > limit {
        return Ok(None);
    }
    let encoded = encode_best(&transformed.data).map_err(entropy_error)?;
    let cost = descriptor.len().checked_add(encoded.archive.len())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "candidate size overflow"))?;
    Ok(Some((cost, descriptor, encoded.archive)))
}

fn entropy_error<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("entropy encoder failed: {e}"))
}
fn model_error<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("transform failed: {e}"))
}

#[cfg(test)]
mod ps82_lz77_expansion_tests {
    use super::*;
    use crate::{
        chain::{ChainResult, ModelChain},
        entropy::{decode_best, encode_best},
    };

    #[test]
    fn expanded_lz77_stream_decodes_with_a_tight_bounded_limit() {
        let input: Vec<u8> = (0..32_768u32)
            .map(|mut x| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xff) as u8
            })
            .collect();
        let chain = ModelChain::new(vec![Stage { model_id: MODEL_LZ77, model_version: 1, parameters: Vec::new() }]);
        let transformed = chain.forward(&input).unwrap();
        assert!(transformed.data.len() > input.len(), "literal-heavy PYLZ must exercise transform expansion");
        let selected = encode_best(&transformed.data).unwrap();
        assert!(decode_best(&selected.archive, input.len()).is_err(), "raw block length alone is too small for the transformed stream");
        let limit = super::super::format::max_transformed_block_len(input.len() as u32);
        let restored_stream = decode_best(&selected.archive, limit).unwrap();
        let restored = chain.inverse(&ChainResult { data: restored_stream, auxiliary: vec![Vec::new()] }).unwrap();
        assert_eq!(restored, input);
    }
}

#[cfg(test)]
mod ps97_selected_block_bound_tests {
    use super::*;
    use super::super::format::decode_block_descriptor;
    use crate::{chain::ChainResult, entropy::decode_best};

    fn xorshift_bytes(len: usize, mut x: u32) -> Vec<u8> {
        (0..len).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; (x >> 24) as u8 }).collect()
    }

    #[test]
    fn selected_frames_always_fit_the_reader_transform_bound() {
        // Random prefix, then groups of [novel byte][4 bytes copied from far back].
        // A 4-byte match at distance >= 16384 costs 4 bytes, so LZ77 can exceed
        // its nominal literal-only worst case on such input.
        let base = xorshift_bytes(20_000, 12_345);
        let noise = xorshift_bytes(40_000, 777);
        let mut input = base.clone();
        let mut k = 0usize;
        while input.len() < 60_000 {
            input.push(noise[k]);
            let src = (noise[k + 1] as usize) * 13 % 3_500;
            input.extend_from_slice(&base[src..src + 4]);
            k += 2;
        }
        let (descriptor, frame) = select_block_representation(&input).unwrap();
        let limit = max_transformed_block_len(input.len() as u32);
        let transformed = decode_best(&frame, limit).expect("writer must not emit frames the reader rejects");
        let restored = if descriptor.is_empty() {
            transformed
        } else {
            let desc = decode_block_descriptor(&descriptor).unwrap();
            let count = desc.stages.len();
            ModelChain::new(desc.stages)
                .inverse(&ChainResult { data: transformed, auxiliary: vec![Vec::new(); count] })
                .unwrap()
        };
        assert_eq!(restored, input);
    }
}

#[cfg(test)]
mod ps98_record_pipeline_tests {
    use super::*;
    use super::super::format::decode_block_descriptor;
    use crate::{chain::ChainResult, entropy::decode_best};

    fn records(count: usize) -> Vec<u8> {
        let mut x = 0x1234_5678u32;
        let mut next = move || { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x };
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
    fn record_blocks_roundtrip_through_the_selected_representation() {
        let input = records(4096);
        let (descriptor, frame) = select_block_representation(&input).unwrap();
        let limit = max_transformed_block_len(input.len() as u32);
        let transformed = decode_best(&frame, limit).unwrap();
        let restored = if descriptor.is_empty() {
            transformed
        } else {
            let desc = decode_block_descriptor(&descriptor).unwrap();
            let count = desc.stages.len();
            ModelChain::new(desc.stages)
                .inverse(&ChainResult { data: transformed, auxiliary: vec![Vec::new(); count] })
                .unwrap()
        };
        assert_eq!(restored, input);
    }
}
