//! PS8.6 bounded-window LZ77 with entropy-aware adaptive search profiles and cost-aware lazy parsing.
//!
//! The transform is byte-exact and self-describing. It emits literal runs and
//! back-references; the normal PYTHIA entropy selector then chooses the best
//! entropy backend for the token stream. The encoder uses a bounded hash-chain
//! search so worst-case match-finder work is linear in input length times a
//! fixed probe budget.

use crate::{model::{require_empty_aux, require_empty_params, ReversibleModel, TransformOutput}, ModelId, UmceError};

pub const MODEL_LZ77: ModelId = 5;
const MAGIC: &[u8; 4] = b"PYLZ";
const VERSION_V1: u8 = 1;
const VERSION: u8 = 2;
const HEADER: usize = 9;
const WINDOW: usize = u16::MAX as usize;
const MIN_MATCH: usize = 4;
const MAX_MATCH: usize = 130;
const HASH_SIZE: usize = 1 << 16;
const SEARCH_PROFILES: [usize; 3] = [32, 48, 64];
const PROFILE_SAMPLE: usize = 64 * 1024;
// One-step lazy parsing is deliberately bounded: only inspect the next byte.
const LAZY_LOOKAHEAD: usize = 1;
const NONE: u32 = u32::MAX;

#[derive(Debug, Default, Clone, Copy)]
pub struct Lz77;

fn hash3(input: &[u8], p: usize) -> usize {
    let x = input[p] as u32;
    let y = input[p + 1] as u32;
    let z = input[p + 2] as u32;
    (((x.wrapping_mul(0x1e35_a7bd)) ^ (y << 11) ^ (z << 3)) >> 16) as usize & (HASH_SIZE - 1)
}

fn put_offset(mut offset: usize, out: &mut Vec<u8>) {
    // Unsigned LEB128: distances below 128 need one byte, below 16,384
    // need two, and the full 16-bit window needs at most three.
    while offset >= 0x80 {
        out.push((offset as u8 & 0x7f) | 0x80);
        offset >>= 7;
    }
    out.push(offset as u8);
}

fn read_offset(input: &[u8], pos: &mut usize) -> Result<usize, UmceError> {
    let mut value = 0usize;
    for (index, shift) in [0u32, 7, 14].into_iter().enumerate() {
        let byte = *input.get(*pos)
            .ok_or(UmceError::InvalidDescriptor("truncated LZ77 offset"))?;
        *pos += 1;
        value |= ((byte & 0x7f) as usize) << shift;
        if byte & 0x80 == 0 {
            // Reject non-minimal encodings so the token stream has one
            // canonical representation for every distance.
            if (index > 0 && value < (1usize << (7 * index)))
                || value == 0 || value > WINDOW
            {
                return Err(UmceError::InvalidDescriptor("invalid or noncanonical LZ77 match offset"));
            }
            return Ok(value);
        }
    }
    Err(UmceError::InvalidDescriptor("overlong LZ77 offset"))
}

fn flush_literals(out: &mut Vec<u8>, literals: &mut Vec<u8>) {
    let mut start = 0;
    while start < literals.len() {
        let count = (literals.len() - start).min(128);
        out.push((count - 1) as u8);
        out.extend_from_slice(&literals[start..start + count]);
        start += count;
    }
    literals.clear();
}

fn offset_bytes(offset: usize) -> usize {
    if offset < 0x80 { 1 } else if offset < 0x4000 { 2 } else { 3 }
}

fn find_match(input: &[u8], pos: usize, head: &[u32], prev: &[u16], max_chain: usize) -> (usize, usize) {
    if pos + MIN_MATCH > input.len() { return (0, 0); }
    let h = hash3(input, pos);
    let mut candidate = head[h];
    let mut probes = 0usize;
    let mut best_len = 0usize;
    let mut best_offset = 0usize;
    while candidate != NONE && probes < max_chain {
        let c = candidate as usize;
        let distance = pos - c;
        if distance > WINDOW { break; }
        let mut len = 0usize;
        while len < MAX_MATCH && pos + len < input.len()
            && input[c + len] == input[pos + len]
        { len += 1; }
        // Hash chains are newest-first, so the first equal-length candidate
        // is also the nearest one and has minimal ULEB128 distance cost.
        if len >= MIN_MATCH && (len > best_len
            || (len == best_len && (offset_bytes(distance) < offset_bytes(best_offset)
                || (offset_bytes(distance) == offset_bytes(best_offset) && distance < best_offset))))
        {
            best_len = len;
            best_offset = distance;
            if len == MAX_MATCH { break; }
        }
        let delta = prev[c] as usize;
        candidate = if delta == 0 { NONE } else { (c - delta) as u32 };
        probes += 1;
    }
    (best_len, best_offset)
}

fn insert_position(input: &[u8], pos: usize, head: &mut [u32], prev: &mut [u16]) {
    if pos + 2 >= input.len() { return; }
    let h = hash3(input, pos);
    let previous = head[h];
    prev[pos] = if previous == NONE { 0 } else {
        let distance = pos - previous as usize;
        if distance <= WINDOW { distance as u16 } else { 0 }
    };
    head[h] = pos as u32;
}

fn encode_tokens_with_depth(input: &[u8], max_chain: usize) -> Result<Vec<u8>, UmceError> {
    let raw_len = u32::try_from(input.len()).map_err(|_| UmceError::LengthOverflow)?;
    let mut out = Vec::with_capacity(input.len().saturating_add(HEADER));
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&raw_len.to_le_bytes());

    let mut head = vec![NONE; HASH_SIZE];
    let mut prev = vec![0u16; input.len()];
    let mut literals = Vec::with_capacity(128);
    let mut pos = 0usize;

    while pos < input.len() {
        let (mut best_len, best_offset) = find_match(input, pos, &head, &prev, max_chain);
        // Lazy parsing: if starting one byte later yields a sufficiently longer
        // match, preserve this byte as a literal and reconsider at pos + 1.
        // This is bounded O(64 * MAX_MATCH) per input byte in the worst
        // case, with both constants fixed and the extra search only on matches.
        if best_len >= MIN_MATCH && best_len < 16 && LAZY_LOOKAHEAD == 1 && pos + 1 < input.len() {
            let (next_len, next_offset) = find_match(input, pos + 1, &head, &prev, max_chain);
            if next_len >= MIN_MATCH {
                // Compare net bytes saved by the current match with the
                // alternative: one literal byte followed by the next match.
                // Include the LEB128 distance cost on both paths.
                let literal_cost = if literals.is_empty() { 2 } else { 1 };
                let current_gain = best_len.saturating_sub(1 + offset_bytes(best_offset));
                let next_gain = next_len.saturating_sub(literal_cost + 1 + offset_bytes(next_offset));
                if next_gain > current_gain { best_len = 0; }
            }
        }

        if best_len >= MIN_MATCH {
            flush_literals(&mut out, &mut literals);
            // 128..255 encode lengths 3..130; length 3 is reserved/invalid.
            out.push((128 + best_len - 3) as u8);
            put_offset(best_offset, &mut out);
            let end = pos + best_len;
            while pos < end {
                insert_position(input, pos, &mut head, &mut prev);
                pos += 1;
            }
        } else {
            literals.push(input[pos]);
            insert_position(input, pos, &mut head, &mut prev);
            pos += 1;
            if literals.len() == 128 { flush_literals(&mut out, &mut literals); }
        }
    }
    flush_literals(&mut out, &mut literals);
    Ok(out)
}

// Pick a bounded search budget using a small prefix, then encode the complete
// block once. Score each candidate by the *complete* entropy frame, including
// PYAE/token-aware framing and its headers, rather than the intermediate PYLZ
// byte count. This is still a sample-based estimate, not a global optimum.
fn choose_search_profile(input: &[u8]) -> Result<usize, UmceError> {
    if input.len() < 4096 { return Ok(32); }
    let sample = &input[..input.len().min(PROFILE_SAMPLE)];
    let mut best_depth = SEARCH_PROFILES[0];
    let mut best_size = usize::MAX;
    for depth in SEARCH_PROFILES {
        let candidate = encode_tokens_with_depth(sample, depth)?;
        let size = crate::entropy::encode_best(&candidate)?.archive.len();
        if size < best_size {
            best_size = size;
            best_depth = depth;
        }
    }
    Ok(best_depth)
}

fn encode_tokens(input: &[u8]) -> Result<Vec<u8>, UmceError> {
    let depth = choose_search_profile(input)?;
    encode_tokens_with_depth(input, depth)
}

fn decode_tokens(input: &[u8]) -> Result<Vec<u8>, UmceError> {
    if input.len() < HEADER || &input[..4] != MAGIC
        || !matches!(input[4], VERSION_V1 | VERSION)
    {
        return Err(UmceError::InvalidDescriptor("bad or truncated LZ77 header"));
    }
    let stream_version = input[4];
    let raw_len = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;
    // The outer block header has already bounded the transformed frame. This
    // additional sanity bound prevents a tiny corrupt token frame requesting
    // an unbounded allocation.
    if raw_len > 64 * 1024 * 1024 {
        return Err(UmceError::InvalidDescriptor("LZ77 output limit exceeded"));
    }
    let mut out = Vec::with_capacity(raw_len);
    let mut pos = HEADER;
    while out.len() < raw_len {
        let tag = *input.get(pos).ok_or(UmceError::InvalidDescriptor("truncated LZ77 token"))?;
        pos += 1;
        if tag < 128 {
            let len = tag as usize + 1;
            let end = pos.checked_add(len).ok_or(UmceError::LengthOverflow)?;
            let bytes = input.get(pos..end).ok_or(UmceError::InvalidDescriptor("truncated LZ77 literal"))?;
            let new_len = out.len().checked_add(len).ok_or(UmceError::LengthOverflow)?;
            if new_len > raw_len { return Err(UmceError::InvalidDescriptor("LZ77 literal exceeds output length")); }
            out.extend_from_slice(bytes);
            pos = end;
        } else {
            let len = (tag as usize & 0x7f) + 3;
            if len < MIN_MATCH { return Err(UmceError::InvalidDescriptor("invalid LZ77 match length")); }
            let offset = if stream_version == VERSION_V1 {
                let end = pos.checked_add(2).ok_or(UmceError::LengthOverflow)?;
                let bytes = input.get(pos..end)
                    .ok_or(UmceError::InvalidDescriptor("truncated LZ77 offset"))?;
                pos = end;
                u16::from_le_bytes([bytes[0], bytes[1]]) as usize
            } else {
                read_offset(input, &mut pos)?
            };
            if offset == 0 || offset > out.len() {
                return Err(UmceError::InvalidDescriptor("invalid LZ77 match offset"));
            }
            let new_len = out.len().checked_add(len).ok_or(UmceError::LengthOverflow)?;
            if new_len > raw_len { return Err(UmceError::InvalidDescriptor("LZ77 match exceeds output length")); }
            for _ in 0..len {
                let b = out[out.len() - offset];
                out.push(b);
            }
        }
    }
    if pos != input.len() { return Err(UmceError::InvalidDescriptor("trailing bytes after LZ77 stream")); }
    Ok(out)
}

impl ReversibleModel for Lz77 {
    fn id(&self) -> ModelId { MODEL_LZ77 }
    fn name(&self) -> &'static str { "lz77_hash_chain" }
    fn validate_parameters(&self, p: &[u8]) -> Result<(), UmceError> { require_empty_params(p) }
    fn forward(&self, input: &[u8], p: &[u8]) -> Result<TransformOutput, UmceError> {
        self.validate_parameters(p)?;
        Ok(TransformOutput { data: encode_tokens(input)?, auxiliary: Vec::new() })
    }
    fn inverse(&self, input: &[u8], p: &[u8], a: &[u8]) -> Result<Vec<u8>, UmceError> {
        self.validate_parameters(p)?;
        require_empty_aux(a)?;
        decode_tokens(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_repetitive_and_varied_inputs() {
        let mut samples = vec![Vec::new(), vec![7], vec![0; 100_000], b"abcabcabcabcabcabc".to_vec()];
        samples.push((0..=255).cycle().take(65_537).collect());
        samples.push((0..100_000).map(|i| ((i * 73 + i / 11) & 255) as u8).collect());
        for input in samples {
            let encoded = encode_tokens(&input).unwrap();
            assert_eq!(encoded[4], VERSION);
            assert_eq!(decode_tokens(&encoded).unwrap(), input);
        }
    }

    #[test]
    fn rejects_malformed_streams() {
        for bad in [
            b"".as_slice(),
            b"PYLZ\x01\x01\x00\x00\x00\xff",
            b"PYLZ\x01\x00\x00\x00\x00\x80\x01\x00",
            b"PYLZ\x01\x04\x00\x00\x00\x81\x01\x00",
        ] {
            assert!(decode_tokens(bad).is_err());
        }
    }

    #[test]
    fn variable_offsets_roundtrip_across_encoding_boundaries() {
        for offset in [1usize, 127, 128, 16_383, 16_384, 65_535] {
            let mut encoded = Vec::new();
            put_offset(offset, &mut encoded);
            let mut pos = 0;
            assert_eq!(read_offset(&encoded, &mut pos).unwrap(), offset);
            assert_eq!(pos, encoded.len());
        }
        let mut pos = 0;
        assert!(read_offset(&[0x80, 0x80, 0x80], &mut pos).is_err());
    }

    #[test]
    fn decodes_legacy_v1_fixed_width_offsets() {
        // Original length=8, four literals, then a match of length 4 at offset 4.
        let legacy = b"PYLZ\x01\x08\x00\x00\x00\x03abcd\x81\x04\x00";
        assert_eq!(decode_tokens(legacy).unwrap(), b"abcdabcd");
    }

    #[test]
    fn rejects_noncanonical_variable_offsets() {
        // The value 4 must be encoded as one byte (0x04), not 0x84 0x00.
        let bad = b"PYLZ\x02\x08\x00\x00\x00\x03abcd\x81\x84\x00";
        assert!(decode_tokens(bad).is_err());
    }

    #[test]
    fn handles_references_beyond_the_window_without_corruption() {
        let mut input: Vec<u8> = (0..100_000).map(|i| ((i * 37 + i / 13) & 255) as u8).collect();
        let prefix = input[..512].to_vec();
        input.extend_from_slice(&prefix);
        let encoded = encode_tokens(&input).unwrap();
        assert_eq!(decode_tokens(&encoded).unwrap(), input);
    }
    #[test]
    fn lazy_parser_roundtrips_overlapping_references() {
        for input in [
            b"abcXabcYabcXabcYabcXabcY".as_slice(),
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".as_slice(),
            b"abracadabra abracadabra abracadabra".as_slice(),
        ] {
            let encoded = encode_tokens(input).unwrap();
            assert_eq!(decode_tokens(&encoded).unwrap(), input);
            assert_eq!(encode_tokens(input).unwrap(), encoded);
        }
    }

    #[test]
    fn offset_cost_uses_leb128_boundaries() {
        assert_eq!(offset_bytes(1), 1);
        assert_eq!(offset_bytes(127), 1);
        assert_eq!(offset_bytes(128), 2);
        assert_eq!(offset_bytes(16_383), 2);
        assert_eq!(offset_bytes(16_384), 3);
        assert_eq!(offset_bytes(65_535), 3);
    }


    #[test]
    fn ps86_search_profiles_are_bounded_and_entropy_scored() {
        assert_eq!(*SEARCH_PROFILES.last().unwrap(), 64);
        assert_eq!(SEARCH_PROFILES, [32, 48, 64]);
        assert_eq!(PROFILE_SAMPLE, 65_536);
        assert_eq!(choose_search_profile(&vec![0u8; 1024]).unwrap(), 32);
        assert_eq!(WINDOW, 65_535);
    }

    #[test]
    fn ps86_profile_selection_is_deterministic() {
        let input: Vec<u8> = (0..100_000).map(|i| ((i * 73 + i / 11) & 255) as u8).collect();
        assert_eq!(choose_search_profile(&input).unwrap(), choose_search_profile(&input).unwrap());
    }

}
