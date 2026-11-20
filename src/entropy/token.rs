//! PS8.2 token-aware entropy framing for the reversible PYLZ stream.
//!
//! The PYLZ token grammar is split into four independently coded lanes:
//! token kinds, token lengths, match offsets, and literal bytes. Each lane is
//! encoded by the existing entropy selector. The outer frame is selected only
//! when its complete byte length beats every ordinary entropy frame.

use crate::UmceError;
use super::{decode_best, encode_best_plain};

const MAGIC: &[u8; 4] = b"PYAE";
const FRAME_VERSION: u8 = 5;
const LANE_COUNT: usize = 6;
const HEADER_LEN: usize = 39; // common 14-byte header + PYLZ version + 6 u32 sizes
const PYLZ_MAGIC: &[u8; 4] = b"PYLZ";
const PYLZ_HEADER: usize = 9;
const MAX_PYLZ_OUTPUT: usize = 64 * 1024 * 1024;

#[derive(Default)]
struct Lanes {
    kinds: Vec<u8>,
    literal_lengths: Vec<u8>,
    match_lengths: Vec<u8>,
    offset_classes: Vec<u8>,
    offset_residuals: Vec<u8>, // u16 LE residual from 2^class
    literals: Vec<u8>,
    pylz_version: u8,
}

fn read_uleb(input: &[u8], pos: &mut usize) -> Result<usize, UmceError> {
    let mut value = 0usize;
    for (i, shift) in [0u32, 7, 14].into_iter().enumerate() {
        let b = *input.get(*pos).ok_or(UmceError::InvalidDescriptor("truncated PYLZ offset"))?;
        *pos = (*pos).checked_add(1).ok_or(UmceError::LengthOverflow)?;
        value |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            if value == 0 || value > u16::MAX as usize || (i > 0 && value < (1usize << (7 * i))) {
                return Err(UmceError::InvalidDescriptor("invalid or noncanonical PYLZ offset"));
            }
            return Ok(value);
        }
    }
    Err(UmceError::InvalidDescriptor("overlong PYLZ offset"))
}

fn put_uleb(mut value: usize, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn parse_pylz(input: &[u8]) -> Result<Option<Lanes>, UmceError> {
    if input.len() < PYLZ_HEADER || &input[..4] != PYLZ_MAGIC || !matches!(input[4], 1 | 2) {
        return Ok(None);
    }
    let version = input[4];
    let original_len = u32::from_le_bytes([input[5], input[6], input[7], input[8]]);
    if original_len as usize > MAX_PYLZ_OUTPUT {
        return Err(UmceError::InvalidDescriptor("PYLZ output limit exceeded"));
    }
    let mut lanes = Lanes { pylz_version: version, ..Lanes::default() };
    let mut pos = PYLZ_HEADER;
    let mut produced = 0usize;
    while produced < original_len as usize {
        let tag = *input.get(pos).ok_or(UmceError::InvalidDescriptor("truncated PYLZ token"))?;
        pos = pos.checked_add(1).ok_or(UmceError::LengthOverflow)?;
        if tag < 128 {
            let len = tag as usize + 1;
            let end = pos.checked_add(len).ok_or(UmceError::LengthOverflow)?;
            let literal = input.get(pos..end).ok_or(UmceError::InvalidDescriptor("truncated PYLZ literal"))?;
            produced = produced.checked_add(len).ok_or(UmceError::LengthOverflow)?;
            if produced > original_len as usize { return Err(UmceError::InvalidDescriptor("PYLZ literal exceeds output length")); }
            lanes.kinds.push(0);
            lanes.literal_lengths.push(tag);
            lanes.literals.extend_from_slice(literal);
            pos = end;
        } else {
            let len = (tag as usize & 0x7f) + 3;
            if len < 4 { return Err(UmceError::InvalidDescriptor("invalid PYLZ match length")); }
            let offset = if version == 1 {
                let end = pos.checked_add(2).ok_or(UmceError::LengthOverflow)?;
                let bytes = input.get(pos..end).ok_or(UmceError::InvalidDescriptor("truncated PYLZ offset"))?;
                pos = end;
                u16::from_le_bytes([bytes[0], bytes[1]]) as usize
            } else {
                read_uleb(input, &mut pos)?
            };
            if offset == 0 || offset > produced || offset > u16::MAX as usize {
                return Err(UmceError::InvalidDescriptor("invalid PYLZ match offset"));
            }
            produced = produced.checked_add(len).ok_or(UmceError::LengthOverflow)?;
            if produced > original_len as usize { return Err(UmceError::InvalidDescriptor("PYLZ match exceeds output length")); }
            lanes.kinds.push(1);
            lanes.match_lengths.push((len - 3) as u8);
            let class = (usize::BITS - 1 - (offset as usize).leading_zeros()) as u8;
            let base = 1usize << class;
            lanes.offset_classes.push(class);
            lanes.offset_residuals.extend_from_slice(&((offset - base) as u16).to_le_bytes());
        }
    }
    if pos != input.len() { return Err(UmceError::InvalidDescriptor("trailing bytes after PYLZ stream")); }
    Ok(Some(lanes))
}

fn encode_section(section: &[u8]) -> Result<Vec<u8>, UmceError> {
    Ok(encode_best_plain(section)?.archive)
}

fn decode_section(frame: &[u8], max_output_len: usize) -> Result<Vec<u8>, UmceError> {
    // Token frames contain only ordinary entropy frames in their lanes. Reject
    // nested v5 frames to keep decoder recursion depth constant.
    if frame.get(4) == Some(&FRAME_VERSION) {
        return Err(UmceError::InvalidDescriptor("nested token-aware lane frame"));
    }
    decode_best(frame, max_output_len)
}

pub(super) fn encode_frame(input: &[u8]) -> Result<Option<Vec<u8>>, UmceError> {
    let lanes = match parse_pylz(input) {
        Ok(Some(x)) => x,
        Ok(None) | Err(_) => return Ok(None), // non-PYLZ or malformed magic is ordinary byte data
    };
    let six = encode_frame_six(&lanes, input.len())?;
    let four = encode_frame_four(&lanes, input.len())?;
    Ok(Some(if four.len() < six.len() { four } else { six }))
}

fn encode_frame_six(lanes: &Lanes, input_len: usize) -> Result<Vec<u8>, UmceError> {
    let sections = [
        encode_section(&lanes.kinds)?,
        encode_section(&lanes.literal_lengths)?,
        encode_section(&lanes.match_lengths)?,
        encode_section(&lanes.offset_classes)?,
        encode_section(&lanes.offset_residuals)?,
        encode_section(&lanes.literals)?,
    ];
    let section_bytes = sections.iter().try_fold(0usize, |sum, section| {
        sum.checked_add(section.len()).ok_or(UmceError::LengthOverflow)
    })?;
    let capacity = HEADER_LEN.checked_add(section_bytes).ok_or(UmceError::LengthOverflow)?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(MAGIC);
    out.push(FRAME_VERSION);
    out.push(0); // six-lane profile
    out.extend_from_slice(&(input_len as u64).to_le_bytes());
    out.push(lanes.pylz_version);
    for section in &sections {
        let len = u32::try_from(section.len()).map_err(|_| UmceError::LengthOverflow)?;
        out.extend_from_slice(&len.to_le_bytes());
    }
    for section in sections { out.extend_from_slice(&section); }
    Ok(out)
}


fn encode_frame_four(lanes: &Lanes, input_len: usize) -> Result<Vec<u8>, UmceError> {
    let mut lengths = Vec::with_capacity(lanes.kinds.len());
    let mut offsets = Vec::with_capacity(lanes.offset_classes.len() * 2);
    let mut literal_pos = 0usize;
    let mut match_pos = 0usize;
    let mut class_pos = 0usize;
    let mut residual_pos = 0usize;
    for &kind in &lanes.kinds {
        if kind == 0 {
            lengths.push(*lanes.literal_lengths.get(literal_pos)
                .ok_or(UmceError::InvalidDescriptor("literal length lane exhausted"))?);
            literal_pos += 1;
        } else {
            lengths.push(*lanes.match_lengths.get(match_pos)
                .ok_or(UmceError::InvalidDescriptor("match length lane exhausted"))?);
            match_pos += 1;
            let class = *lanes.offset_classes.get(class_pos)
                .ok_or(UmceError::InvalidDescriptor("offset class lane exhausted"))? as usize;
            class_pos += 1;
            let end = residual_pos.checked_add(2).ok_or(UmceError::LengthOverflow)?;
            let residual_bytes = lanes.offset_residuals.get(residual_pos..end)
                .ok_or(UmceError::InvalidDescriptor("offset residual lane exhausted"))?;
            residual_pos = end;
            if class > 15 { return Err(UmceError::InvalidDescriptor("invalid offset class")); }
            let residual = u16::from_le_bytes([residual_bytes[0], residual_bytes[1]]) as usize;
            let base = 1usize << class;
            if residual >= base { return Err(UmceError::InvalidDescriptor("noncanonical offset residual")); }
            let offset = base.checked_add(residual).ok_or(UmceError::LengthOverflow)?;
            offsets.extend_from_slice(&(offset as u16).to_le_bytes());
        }
    }
    if literal_pos != lanes.literal_lengths.len() || match_pos != lanes.match_lengths.len()
        || class_pos != lanes.offset_classes.len() || residual_pos != lanes.offset_residuals.len()
    { return Err(UmceError::InvalidDescriptor("unused token metadata lanes")); }
    let sections = [
        encode_section(&lanes.kinds)?,
        encode_section(&lengths)?,
        encode_section(&offsets)?,
        encode_section(&lanes.literals)?,
    ];
    let section_bytes = sections.iter().try_fold(0usize, |sum, section| {
        sum.checked_add(section.len()).ok_or(UmceError::LengthOverflow)
    })?;
    let header_len = 31usize;
    let capacity = header_len.checked_add(section_bytes).ok_or(UmceError::LengthOverflow)?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(MAGIC);
    out.push(FRAME_VERSION);
    out.push(1); // four-lane compact profile
    out.extend_from_slice(&(input_len as u64).to_le_bytes());
    out.push(lanes.pylz_version);
    for section in &sections {
        let len = u32::try_from(section.len()).map_err(|_| UmceError::LengthOverflow)?;
        out.extend_from_slice(&len.to_le_bytes());
    }
    for section in sections { out.extend_from_slice(&section); }
    Ok(out)
}

fn decode_frame_six(input: &[u8], max_output_len: usize) -> Result<Vec<u8>, UmceError> {
    if input.len() < HEADER_LEN || &input[..4] != MAGIC || input[4] != FRAME_VERSION || input[5] != 0 {
        return Err(UmceError::InvalidDescriptor("bad token-aware entropy frame"));
    }
    let out_len_u64 = u64::from_le_bytes(input[6..14].try_into().map_err(|_| UmceError::InvalidDescriptor("bad token-aware length"))?);
    let out_len = usize::try_from(out_len_u64).map_err(|_| UmceError::LengthOverflow)?;
    if out_len > max_output_len { return Err(UmceError::InvalidDescriptor("token-aware output limit exceeded")); }
    if out_len < PYLZ_HEADER { return Err(UmceError::InvalidDescriptor("token-aware PYLZ stream is too short")); }
    let pylz_version = input[14];
    if !matches!(pylz_version, 1 | 2) { return Err(UmceError::InvalidDescriptor("unsupported PYLZ version in token frame")); }
    let mut pos = 15usize;
    let mut section_lengths = Vec::with_capacity(LANE_COUNT);
    for _ in 0..LANE_COUNT {
        let end = pos.checked_add(4).ok_or(UmceError::LengthOverflow)?;
        let bytes = input.get(pos..end).ok_or(UmceError::InvalidDescriptor("truncated token lane sizes"))?;
        let len = u32::from_le_bytes(bytes.try_into().map_err(|_| UmceError::InvalidDescriptor("bad token lane size"))?) as usize;
        section_lengths.push(len);
        pos = end;
    }
    let mut ranges = Vec::with_capacity(LANE_COUNT);
    for len in section_lengths {
        let end = pos.checked_add(len).ok_or(UmceError::LengthOverflow)?;
        if end > input.len() { return Err(UmceError::InvalidDescriptor("truncated token lane")); }
        ranges.push(pos..end);
        pos = end;
    }
    if pos != input.len() { return Err(UmceError::InvalidDescriptor("trailing bytes after token lanes")); }
    // The six normalized lanes may exceed the serialized PYLZ stream because
    // offsets are expanded to class + u16 residual. Bound their aggregate
    // decoded allocation rather than granting every lane the full block limit.
    let mut lane_budget = out_len.checked_mul(3).ok_or(UmceError::LengthOverflow)?;
    let mut lanes = Vec::with_capacity(LANE_COUNT);
    for range in ranges {
        let lane = decode_section(&input[range], lane_budget)?;
        lane_budget = lane_budget.checked_sub(lane.len())
            .ok_or(UmceError::InvalidDescriptor("token lane memory budget exceeded"))?;
        lanes.push(lane);
    }
    let [kinds, literal_lengths, match_lengths, offset_classes, offset_residuals, literals] =
        <[Vec<u8>; LANE_COUNT]>::try_from(lanes)
            .map_err(|_| UmceError::InvalidDescriptor("wrong token lane count"))?;

    let mut out = Vec::with_capacity(out_len);
    out.extend_from_slice(PYLZ_MAGIC);
    out.push(pylz_version);
    out.extend_from_slice(&[0u8; 4]);
    // The inner PYLZ header's raw length is not separately stored in the outer
    // frame, so it is recovered from the sum of the token lane semantics below.
    let mut literal_length_pos = 0usize;
    let mut match_length_pos = 0usize;
    let mut class_pos = 0usize;
    let mut residual_pos = 0usize;
    let mut literal_pos = 0usize;
    let mut original_len = 0usize;
    let mut token_bytes = Vec::new();
    for &kind in &kinds {
        if kind == 0 {
            let length = *literal_lengths.get(literal_length_pos)
                .ok_or(UmceError::InvalidDescriptor("literal length lane exhausted"))?;
            literal_length_pos += 1;
            token_bytes.push(length);
            let count = length as usize + 1;
            let end = literal_pos.checked_add(count).ok_or(UmceError::LengthOverflow)?;
            let bytes = literals.get(literal_pos..end).ok_or(UmceError::InvalidDescriptor("literal lane exhausted"))?;
            token_bytes.extend_from_slice(bytes);
            literal_pos = end;
            original_len = original_len.checked_add(count).ok_or(UmceError::LengthOverflow)?;
        } else if kind == 1 {
            let length = *match_lengths.get(match_length_pos)
                .ok_or(UmceError::InvalidDescriptor("match length lane exhausted"))?;
            match_length_pos += 1;
            let match_len = length as usize + 3;
            if match_len < 4 { return Err(UmceError::InvalidDescriptor("invalid match length lane")); }
            token_bytes.push(0x80 | length);
            let class = *offset_classes.get(class_pos)
                .ok_or(UmceError::InvalidDescriptor("offset class lane exhausted"))? as usize;
            class_pos += 1;
            if class > 15 { return Err(UmceError::InvalidDescriptor("invalid offset class")); }
            let end = residual_pos.checked_add(2).ok_or(UmceError::LengthOverflow)?;
            let bytes = offset_residuals.get(residual_pos..end)
                .ok_or(UmceError::InvalidDescriptor("offset residual lane exhausted"))?;
            let residual = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
            let base = 1usize << class;
            if residual >= base { return Err(UmceError::InvalidDescriptor("noncanonical offset residual")); }
            let offset = base.checked_add(residual).ok_or(UmceError::LengthOverflow)?;
            if offset == 0 || offset > original_len || offset > u16::MAX as usize {
                return Err(UmceError::InvalidDescriptor("invalid offset lane value"));
            }
            if pylz_version == 1 { token_bytes.extend_from_slice(&(offset as u16).to_le_bytes()); }
            else { put_uleb(offset, &mut token_bytes); }
            residual_pos = end;
            original_len = original_len.checked_add(match_len).ok_or(UmceError::LengthOverflow)?;
        } else {
            return Err(UmceError::InvalidDescriptor("unknown token kind"));
        }
        if original_len > MAX_PYLZ_OUTPUT { return Err(UmceError::InvalidDescriptor("PYLZ output limit exceeded")); }
    }
    if literal_length_pos != literal_lengths.len()
        || match_length_pos != match_lengths.len()
        || class_pos != offset_classes.len()
        || residual_pos != offset_residuals.len()
        || literal_pos != literals.len()
    {
        return Err(UmceError::InvalidDescriptor("unused bytes in token lanes"));
    }
    out[5..9].copy_from_slice(&(original_len as u32).to_le_bytes());
    out.extend_from_slice(&token_bytes);
    if out.len() != out_len { return Err(UmceError::InvalidDescriptor("token-aware reconstructed length mismatch")); }
    Ok(out)
}

fn decode_frame_four(input: &[u8], max_output_len: usize) -> Result<Vec<u8>, UmceError> {
    const FOUR_HEADER: usize = 31;
    if input.len() < FOUR_HEADER || &input[..4] != MAGIC || input[4] != FRAME_VERSION || input[5] != 1 {
        return Err(UmceError::InvalidDescriptor("bad four-lane token-aware frame"));
    }
    let out_len_u64 = u64::from_le_bytes(input[6..14].try_into().map_err(|_| UmceError::InvalidDescriptor("bad token-aware length"))?);
    let out_len = usize::try_from(out_len_u64).map_err(|_| UmceError::LengthOverflow)?;
    if out_len > max_output_len { return Err(UmceError::InvalidDescriptor("token-aware output limit exceeded")); }
    if out_len < PYLZ_HEADER { return Err(UmceError::InvalidDescriptor("token-aware PYLZ stream is too short")); }
    let version = input[14];
    if !matches!(version, 1 | 2) { return Err(UmceError::InvalidDescriptor("unsupported PYLZ version in token frame")); }
    let mut pos = 15usize;
    let mut sizes = Vec::with_capacity(4);
    for _ in 0..4 {
        let end = pos.checked_add(4).ok_or(UmceError::LengthOverflow)?;
        let b = input.get(pos..end).ok_or(UmceError::InvalidDescriptor("truncated token lane sizes"))?;
        sizes.push(u32::from_le_bytes(b.try_into().map_err(|_| UmceError::InvalidDescriptor("bad token lane size"))?) as usize);
        pos = end;
    }
    let mut ranges = Vec::with_capacity(4);
    for size in sizes {
        let end = pos.checked_add(size).ok_or(UmceError::LengthOverflow)?;
        if end > input.len() { return Err(UmceError::InvalidDescriptor("truncated token lane")); }
        ranges.push(pos..end);
        pos = end;
    }
    if pos != input.len() { return Err(UmceError::InvalidDescriptor("trailing bytes after token lanes")); }
    let mut budget = out_len.checked_mul(3).ok_or(UmceError::LengthOverflow)?;
    let mut decoded = Vec::with_capacity(4);
    for range in ranges {
        let lane = decode_section(&input[range], budget)?;
        budget = budget.checked_sub(lane.len()).ok_or(UmceError::InvalidDescriptor("token lane memory budget exceeded"))?;
        decoded.push(lane);
    }
    let [kinds, lengths, offsets, literals] = <[Vec<u8>; 4]>::try_from(decoded)
        .map_err(|_| UmceError::InvalidDescriptor("wrong token lane count"))?;
    if kinds.len() != lengths.len() { return Err(UmceError::InvalidDescriptor("token kind/length count mismatch")); }
    let mut out = Vec::with_capacity(out_len);
    out.extend_from_slice(PYLZ_MAGIC);
    out.push(version);
    out.extend_from_slice(&[0u8; 4]);
    let mut produced = 0usize;
    let mut offset_pos = 0usize;
    let mut literal_pos = 0usize;
    let mut token_bytes = Vec::new();
    for (&kind, &length) in kinds.iter().zip(lengths.iter()) {
        if kind == 0 {
            let count = length as usize + 1;
            let end = literal_pos.checked_add(count).ok_or(UmceError::LengthOverflow)?;
            let bytes = literals.get(literal_pos..end).ok_or(UmceError::InvalidDescriptor("literal lane exhausted"))?;
            token_bytes.push(length);
            token_bytes.extend_from_slice(bytes);
            literal_pos = end;
            produced = produced.checked_add(count).ok_or(UmceError::LengthOverflow)?;
        } else if kind == 1 {
            let end = offset_pos.checked_add(2).ok_or(UmceError::LengthOverflow)?;
            let bytes = offsets.get(offset_pos..end).ok_or(UmceError::InvalidDescriptor("offset lane exhausted"))?;
            let offset = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
            if offset == 0 || offset > produced { return Err(UmceError::InvalidDescriptor("invalid offset lane value")); }
            let match_len = length as usize + 3;
            if match_len < 4 { return Err(UmceError::InvalidDescriptor("invalid match length lane")); }
            token_bytes.push(0x80 | length);
            if version == 1 { token_bytes.extend_from_slice(bytes); } else { put_uleb(offset, &mut token_bytes); }
            offset_pos = end;
            produced = produced.checked_add(match_len).ok_or(UmceError::LengthOverflow)?;
        } else {
            return Err(UmceError::InvalidDescriptor("unknown token kind"));
        }
        if produced > MAX_PYLZ_OUTPUT { return Err(UmceError::InvalidDescriptor("PYLZ output limit exceeded")); }
    }
    if literal_pos != literals.len() || offset_pos != offsets.len() {
        return Err(UmceError::InvalidDescriptor("unused bytes in token lanes"));
    }
    out[5..9].copy_from_slice(&(produced as u32).to_le_bytes());
    out.extend_from_slice(&token_bytes);
    if out.len() != out_len { return Err(UmceError::InvalidDescriptor("token-aware reconstructed length mismatch")); }
    Ok(out)
}

pub(super) fn decode_frame(input: &[u8], max_output_len: usize) -> Result<Vec<u8>, UmceError> {
    if input.len() < 6 || &input[..4] != MAGIC || input[4] != FRAME_VERSION {
        return Err(UmceError::InvalidDescriptor("bad token-aware entropy frame"));
    }
    match input[5] {
        0 => decode_frame_six(input, max_output_len),
        1 => decode_frame_four(input, max_output_len),
        _ => Err(UmceError::InvalidDescriptor("unknown token-aware profile")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pylz() -> Vec<u8> {
        // PYLZ v2: four literals "abcd", then a 4-byte match at distance 4.
        b"PYLZ\x02\x08\x00\x00\x00\x03abcd\x81\x04".to_vec()
    }

    #[test]
    fn both_token_profiles_roundtrip_and_are_canonical() {
        let original = sample_pylz();
        let lanes = parse_pylz(&original).unwrap().unwrap();
        for frame in [
            encode_frame_four(&lanes, original.len()).unwrap(),
            encode_frame_six(&lanes, original.len()).unwrap(),
            encode_frame(&original).unwrap().unwrap(),
        ] {
            let decoded = decode_frame(&frame, original.len()).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn malformed_pylz_is_not_tokenized() {
        assert!(parse_pylz(b"PYLZ\x02\x08\x00\x00\x00\x81\x00").is_err());
    }

    #[test]
    fn legacy_v1_offsets_survive_token_lane_roundtrip() {
        let legacy = b"PYLZ\x01\x08\x00\x00\x00\x03abcd\x81\x04\x00";
        let frame = encode_frame(legacy).unwrap().unwrap();
        assert_eq!(decode_frame(&frame, legacy.len()).unwrap(), legacy);
    }

    #[test]
    fn rejects_nested_token_lane_frames() {
        assert!(decode_section(b"PYAE\x05", 1024).is_err());
    }

    #[test]
    fn frame_rejects_trailing_and_truncated_sections() {
        let original = sample_pylz();
        let mut frame = encode_frame(&original).unwrap().unwrap();
        frame.push(0);
        assert!(decode_frame(&frame, original.len()).is_err());
        frame.pop();
        frame.pop();
        assert!(decode_frame(&frame, original.len()).is_err());
    }
}
