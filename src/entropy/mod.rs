//! PS6.5 Adaptive Entropy Engine (AEE).
//!
//! Version 1 uses an online order-0 byte model and an integer arithmetic coder.
//! The decoder mirrors every model update, so no frequency table is serialized.
//! A raw mode is selected whenever coding would not reduce the payload.

use crate::{ArchiveCost, UmceError};

const MAGIC: &[u8; 4] = b"PYAE";
const VERSION: u8 = 1;
const MODE_RAW: u8 = 0;
const MODE_ADAPTIVE: u8 = 1;
const HEADER_LEN: usize = 14;
const MAX_TOTAL: u32 = 1 << 15;

#[derive(Clone)]
struct AdaptiveModel {
    freq: [u32; 256],
    fenwick: [u32; 257],
    total: u32,
}

impl AdaptiveModel {
    fn new() -> Self {
        let mut model = Self { freq: [1; 256], fenwick: [0; 257], total: 256 };
        model.rebuild_fenwick();
        model
    }

    fn prefix_sum(&self, count: usize) -> u32 {
        let mut index = count;
        let mut sum = 0u32;
        while index != 0 {
            sum += self.fenwick[index];
            index &= index - 1;
        }
        sum
    }

    fn cumulative(&self, symbol: usize) -> (u32, u32) {
        let low = self.prefix_sum(symbol);
        (low, low + self.freq[symbol])
    }

    fn symbol_for(&self, scaled: u32) -> Option<(usize, u32, u32)> {
        if scaled >= self.total { return None; }
        // Fenwick lower-bound search: find the largest prefix <= scaled.
        let mut index = 0usize;
        let mut accumulated = 0u32;
        let mut step = 256usize;
        while step != 0 {
            let next = index + step;
            if next <= 256 && accumulated + self.fenwick[next] <= scaled {
                index = next;
                accumulated += self.fenwick[next];
            }
            step >>= 1;
        }
        if index >= 256 { return None; }
        Some((index, accumulated, accumulated + self.freq[index]))
    }

    fn fenwick_add(&mut self, symbol: usize, delta: u32) {
        let mut index = symbol + 1;
        while index <= 256 {
            self.fenwick[index] += delta;
            index += index & (!index + 1);
        }
    }

    fn rebuild_fenwick(&mut self) {
        self.fenwick = [0; 257];
        for symbol in 0..256 {
            let frequency = self.freq[symbol];
            self.fenwick_add(symbol, frequency);
        }
    }

    fn update(&mut self, symbol: usize) {
        self.freq[symbol] += 1;
        self.total += 1;
        self.fenwick_add(symbol, 1);
        if self.total >= MAX_TOTAL {
            for f in &mut self.freq { *f = (*f + 1) / 2; }
            self.total = self.freq.iter().sum();
            self.rebuild_fenwick();
        }
    }
}

struct BitWriter { bytes: Vec<u8>, current: u8, used: u8 }
impl BitWriter {
    fn new() -> Self { Self { bytes: Vec::new(), current: 0, used: 0 } }
    fn bit(&mut self, bit: u8) {
        self.current = (self.current << 1) | (bit & 1);
        self.used += 1;
        if self.used == 8 { self.bytes.push(self.current); self.current = 0; self.used = 0; }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.used != 0 { self.current <<= 8 - self.used; self.bytes.push(self.current); }
        self.bytes
    }
}

struct BitReader<'a> { bytes: &'a [u8], bit_pos: usize }
impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, bit_pos: 0 } }
    // Arithmetic decoding permits zero-padding beyond the final coded bit.
    fn bit(&mut self) -> u8 {
        let byte = self.bit_pos / 8;
        let shift = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        self.bytes.get(byte).map(|b| (b >> shift) & 1).unwrap_or(0)
    }
}

const TOP: u64 = 0xffff_ffff;
const FIRST_QTR: u64 = 0x4000_0000;
const HALF: u64 = 0x8000_0000;
const THIRD_QTR: u64 = 0xc000_0000;

fn arithmetic_encode(input: &[u8]) -> Vec<u8> {
    let mut low = 0u64;
    let mut high = TOP;
    let mut pending = 0u64;
    let mut out = BitWriter::new();
    let mut model = AdaptiveModel::new();
    for &byte in input {
        let (cum_low, cum_high) = model.cumulative(byte as usize);
        let range = high - low + 1;
        high = low + (range * cum_high as u64 / model.total as u64) - 1;
        low += range * cum_low as u64 / model.total as u64;
        loop {
            if high < HALF {
                out.bit(0); for _ in 0..pending { out.bit(1); } pending = 0;
            } else if low >= HALF {
                out.bit(1); for _ in 0..pending { out.bit(0); } pending = 0;
                low -= HALF; high -= HALF;
            } else if low >= FIRST_QTR && high < THIRD_QTR {
                pending += 1; low -= FIRST_QTR; high -= FIRST_QTR;
            } else { break; }
            low = (low << 1) & TOP;
            high = ((high << 1) & TOP) | 1;
        }
        model.update(byte as usize);
    }
    pending += 1;
    if low < FIRST_QTR {
        out.bit(0); for _ in 0..pending { out.bit(1); }
    } else {
        out.bit(1); for _ in 0..pending { out.bit(0); }
    }
    out.finish()
}

fn arithmetic_decode(payload: &[u8], output_len: usize) -> Result<Vec<u8>, UmceError> {
    if payload.is_empty() { return Err(UmceError::InvalidDescriptor("missing arithmetic payload")); }
    let mut reader = BitReader::new(payload);
    let mut code = 0u64;
    for _ in 0..32 { code = (code << 1) | reader.bit() as u64; }
    let mut low = 0u64;
    let mut high = TOP;
    let mut model = AdaptiveModel::new();
    let mut output = Vec::with_capacity(output_len);
    for _ in 0..output_len {
        let range = high - low + 1;
        let scaled = (((code - low + 1) * model.total as u64 - 1) / range) as u32;
        let (symbol, cum_low, cum_high) = model.symbol_for(scaled)
            .ok_or(UmceError::InvalidDescriptor("arithmetic symbol outside model"))?;
        output.push(symbol as u8);
        high = low + (range * cum_high as u64 / model.total as u64) - 1;
        low += range * cum_low as u64 / model.total as u64;
        loop {
            if high < HALF {
            } else if low >= HALF {
                low -= HALF; high -= HALF; code -= HALF;
            } else if low >= FIRST_QTR && high < THIRD_QTR {
                low -= FIRST_QTR; high -= FIRST_QTR; code -= FIRST_QTR;
            } else { break; }
            low = (low << 1) & TOP;
            high = ((high << 1) & TOP) | 1;
            code = ((code << 1) & TOP) | reader.bit() as u64;
        }
        model.update(symbol);
    }
    Ok(output)
}

/// Encodes a complete self-describing AEE frame. The 14-byte header is always
/// counted. Raw mode is used unless arithmetic payload is strictly smaller.
pub fn encode(input: &[u8]) -> Result<Vec<u8>, UmceError> {
    let original_len = u64::try_from(input.len()).map_err(|_| UmceError::LengthOverflow)?;
    let compressed = arithmetic_encode(input);
    let (mode, payload) = if compressed.len() < input.len() { (MODE_ADAPTIVE, compressed.as_slice()) }
        else { (MODE_RAW, input) };
    let mut archive = Vec::with_capacity(HEADER_LEN + payload.len());
    archive.extend_from_slice(MAGIC);
    archive.push(VERSION);
    archive.push(mode);
    archive.extend_from_slice(&original_len.to_le_bytes());
    archive.extend_from_slice(payload);
    Ok(archive)
}

/// Decodes a complete AEE frame and rejects unknown modes, versions and lengths.
pub fn decode(archive: &[u8], max_output_len: usize) -> Result<Vec<u8>, UmceError> {
    if archive.len() < HEADER_LEN || &archive[..4] != MAGIC {
        return Err(UmceError::InvalidDescriptor("bad or truncated AEE header"));
    }
    if archive[4] != VERSION { return Err(UmceError::InvalidDescriptor("unsupported AEE version")); }
    let len_u64 = u64::from_le_bytes(archive[6..14].try_into().unwrap());
    let len = usize::try_from(len_u64).map_err(|_| UmceError::LengthOverflow)?;
    if len > max_output_len { return Err(UmceError::InvalidDescriptor("AEE output limit exceeded")); }
    let payload = &archive[HEADER_LEN..];
    match archive[5] {
        MODE_RAW => {
            if payload.len() != len { return Err(UmceError::LengthMismatch { expected: len, actual: payload.len() }); }
            Ok(payload.to_vec())
        }
        MODE_ADAPTIVE => arithmetic_decode(payload, len),
        _ => Err(UmceError::InvalidDescriptor("unknown AEE mode")),
    }
}

/// Returns the exact framed size, including the AEE header.
pub fn encoded_size(input: &[u8]) -> Result<usize, UmceError> { Ok(encode(input)?.len()) }

/// Adds the exact AEE frame overhead and selected payload to an existing archive
/// cost. Existing payload bytes are preserved and AEE framing is charged once.
pub fn add_to_archive_cost(input: &[u8], mut cost: ArchiveCost) -> Result<ArchiveCost, UmceError> {
    let compressed = arithmetic_encode(input);
    let payload_len = if compressed.len() < input.len() { compressed.len() } else { input.len() };
    cost.framing = cost.framing.checked_add(HEADER_LEN as u64).ok_or(UmceError::LengthOverflow)?;
    cost.payload = cost.payload.checked_add(payload_len as u64).ok_or(UmceError::LengthOverflow)?;
    cost.total()?;
    Ok(cost)
}

#[cfg(test)]
mod ps82_token_tests {
    use super::*;

    #[test]
    fn token_aware_candidate_is_exact_and_roundtrips() {
        // PYLZ v2: four literals followed by a four-byte back-reference.
        let pylz = b"PYLZ\x02\x08\x00\x00\x00\x03abcd\x81\x04";
        let token_frame = token::encode_frame(pylz).unwrap().unwrap();
        assert_eq!(decode_best(&token_frame, pylz.len()).unwrap(), pylz);
        let forced_token = EntropySelection { codec: EntropyCodec::TokenAware, archive: token_frame, candidates: Vec::new() };
        let (token_cost, token_codec) = add_selection_to_archive_cost(pylz.len(), &forced_token, ArchiveCost::default()).unwrap();
        assert_eq!(token_codec, EntropyCodec::TokenAware);
        assert_eq!(token_cost.total().unwrap() as usize, forced_token.archive.len());

        let selected = encode_best(pylz).unwrap();
        assert!(selected.candidates.iter().any(|c| c.codec == EntropyCodec::TokenAware));
        assert_eq!(decode_best(&selected.archive, pylz.len()).unwrap(), pylz);
        let (cost, codec) = add_selection_to_archive_cost(pylz.len(), &selected, ArchiveCost::default()).unwrap();
        assert_eq!(codec, selected.codec);
        assert_eq!(cost.total().unwrap() as usize, selected.archive.len());
    }

    #[test]
    fn malformed_pylz_prefix_remains_ordinary_data() {
        let data = b"PYLZ-not-a-valid-transform-stream";
        let selected = encode_best(data).unwrap();
        assert!(!selected.candidates.iter().any(|c| c.codec == EntropyCodec::TokenAware));
        assert_eq!(decode_best(&selected.archive, data.len()).unwrap(), data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty_short_and_adaptive_data() {
        for data in [Vec::new(), vec![0], vec![0; 4096], b"abracadabra abracadabra".to_vec(), (0..=255).cycle().take(8192).collect()] {
            let encoded = encode(&data).unwrap();
            assert_eq!(decode(&encoded, 1_000_000).unwrap(), data);
        }
    }

    #[test]
    fn adaptive_frame_never_expands_and_roundtrips_random_like_data() {
        // The exact mode is an encoder decision: a deterministic vector can
        // contain exploitable order-0 structure even when it looks random.
        // Assert the contract, not a particular codec mode.
        let data: Vec<u8> = (0..65536).map(|i| ((i * 73 + i / 11) & 255) as u8).collect();
        let encoded = encode(&data).unwrap();
        assert!(encoded.len() <= HEADER_LEN + data.len());
        assert_eq!(decode(&encoded, data.len()).unwrap(), data);
        assert!(matches!(encoded[5], MODE_RAW | MODE_ADAPTIVE));
    }

    #[test]
    fn malformed_entropy_metadata_is_rejected_without_panicking() {
        for codec in [EntropyCodec::Rans, EntropyCodec::ContextRans, EntropyCodec::ContextRansBackoff] {
            let selection = EntropySelection {
                codec,
                archive: vec![0; HEADER_LEN],
                candidates: Vec::new(),
            };
            assert!(add_selection_to_archive_cost(0, &selection, ArchiveCost::default()).is_err());
        }
    }

    #[test]
    fn archive_cost_includes_aee_framing_and_payload() {
        let cost = add_to_archive_cost(b"abc", ArchiveCost::default()).unwrap();
        assert_eq!(cost.framing, HEADER_LEN as u64);
        assert_eq!(cost.payload, 3);
        assert_eq!(cost.total().unwrap(), HEADER_LEN as u64 + 3);
    }

    #[test]
    fn rejects_bad_header_and_output_limit() {
        assert!(decode(b"bad", 100).is_err());
        let encoded = encode(b"data").unwrap();
        assert!(decode(&encoded, 3).is_err());
    }
}

/// PS6.6 rANS implementation and exact full-frame selector.
pub mod rans;
/// PS6.7 order-1 context model with rANS backend.
pub mod cme;
/// PS6.8 bounded order-1..4 context rANS with longest-context backoff.
pub mod cme68;
/// PS8.2 token-aware entropy framing for PYLZ streams.
pub mod token;

const RANS_VERSION: u8 = 2;
const MODE_RANS: u8 = 2;

/// Entropy backend selected after comparing complete framed byte counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyCodec {
    Raw,
    AdaptiveArithmetic,
    Rans,
    ContextRans,
    ContextRansBackoff,
    ContextRansPruned,
    TokenAware,
}

/// Exact candidate measurements. `archive_bytes` includes the codec frame,
/// adaptive/rANS metadata, the rANS frequency table and its terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntropyCandidate {
    pub codec: EntropyCodec,
    pub archive_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntropySelection {
    pub codec: EntropyCodec,
    pub archive: Vec<u8>,
    pub candidates: Vec<EntropyCandidate>,
}

fn raw_frame(input: &[u8]) -> Result<Vec<u8>, UmceError> {
    let len = u64::try_from(input.len()).map_err(|_| UmceError::LengthOverflow)?;
    let mut out = Vec::with_capacity(HEADER_LEN.checked_add(input.len()).ok_or(UmceError::LengthOverflow)?);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(MODE_RAW);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(input);
    Ok(out)
}

fn rans_frame(input: &[u8]) -> Result<Vec<u8>, UmceError> {
    let len = u64::try_from(input.len()).map_err(|_| UmceError::LengthOverflow)?;
    let encoded = rans::encode_payload(input)?;
    let payload_len = rans::framed_payload_size(&encoded)?;
    let capacity = HEADER_LEN.checked_add(payload_len).ok_or(UmceError::LengthOverflow)?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(MAGIC);
    out.push(RANS_VERSION);
    out.push(MODE_RANS);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&encoded.table);
    out.extend_from_slice(&encoded.state.to_le_bytes());
    out.extend_from_slice(&encoded.renorm);
    Ok(out)
}

/// Compares raw, adaptive arithmetic and rANS by exact complete frame size.
/// Ties are resolved deterministically: adaptive, then rANS, then raw.
fn encode_best_plain(input: &[u8]) -> Result<EntropySelection, UmceError> {
    let adaptive = encode(input)?;
    let raw = raw_frame(input)?;
    let adaptive_is_compressed = adaptive.get(5) == Some(&MODE_ADAPTIVE);
    let rans = if input.is_empty() { None } else { Some(rans_frame(input)?) };
    let context = if input.is_empty() { None } else { Some(cme::frame(input)?) };
    let mut backoff_frames = Vec::new();
    let mut pruned_frames = Vec::new();
    if !input.is_empty() {
        for order in 1..=4 {
            if let Ok(frame) = cme68::frame(input, order) { backoff_frames.push((order, frame)); }
            if let Ok(frame) = cme68::frame_pruned(input, order) { pruned_frames.push((order, frame)); }
        }
    }

    let mut candidates = Vec::with_capacity(12);
    if adaptive_is_compressed {
        candidates.push(EntropyCandidate { codec: EntropyCodec::AdaptiveArithmetic, archive_bytes: adaptive.len() });
    }
    if let Some(ref frame) = rans {
        candidates.push(EntropyCandidate { codec: EntropyCodec::Rans, archive_bytes: frame.len() });
    }
    if let Some(ref frame) = context { candidates.push(EntropyCandidate { codec: EntropyCodec::ContextRans, archive_bytes: frame.len() }); }
    for (_, frame) in &backoff_frames { candidates.push(EntropyCandidate { codec: EntropyCodec::ContextRansBackoff, archive_bytes: frame.len() }); }
    for (_, frame) in &pruned_frames { candidates.push(EntropyCandidate { codec: EntropyCodec::ContextRansPruned, archive_bytes: frame.len() }); }
    candidates.push(EntropyCandidate { codec: EntropyCodec::Raw, archive_bytes: raw.len() });

    // Adaptive is the initial winner when it actually compressed; otherwise
    // RAW is the baseline. Strict comparisons make ties deterministic.
    let (mut best_codec, mut best_size, mut best_archive) = if adaptive_is_compressed {
        (EntropyCodec::AdaptiveArithmetic, adaptive.len(), adaptive)
    } else {
        (EntropyCodec::Raw, raw.len(), raw)
    };
    if let Some(frame) = rans {
        if frame.len() < best_size { best_codec = EntropyCodec::Rans; best_size = frame.len(); best_archive = frame; }
    }
    if let Some(frame) = context {
        if frame.len() < best_size { best_codec = EntropyCodec::ContextRans; best_size = frame.len(); best_archive = frame; }
    }
    for (_, frame) in backoff_frames { if frame.len() < best_size { best_codec = EntropyCodec::ContextRansBackoff; best_size = frame.len(); best_archive = frame; } }
    for (_, frame) in pruned_frames { if frame.len() < best_size { best_codec = EntropyCodec::ContextRansPruned; best_size = frame.len(); best_archive = frame; } }
    Ok(EntropySelection { codec: best_codec, archive: best_archive, candidates })
}

/// Full-frame selector. PYLZ streams additionally get a token-aware candidate;
/// it is retained only when its complete frame is strictly smaller.
pub fn encode_best(input: &[u8]) -> Result<EntropySelection, UmceError> {
    let mut selected = encode_best_plain(input)?;
    if input.starts_with(b"PYLZ") {
        if let Some(frame) = token::encode_frame(input)? {
            selected.candidates.push(EntropyCandidate { codec: EntropyCodec::TokenAware, archive_bytes: frame.len() });
            if frame.len() < selected.archive.len() {
                selected.codec = EntropyCodec::TokenAware;
                selected.archive = frame;
            }
        }
    }
    Ok(selected)
}

/// Decodes PS6.5 adaptive/raw, PS6.6 global rANS, PS6.7 context-rANS and PS8.2 token frames.
pub fn decode_best(archive: &[u8], max_output_len: usize) -> Result<Vec<u8>, UmceError> {
    if archive.len() < HEADER_LEN || &archive[..4] != MAGIC {
        return Err(UmceError::InvalidDescriptor("bad or truncated entropy frame"));
    }
    if archive[4] == 5 { return token::decode_frame(archive, max_output_len); }
    if archive[4] == 4 || archive[4] == 6 { return cme68::decode_frame(archive, max_output_len); }
    if archive[4] == 3 { return cme::decode_frame(archive, max_output_len); }
    if archive[4] != RANS_VERSION { return decode(archive, max_output_len); }
    if archive[5] != MODE_RANS {
        return Err(UmceError::InvalidDescriptor("unknown PS6.6 entropy mode"));
    }
    let len_u64 = u64::from_le_bytes(archive[6..14].try_into().unwrap());
    let len = usize::try_from(len_u64).map_err(|_| UmceError::LengthOverflow)?;
    if len > max_output_len {
        return Err(UmceError::InvalidDescriptor("entropy output limit exceeded"));
    }
    rans::decode_payload(&archive[HEADER_LEN..], len)
}

/// Reads one byte from a self-describing frame without unchecked indexing.
fn frame_read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8, UmceError> {
    let value = *bytes.get(*pos).ok_or(UmceError::InvalidDescriptor("truncated entropy frame metadata"))?;
    *pos = (*pos).checked_add(1).ok_or(UmceError::LengthOverflow)?;
    Ok(value)
}

/// Reads a little-endian u16 from a self-describing frame with bounds checks.
fn frame_read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16, UmceError> {
    let end = (*pos).checked_add(2).ok_or(UmceError::LengthOverflow)?;
    let slice = bytes.get(*pos..end).ok_or(UmceError::InvalidDescriptor("truncated entropy frame metadata"))?;
    *pos = end;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

/// Advances over a variable-length metadata region after checking its extent.
fn frame_skip(bytes: &[u8], pos: &mut usize, count: usize) -> Result<(), UmceError> {
    let end = (*pos).checked_add(count).ok_or(UmceError::LengthOverflow)?;
    if end > bytes.len() {
        return Err(UmceError::InvalidDescriptor("truncated entropy frame metadata"));
    }
    *pos = end;
    Ok(())
}

/// Adds the selected frame's exact bytes to ArchiveCost. Metadata parsing is
/// checked because this public accounting API must not panic on malformed data.
pub fn add_selection_to_archive_cost(
    input_len: usize,
    selection: &EntropySelection,
    mut cost: ArchiveCost,
) -> Result<(ArchiveCost, EntropyCodec), UmceError> {
    let frame_overhead = HEADER_LEN as u64;
    let (framing, payload) = match selection.codec {
        EntropyCodec::Raw => (frame_overhead, u64::try_from(input_len).map_err(|_| UmceError::LengthOverflow)?),
        EntropyCodec::TokenAware => (u64::try_from(selection.archive.len()).map_err(|_| UmceError::LengthOverflow)?, 0),
        EntropyCodec::AdaptiveArithmetic => {
            let payload = selection.archive.len().checked_sub(HEADER_LEN)
                .ok_or(UmceError::InvalidDescriptor("truncated adaptive frame"))?;
            (frame_overhead, u64::try_from(payload).map_err(|_| UmceError::LengthOverflow)?)
        }
        EntropyCodec::ContextRansBackoff | EntropyCodec::ContextRansPruned => {
            let count_bytes = selection.archive.get(HEADER_LEN + 1..HEADER_LEN + 5)
                .ok_or(UmceError::InvalidDescriptor("bad PS6.8 model count"))?;
            let count = u32::from_le_bytes(count_bytes.try_into()
                .map_err(|_| UmceError::InvalidDescriptor("bad PS6.8 model count"))?) as usize;
            let mut pos = HEADER_LEN.checked_add(5).ok_or(UmceError::LengthOverflow)?;
            for _ in 0..count {
                let key_len = frame_read_u8(&selection.archive, &mut pos)? as usize;
                frame_skip(&selection.archive, &mut pos, key_len)?;
                let n = frame_read_u16(&selection.archive, &mut pos)? as usize;
                frame_skip(&selection.archive, &mut pos, n.checked_mul(3).ok_or(UmceError::LengthOverflow)?)?;
            }
            let framing = u64::try_from(pos).map_err(|_| UmceError::LengthOverflow)?;
            let payload = selection.archive.len().checked_sub(pos)
                .ok_or(UmceError::InvalidDescriptor("context frame metadata exceeds frame length"))?;
            (framing, u64::try_from(payload).map_err(|_| UmceError::LengthOverflow)?)
        }
        EntropyCodec::ContextRans => {
            let mut pos = HEADER_LEN;
            let global_n = frame_read_u16(&selection.archive, &mut pos)? as usize;
            frame_skip(&selection.archive, &mut pos, global_n.checked_mul(3).ok_or(UmceError::LengthOverflow)?)?;
            let contexts = frame_read_u16(&selection.archive, &mut pos)? as usize;
            for _ in 0..contexts {
                frame_skip(&selection.archive, &mut pos, 1)?;
                let n = frame_read_u16(&selection.archive, &mut pos)? as usize;
                frame_skip(&selection.archive, &mut pos, n.checked_mul(3).ok_or(UmceError::LengthOverflow)?)?;
            }
            let framing = u64::try_from(pos).map_err(|_| UmceError::LengthOverflow)?;
            let payload = selection.archive.len().checked_sub(pos)
                .ok_or(UmceError::InvalidDescriptor("context frame metadata exceeds frame length"))?;
            (framing, u64::try_from(payload).map_err(|_| UmceError::LengthOverflow)?)
        }
        EntropyCodec::Rans => {
            let mut pos = HEADER_LEN;
            let table_len = frame_read_u16(&selection.archive, &mut pos)? as usize;
            frame_skip(&selection.archive, &mut pos, table_len.checked_mul(3).ok_or(UmceError::LengthOverflow)?)?;
            let framing = u64::try_from(pos).map_err(|_| UmceError::LengthOverflow)?;
            let payload = selection.archive.len().checked_sub(pos)
                .ok_or(UmceError::InvalidDescriptor("rANS table exceeds frame length"))?;
            (framing, u64::try_from(payload).map_err(|_| UmceError::LengthOverflow)?)
        }
    };
    cost.framing = cost.framing.checked_add(framing).ok_or(UmceError::LengthOverflow)?;
    cost.payload = cost.payload.checked_add(payload).ok_or(UmceError::LengthOverflow)?;
    cost.total()?;
    Ok((cost, selection.codec))
}

/// Convenience wrapper. If the caller already has an `EntropySelection`, use
/// `add_selection_to_archive_cost` to avoid encoding every candidate again.
pub fn add_best_to_archive_cost(input: &[u8], cost: ArchiveCost) -> Result<(ArchiveCost, EntropyCodec), UmceError> {
    let selection = encode_best(input)?;
    add_selection_to_archive_cost(input.len(), &selection, cost)
}

#[cfg(test)]
mod ps66_tests {
    use super::*;

    #[test]
    fn adaptive_model_fenwick_lookup_matches_all_symbol_intervals() {
        let mut model = AdaptiveModel::new();
        for symbol in 0..256 {
            let (low, high) = model.cumulative(symbol);
            assert_eq!((low, high), (symbol as u32, symbol as u32 + 1));
            assert_eq!(model.symbol_for(low), Some((symbol, low, high)));
        }
        for symbol in (0..256).rev() { model.update(symbol); }
        let mut cumulative = 0u32;
        for symbol in 0..256 {
            let (low, high) = model.cumulative(symbol);
            assert_eq!(low, cumulative);
            assert_eq!(model.symbol_for(low), Some((symbol, low, high)));
            assert_eq!(model.symbol_for(high - 1), Some((symbol, low, high)));
            cumulative = high;
        }
        assert_eq!(cumulative, model.total);
    }

    #[test]
    fn rans_roundtrip_and_canonical_frame() {
        for data in [
            b"a".to_vec(),
            b"hello world".repeat(100),
            (0..=255).cycle().take(4096).collect(),
            vec![0; 10_000],
        ] {
            let frame = rans_frame(&data).unwrap();
            assert_eq!(decode_best(&frame, 20_000).unwrap(), data);
            assert_eq!(frame[4], RANS_VERSION);
            assert_eq!(frame[5], MODE_RANS);
        }
    }

    #[test]
    fn selector_chooses_smallest_complete_frame() {
        let data = b"PYTHIA adaptive entropy selector test ".repeat(200);
        let selection = encode_best(&data).unwrap();
        let minimum = selection.candidates.iter().map(|c| c.archive_bytes).min().unwrap();
        assert_eq!(selection.archive.len(), minimum);
        assert_eq!(decode_best(&selection.archive, 100_000).unwrap(), data);
    }

    #[test]
    fn random_like_data_is_not_expanded_by_selector() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 73 + i / 11) & 255) as u8).collect();
        let selection = encode_best(&data).unwrap();
        assert!(selection.archive.len() <= HEADER_LEN + data.len());
        assert_eq!(decode_best(&selection.archive, 10_000).unwrap(), data);
    }
}

#[cfg(test)]
mod ps67_tests {
    use super::*;
    #[test]
    fn context_frame_roundtrips_structured_inputs() {
        for data in [b"abababababababababab".repeat(100), b"the quick brown fox ".repeat(300), (0..=255).cycle().take(8192).collect()] {
            let frame = cme::frame(&data).unwrap();
            assert_eq!(decode_best(&frame, 100_000).unwrap(), data);
        }
    }
    #[test]
    fn selector_accounts_for_context_table_overhead() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 73 + i / 11) & 255) as u8).collect();
        let selected = encode_best(&data).unwrap();
        assert_eq!(selected.archive.len(), selected.candidates.iter().map(|c| c.archive_bytes).min().unwrap());
        assert!(selected.archive.len() <= HEADER_LEN + data.len());
    }
    #[test]
    fn ps68_backoff_orders_roundtrip_and_are_fully_framed() {
        let inputs = [b"ABRACADABRA ABRACADABRA ".repeat(500), b"abcabcabcabcxyzxyzxyz".repeat(300), (0..=255).cycle().take(4096).collect()];
        for data in inputs { for order in 1..=4 { let frame = cme68::frame(&data, order).unwrap(); assert_eq!(decode_best(&frame, 100_000).unwrap(), data); assert_eq!(frame[4], 4); assert_eq!(frame[5], order as u8); } }
    }
    #[test]
    fn selector_accounts_for_ps68_full_frame_size() {
        let data = b"abcabcabcabcxyzxyzxyz".repeat(300); let selected = encode_best(&data).unwrap();
        assert_eq!(selected.archive.len(), selected.candidates.iter().map(|c| c.archive_bytes).min().unwrap());
        assert_eq!(decode_best(&selected.archive, 100_000).unwrap(), data);
        if selected.codec == EntropyCodec::ContextRansBackoff { let (cost, codec) = add_best_to_archive_cost(&data, ArchiveCost::default()).unwrap(); assert_eq!(codec, selected.codec); assert_eq!(cost.total().unwrap() as usize, selected.archive.len()); }
    }
    #[test]
    fn context_archive_cost_splits_metadata_and_coded_stream() {
        let data = b"ABABABABABABABABABAB".repeat(100);
        let selected = encode_best(&data).unwrap();
        if selected.codec == EntropyCodec::ContextRans {
            let (cost, codec) = add_best_to_archive_cost(&data, ArchiveCost::default()).unwrap();
            assert_eq!(codec, EntropyCodec::ContextRans);
            assert_eq!(cost.total().unwrap() as usize, selected.archive.len());
        }
    }
}


#[cfg(test)]
mod ps90_cost_accounting_tests {
    use super::*;

    #[test]
    fn pruned_context_frame_cost_counts_all_metadata_exactly() {
        let input = b"cost accounting for pruned contexts ".repeat(256);
        let frame = cme68::frame_pruned(&input, 3).unwrap();
        let selection = EntropySelection {
            codec: EntropyCodec::ContextRansPruned,
            archive: frame.clone(),
            candidates: Vec::new(),
        };
        let (cost, codec) = add_selection_to_archive_cost(input.len(), &selection, ArchiveCost::default()).unwrap();
        assert_eq!(codec, EntropyCodec::ContextRansPruned);
        assert_eq!(cost.total().unwrap() as usize, frame.len());
    }
}


#[cfg(test)]
mod ps90_pruned_context_tests {
    use super::{cme68, decode_best, encode_best, EntropyCodec};

    #[test]
    fn pruned_context_frames_roundtrip_patterns_and_orders() {
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"abcabcabcabcxyzxyz".repeat(512),
            b"the quick brown fox jumps over the lazy dog ".repeat(256),
            (0..65536usize).map(|i| ((i * 73 + i / 11) & 255) as u8).collect(),
            (0..65536usize).map(|_| 0u8).collect(),
        ];
        for order in 1..=4 {
            assert!(cme68::frame_pruned(&[], order).is_err());
        }
        for data in cases.iter().filter(|x| !x.is_empty()) {
            for order in 1..=4 {
                let frame = cme68::frame_pruned(data, order).unwrap();
                assert_eq!(frame[4], 6);
                assert_eq!(frame[5], order as u8);
                assert_eq!(decode_best(&frame, data.len()).unwrap(), *data);
            }
        }
    }

    #[test]
    fn pruned_context_candidate_is_registered_and_roundtrips() {
        let data = b"abababababababababababababababab".repeat(2048);
        let selected = encode_best(&data).unwrap();
        assert!(selected.candidates.iter().any(|c| c.codec == EntropyCodec::ContextRansPruned));
        assert_eq!(decode_best(&selected.archive, data.len()).unwrap(), data);
    }

    #[test]
    fn old_ps68_frame_version_remains_decodable() {
        let data = b"legacy PS6.8 frame compatibility ".repeat(128);
        let frame = cme68::frame(&data, 3).unwrap();
        assert_eq!(frame[4], 4);
        assert_eq!(decode_best(&frame, data.len()).unwrap(), data);
    }
}

