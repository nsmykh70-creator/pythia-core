//! PS6.6 byte-oriented rANS coder with a canonical sparse frequency table.
//!
//! The normalization and table representation are deterministic. Every symbol
//! present in the input receives at least one slot in the 12-bit distribution.

use crate::UmceError;

const SCALE_BITS: u32 = 12;
const SCALE: u32 = 1 << SCALE_BITS;
const RANS_L: u64 = 1 << 23;
const TABLE_PREFIX: usize = 2; // number of non-zero symbols, u16 LE
const ENTRY_SIZE: usize = 3;   // symbol u8 + frequency u16 LE
const STATE_SIZE: usize = 4;   // normalized state is serialized as u32 LE

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RansEncoded {
    pub table: Vec<u8>,
    pub state: u32,
    pub renorm: Vec<u8>,
}

fn normalize(input: &[u8]) -> Result<[u16; 256], UmceError> {
    let mut counts = [0u64; 256];
    for &b in input { counts[b as usize] += 1; }
    let total: u64 = counts.iter().sum();
    let present = counts.iter().filter(|&&n| n != 0).count();
    let mut freq = [0u16; 256];
    if present == 0 { return Ok(freq); }
    if present > SCALE as usize {
        return Err(UmceError::InvalidDescriptor("too many rANS symbols"));
    }

    // Reserve one slot for every observed symbol; distribute the remaining
    // slots proportionally using largest remainders (ties: lower byte value).
    let remaining = SCALE as u64 - present as u64;
    let mut remainders: Vec<(u64, usize)> = Vec::with_capacity(present);
    let mut assigned = present as u64;
    for (symbol, &count) in counts.iter().enumerate() {
        if count == 0 { continue; }
        let product = count.checked_mul(remaining).ok_or(UmceError::LengthOverflow)?;
        let extra = product / total;
        let remainder = product % total;
        freq[symbol] = (1 + extra) as u16;
        assigned += extra;
        remainders.push((remainder, symbol));
    }
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let left = (SCALE as u64 - assigned) as usize;
    for &(_, symbol) in remainders.iter().take(left) { freq[symbol] += 1; }
    Ok(freq)
}

fn serialize_table(freq: &[u16; 256]) -> Vec<u8> {
    let count = freq.iter().filter(|&&f| f != 0).count() as u16;
    let mut out = Vec::with_capacity(TABLE_PREFIX + count as usize * ENTRY_SIZE);
    out.extend_from_slice(&count.to_le_bytes());
    for (symbol, &f) in freq.iter().enumerate() {
        if f != 0 {
            out.push(symbol as u8);
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

fn parse_table(bytes: &[u8]) -> Result<([u16; 256], [u8; SCALE as usize], usize), UmceError> {
    if bytes.len() < TABLE_PREFIX {
        return Err(UmceError::InvalidDescriptor("truncated rANS table"));
    }
    let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    if count == 0 || count > 256 {
        return Err(UmceError::InvalidDescriptor("invalid rANS symbol count"));
    }
    let table_len = TABLE_PREFIX.checked_add(count.checked_mul(ENTRY_SIZE).ok_or(UmceError::LengthOverflow)?)
        .ok_or(UmceError::LengthOverflow)?;
    if bytes.len() < table_len {
        return Err(UmceError::InvalidDescriptor("truncated rANS frequency table"));
    }
    let mut freq = [0u16; 256];
    let mut sum = 0u32;
    let mut previous: Option<u8> = None;
    for entry in bytes[TABLE_PREFIX..table_len].chunks_exact(ENTRY_SIZE) {
        let symbol = entry[0];
        let f = u16::from_le_bytes([entry[1], entry[2]]);
        if f == 0 || previous.map_or(false, |p| symbol <= p) {
            return Err(UmceError::InvalidDescriptor("non-canonical rANS table"));
        }
        previous = Some(symbol);
        freq[symbol as usize] = f;
        sum = sum.checked_add(f as u32).ok_or(UmceError::LengthOverflow)?;
    }
    if sum != SCALE {
        return Err(UmceError::InvalidDescriptor("rANS frequencies do not sum to scale"));
    }
    let mut decode = [0u8; SCALE as usize];
    let mut base = 0usize;
    for symbol in 0..256 {
        let end = base + freq[symbol] as usize;
        decode[base..end].fill(symbol as u8);
        base = end;
    }
    Ok((freq, decode, table_len))
}

pub(super) fn encode_payload(input: &[u8]) -> Result<RansEncoded, UmceError> {
    if input.is_empty() {
        return Err(UmceError::InvalidDescriptor("rANS requires non-empty input"));
    }
    let freq = normalize(input)?;
    let table = serialize_table(&freq);
    let mut cumulative = [0u32; 257];
    for i in 0..256 { cumulative[i + 1] = cumulative[i] + freq[i] as u32; }
    if cumulative[256] != SCALE {
        return Err(UmceError::InvalidDescriptor("invalid normalized rANS distribution"));
    }

    let mut state = RANS_L;
    let mut emitted = Vec::new();
    for &byte in input.iter().rev() {
        let s = byte as usize;
        let f = freq[s] as u64;
        let c = cumulative[s] as u64;
        if f == 0 { return Err(UmceError::InvalidDescriptor("rANS symbol absent from table")); }
        let x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while state >= x_max {
            emitted.push((state & 0xff) as u8);
            state >>= 8;
        }
        state = ((state / f) << SCALE_BITS) + (state % f) + c;
    }
    let state32 = u32::try_from(state).map_err(|_| UmceError::LengthOverflow)?;
    emitted.reverse();
    Ok(RansEncoded { table, state: state32, renorm: emitted })
}

pub(super) fn decode_payload(payload: &[u8], output_len: usize) -> Result<Vec<u8>, UmceError> {
    if output_len == 0 {
        if payload.is_empty() { return Ok(Vec::new()); }
        return Err(UmceError::InvalidDescriptor("unexpected rANS payload for empty input"));
    }
    let (freq, decode_table, table_len) = parse_table(payload)?;
    if payload.len() < table_len + STATE_SIZE {
        return Err(UmceError::InvalidDescriptor("truncated rANS state"));
    }
    let mut cumulative = [0u32; 257];
    for i in 0..256 { cumulative[i + 1] = cumulative[i] + freq[i] as u32; }
    let state_bytes = &payload[table_len..table_len + STATE_SIZE];
    let mut state = u32::from_le_bytes(state_bytes.try_into().unwrap()) as u64;
    if state < RANS_L {
        return Err(UmceError::InvalidDescriptor("invalid initial rANS state"));
    }
    let renorm = &payload[table_len + STATE_SIZE..];
    let mut cursor = 0usize;
    let mut output = Vec::with_capacity(output_len);
    for _ in 0..output_len {
        let slot = (state as u32) & (SCALE - 1);
        let symbol = decode_table[slot as usize] as usize;
        let f = freq[symbol] as u64;
        let c = cumulative[symbol] as u64;
        state = f * (state >> SCALE_BITS) + (slot as u64 - c);
        while state < RANS_L {
            let b = *renorm.get(cursor).ok_or(UmceError::InvalidDescriptor("truncated rANS renormalization stream"))?;
            cursor += 1;
            state = (state << 8) | b as u64;
        }
        output.push(symbol as u8);
    }
    if cursor != renorm.len() || state != RANS_L {
        return Err(UmceError::InvalidDescriptor("rANS stream has trailing data or invalid terminal state"));
    }
    Ok(output)
}

pub(super) fn framed_payload_size(encoded: &RansEncoded) -> Result<usize, UmceError> {
    encoded.table.len().checked_add(STATE_SIZE).and_then(|n| n.checked_add(encoded.renorm.len()))
        .ok_or(UmceError::LengthOverflow)
}
