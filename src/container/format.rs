//! PYTHIA format 2.2 on-disk structures. All integers are little-endian.
use std::io::{self, Read, Write};
use super::crc32c;
use crate::{descriptor::ChainDescriptor, UmceError};

pub const ARCHIVE_MAGIC: &[u8; 8] = b"PYTHIA\r\n";
pub const BLOCK_MAGIC: &[u8; 4] = b"PYBK";
pub const MAJOR_VERSION: u16 = 2;
pub const MINOR_VERSION: u16 = 2;
pub const LEGACY_MAJOR_VERSION: u16 = 1;
pub const LEGACY_MAX_MINOR_VERSION: u16 = 1;
pub const SUPERBLOCK_LEN: usize = 36;
pub const BLOCK_HEADER_V0_LEN: usize = 28;
pub const BLOCK_HEADER_LEN: usize = 32;
pub const MAX_DESCRIPTOR_LEN: u32 = 4096;
/// Limits applied by every reader to a block's transform descriptor. The only
/// built-in parameterized model (algebraic recurrence) carries 4 bytes, and the
/// writer search emits at most two stages, so these bounds are generous but finite.
pub const MAX_DESCRIPTOR_STAGES: usize = 2;
pub const MAX_DESCRIPTOR_PARAMETER_BYTES: usize = 64;
pub const FOOTER_MAGIC: &[u8; 4] = b"PYEN";
pub const FOOTER_LEN: usize = 24;
pub const DEFAULT_BLOCK_SIZE: u32 = 1 << 20;
pub const MAX_BLOCK_SIZE: u32 = 16 << 20;
pub const UNKNOWN_SIZE: u64 = u64::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Superblock { pub major: u16, pub minor: u16, pub flags: u32, pub block_size: u32, pub declared_size: u64 }

impl Superblock {
    pub fn new(block_size: u32) -> io::Result<Self> {
        if block_size == 0 || block_size > MAX_BLOCK_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "block size outside supported range"));
        }
        Ok(Self { major: MAJOR_VERSION, minor: MINOR_VERSION, flags: 0, block_size, declared_size: UNKNOWN_SIZE })
    }

    pub fn encode(self) -> [u8; SUPERBLOCK_LEN] {
        let mut b = [0u8; SUPERBLOCK_LEN];
        b[..8].copy_from_slice(ARCHIVE_MAGIC);
        b[8..10].copy_from_slice(&self.major.to_le_bytes());
        b[10..12].copy_from_slice(&self.minor.to_le_bytes());
        b[12..16].copy_from_slice(&self.flags.to_le_bytes());
        b[16..20].copy_from_slice(&self.block_size.to_le_bytes());
        b[20..28].copy_from_slice(&self.declared_size.to_le_bytes());
        // bytes 28..32 are reserved and must be zero
        let crc = crc32c::checksum(&b[..32]);
        b[32..36].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8; SUPERBLOCK_LEN]) -> io::Result<Self> {
        if &b[..8] != ARCHIVE_MAGIC { return Err(invalid("not a PYTHIA archive")); }
        let major = u16::from_le_bytes([b[8], b[9]]);
        let minor = u16::from_le_bytes([b[10], b[11]]);
        let supported = match major {
            LEGACY_MAJOR_VERSION => minor <= LEGACY_MAX_MINOR_VERSION,
            MAJOR_VERSION => minor <= MINOR_VERSION,
            _ => false,
        };
        if !supported { return Err(invalid("unsupported PYTHIA format version")); }
        if b[28..32] != [0, 0, 0, 0] { return Err(invalid("nonzero reserved superblock bytes")); }
        let expected = u32::from_le_bytes(b[32..36].try_into().unwrap());
        if crc32c::checksum(&b[..32]) != expected { return Err(invalid("superblock CRC32C mismatch")); }
        let flags = u32::from_le_bytes(b[12..16].try_into().unwrap());
        if flags != 0 { return Err(invalid("unknown mandatory superblock flags")); }
        let block_size = u32::from_le_bytes(b[16..20].try_into().unwrap());
        if block_size == 0 || block_size > MAX_BLOCK_SIZE { return Err(invalid("invalid block size")); }
        Ok(Self { major, minor, flags, block_size, declared_size: u64::from_le_bytes(b[20..28].try_into().unwrap()) })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader { pub sequence: u64, pub raw_len: u32, pub frame_len: u32, pub descriptor_len: u32, pub raw_crc32c: u32 }
impl BlockHeader {
    pub fn encode(self) -> [u8; BLOCK_HEADER_LEN] {
        let mut b = [0u8; BLOCK_HEADER_LEN];
        b[..4].copy_from_slice(BLOCK_MAGIC);
        b[4..12].copy_from_slice(&self.sequence.to_le_bytes());
        b[12..16].copy_from_slice(&self.raw_len.to_le_bytes());
        b[16..20].copy_from_slice(&self.frame_len.to_le_bytes());
        b[20..24].copy_from_slice(&self.descriptor_len.to_le_bytes());
        b[24..28].copy_from_slice(&self.raw_crc32c.to_le_bytes());
        let crc = crc32c::checksum(&b[..28]);
        b[28..32].copy_from_slice(&crc.to_le_bytes());
        b
    }
    pub fn decode(b: &[u8; BLOCK_HEADER_LEN], max_block: u32) -> io::Result<Self> {
        if &b[..4] != BLOCK_MAGIC { return Err(invalid("bad block magic")); }
        let expected = u32::from_le_bytes(b[28..32].try_into().unwrap());
        if crc32c::checksum(&b[..28]) != expected { return Err(invalid("block header CRC32C mismatch")); }
        let h = Self {
            sequence: u64::from_le_bytes(b[4..12].try_into().unwrap()),
            raw_len: u32::from_le_bytes(b[12..16].try_into().unwrap()),
            frame_len: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            descriptor_len: u32::from_le_bytes(b[20..24].try_into().unwrap()),
            raw_crc32c: u32::from_le_bytes(b[24..28].try_into().unwrap()),
        };
        validate_block_header(h, max_block)?;
        Ok(h)
    }

    pub fn decode_v0(b: &[u8; BLOCK_HEADER_V0_LEN], max_block: u32) -> io::Result<Self> {
        if &b[..4] != BLOCK_MAGIC { return Err(invalid("bad block magic")); }
        let expected = u32::from_le_bytes(b[24..28].try_into().unwrap());
        if crc32c::checksum(&b[..24]) != expected { return Err(invalid("block header CRC32C mismatch")); }
        let h = Self {
            sequence: u64::from_le_bytes(b[4..12].try_into().unwrap()),
            raw_len: u32::from_le_bytes(b[12..16].try_into().unwrap()),
            frame_len: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            descriptor_len: 0,
            raw_crc32c: u32::from_le_bytes(b[20..24].try_into().unwrap()),
        };
        validate_block_header(h, max_block)?;
        Ok(h)
    }

}

/// The single place where readers parse a block's transform descriptor, so the
/// single-file reader, the sequential multi-file reader and the indexed reader
/// can never apply different limits.
pub(crate) fn decode_block_descriptor(bytes: &[u8]) -> Result<ChainDescriptor, UmceError> {
    ChainDescriptor::decode(bytes, MAX_DESCRIPTOR_STAGES, MAX_DESCRIPTOR_PARAMETER_BYTES)
}

/// Maximum byte length of the current reversible transform output for a raw
/// block. LZ77's literal-only worst case is a 9-byte PYLZ header, one tag per
/// 128 input bytes, plus the original bytes. All other built-in transforms are
/// length-preserving. This bound is used before entropy-frame allocation.
pub(crate) fn max_transformed_block_len(raw_len: u32) -> usize {
    let n = raw_len as usize;
    n.saturating_add((n.saturating_add(127)) / 128).saturating_add(9)
}

fn validate_block_header(h: BlockHeader, max_block: u32) -> io::Result<()> {
    if h.raw_len == 0 || h.raw_len > max_block { return Err(invalid("invalid uncompressed block length")); }
    if h.descriptor_len > MAX_DESCRIPTOR_LEN { return Err(invalid("transform descriptor exceeds safety limit")); }
    let max_frame = max_block.checked_mul(2).and_then(|v| v.checked_add(1 << 20)).unwrap_or(u32::MAX);
    if h.frame_len == 0 || h.frame_len > max_frame { return Err(invalid("encoded block length exceeds safety limit")); }
    Ok(())
}

pub(crate) fn read_superblock<R: Read>(r: &mut R) -> io::Result<Superblock> {
    let mut b = [0u8; SUPERBLOCK_LEN]; r.read_exact(&mut b)?; Superblock::decode(&b)
}
pub(crate) fn write_superblock<W: Write>(w: &mut W, s: Superblock) -> io::Result<()> { w.write_all(&s.encode()) }
fn invalid(s: &'static str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }

/// Footer makes a clean end-of-stream distinguishable from truncation.
pub fn encode_footer(block_count: u64, total_raw: u64) -> [u8; FOOTER_LEN] {
    let mut b = [0u8; FOOTER_LEN];
    b[..4].copy_from_slice(FOOTER_MAGIC);
    b[4..12].copy_from_slice(&block_count.to_le_bytes());
    b[12..20].copy_from_slice(&total_raw.to_le_bytes());
    let crc = crc32c::checksum(&b[..20]);
    b[20..24].copy_from_slice(&crc.to_le_bytes());
    b
}
pub fn decode_footer(b: &[u8; FOOTER_LEN]) -> io::Result<(u64, u64)> {
    if &b[..4] != FOOTER_MAGIC { return Err(invalid("bad archive footer magic")); }
    let expected = u32::from_le_bytes(b[20..24].try_into().unwrap());
    if crc32c::checksum(&b[..20]) != expected { return Err(invalid("archive footer CRC32C mismatch")); }
    Ok((u64::from_le_bytes(b[4..12].try_into().unwrap()), u64::from_le_bytes(b[12..20].try_into().unwrap())))
}

#[cfg(test)]
mod ps80_version_tests {
    use super::*;

    #[test]
    fn transformed_length_bound_covers_lz77_literal_worst_case() {
        for n in [1u32, 127, 128, 129, 4096, 1 << 20] {
            let max = max_transformed_block_len(n);
            let literal_only_pylz = 9usize + n as usize + (n as usize + 127) / 128;
            assert!(max >= literal_only_pylz);
        }
    }

    #[test]
    fn writes_minor_two_and_still_reads_minor_one_and_zero() {
        let current = Superblock::new(1 << 20).unwrap();
        assert_eq!(current.minor, 2);
        assert_eq!(Superblock::decode(&current.encode()).unwrap(), current);
        for minor in [0, 1] {
            let old = Superblock { major: MAJOR_VERSION, minor, flags: 0, block_size: 1 << 20, declared_size: UNKNOWN_SIZE };
            assert_eq!(Superblock::decode(&old.encode()).unwrap(), old);
        }
    }
}

#[cfg(test)]
mod ps97_descriptor_limit_tests {
    use super::*;
    use crate::{
        chain::{ModelChain, Stage},
        domain::Domain,
        transforms::{fit_algebraic_recurrence, MODEL_ALGEBRAIC_RECURRENCE, MODEL_DELTA8},
    };

    #[test]
    fn block_descriptor_decoder_accepts_parameterized_chains() {
        let params = fit_algebraic_recurrence(b"abcdefghijklmnop").to_vec();
        let chain = ModelChain::new(vec![Stage { model_id: MODEL_ALGEBRAIC_RECURRENCE, model_version: 1, parameters: params }]);
        let bytes = ChainDescriptor::from_chain(&chain, Domain::Bytes, Domain::Bytes).encode().unwrap();
        let decoded = decode_block_descriptor(&bytes).unwrap();
        assert_eq!(decoded.stages[0].parameters.len(), 4);
        // PS9.6 readers used a parameter limit of zero, which rejects exactly this descriptor.
        assert!(ChainDescriptor::decode(&bytes, 2, 0).is_err());
    }

    #[test]
    fn block_descriptor_decoder_still_bounds_parameter_bytes() {
        let chain = ModelChain::new(vec![Stage { model_id: MODEL_DELTA8, model_version: 1, parameters: vec![0u8; MAX_DESCRIPTOR_PARAMETER_BYTES + 1] }]);
        let bytes = ChainDescriptor::from_chain(&chain, Domain::Bytes, Domain::Bytes).encode().unwrap();
        assert!(decode_block_descriptor(&bytes).is_err());
    }
}
