use std::io::{self, Read};
use crate::{
    chain::{ChainResult, ModelChain},
    domain::Domain,
    entropy::decode_best,
};
use super::{crc32c, format::{read_superblock, decode_footer, decode_block_descriptor, max_transformed_block_len, BlockHeader, Superblock, BLOCK_HEADER_LEN, BLOCK_HEADER_V0_LEN, FOOTER_LEN, FOOTER_MAGIC}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo { pub sequence: u64, pub header_len: u16, pub uncompressed_len: u32, pub encoded_len: u32, pub descriptor_len: u32, pub transform_stages: u16, pub checksum: u32 }

/// Sequential reader. It validates block framing before allocating and checks
/// the decompressed bytes with CRC32C before returning them to the caller.
pub struct ArchiveReader<R: Read> {
    inner: R,
    pub superblock: Superblock,
    expected_sequence: u64,
    total_output: u64,
    ended: bool,
    pending: Vec<u8>,
    pending_pos: usize,
}

impl<R: Read> ArchiveReader<R> {
    pub fn new(mut inner: R) -> io::Result<Self> {
        let superblock = read_superblock(&mut inner)?;
        Ok(Self { inner, superblock, expected_sequence: 0, total_output: 0, ended: false, pending: Vec::new(), pending_pos: 0 })
    }

    pub fn next_block(&mut self) -> io::Result<Option<(BlockInfo, Vec<u8>)>> {
        if self.pending_pos < self.pending.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "consume the pending block before requesting the next block"));
        }
        if self.ended { return Ok(None); }
        let mut prefix = [0u8; 4];
        self.inner.read_exact(&mut prefix).map_err(|_| invalid("missing or truncated archive footer/block"))?;
        if &prefix == FOOTER_MAGIC {
            let mut footer = [0u8; FOOTER_LEN];
            footer[..4].copy_from_slice(&prefix);
            self.inner.read_exact(&mut footer[4..]).map_err(|_| invalid("truncated archive footer"))?;
            let (blocks, total) = decode_footer(&footer)?;
            if blocks != self.expected_sequence || total != self.total_output { return Err(invalid("archive footer counters mismatch")); }
            if self.superblock.declared_size != u64::MAX && self.total_output != self.superblock.declared_size { return Err(invalid("archive uncompressed size does not match superblock")); }
            let mut trailing = [0u8; 1];
            if self.inner.read(&mut trailing)? != 0 { return Err(invalid("trailing bytes after archive footer")); }
            self.ended = true;
            return Ok(None);
        }
        let legacy_v1_0 = self.superblock.major == 1 && self.superblock.minor == 0;
        let header = if legacy_v1_0 {
            let mut raw_header = [0u8; BLOCK_HEADER_V0_LEN];
            raw_header[..4].copy_from_slice(&prefix);
            self.inner.read_exact(&mut raw_header[4..])?;
            BlockHeader::decode_v0(&raw_header, self.superblock.block_size)?
        } else {
            let mut raw_header = [0u8; BLOCK_HEADER_LEN];
            raw_header[..4].copy_from_slice(&prefix);
            self.inner.read_exact(&mut raw_header[4..])?;
            BlockHeader::decode(&raw_header, self.superblock.block_size)?
        };
        if header.sequence != self.expected_sequence { return Err(invalid("nonsequential block number")); }
        let mut descriptor = vec![0u8; header.descriptor_len as usize];
        self.inner.read_exact(&mut descriptor)?;
        let frame_len = header.frame_len as usize;
        let mut frame = vec![0u8; frame_len];
        self.inner.read_exact(&mut frame)?;
        let transformed = decode_best(&frame, max_transformed_block_len(header.raw_len)).map_err(|e| invalid_owned(format!("entropy frame rejected: {e}")))?;
        let (output, transform_stages) = if descriptor.is_empty() {
            (transformed, 0usize)
        } else {
            let desc = decode_block_descriptor(&descriptor)
                .map_err(|e| invalid_owned(format!("transform descriptor rejected: {e}")))?;
            if desc.input_domain != Domain::Bytes || desc.output_domain != Domain::Bytes || desc.stages.is_empty() {
                return Err(invalid("unsupported transform descriptor domains or empty chain"));
            }
            let stage_count = desc.stages.len();
            let chain = ModelChain::new(desc.stages);
            chain.validate_domains(Domain::Bytes, Domain::Bytes)
                .map_err(|e| invalid_owned(format!("transform chain rejected: {e}")))?;
            let chain_result = ChainResult { data: transformed, auxiliary: vec![Vec::new(); stage_count] };
            let restored = chain.inverse(&chain_result)
                .map_err(|e| invalid_owned(format!("inverse transform failed: {e}")))?;
            (restored, stage_count)
        };
        if output.len() != header.raw_len as usize { return Err(invalid("decoded block length mismatch")); }
        let actual_crc = crc32c::checksum(&output);
        if actual_crc != header.raw_crc32c { return Err(invalid("uncompressed block CRC32C mismatch")); }
        self.total_output = self.total_output.checked_add(output.len() as u64).ok_or_else(|| invalid("archive output length overflow"))?;
        self.expected_sequence = self.expected_sequence.checked_add(1).ok_or_else(|| invalid("too many blocks"))?;
        let header_len = if legacy_v1_0 { BLOCK_HEADER_V0_LEN as u16 } else { BLOCK_HEADER_LEN as u16 };
        let info = BlockInfo { sequence: header.sequence, header_len, uncompressed_len: header.raw_len, encoded_len: header.frame_len, descriptor_len: header.descriptor_len, transform_stages: transform_stages as u16, checksum: header.raw_crc32c };
        Ok(Some((info, output)))
    }

    pub fn total_uncompressed_bytes(&self) -> u64 { self.total_output }
    pub fn blocks_read(&self) -> u64 { self.expected_sequence }
}

fn invalid(s: &'static str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }
fn invalid_owned(s: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }

impl<R: Read> Read for ArchiveReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() { return Ok(0); }
        loop {
            if self.pending_pos < self.pending.len() {
                let n = (self.pending.len() - self.pending_pos).min(out.len());
                out[..n].copy_from_slice(&self.pending[self.pending_pos..self.pending_pos + n]);
                self.pending_pos += n;
                if self.pending_pos == self.pending.len() { self.pending.clear(); self.pending_pos = 0; }
                return Ok(n);
            }
            match self.next_block()? {
                Some((_info, bytes)) => self.pending = bytes,
                None => return Ok(0),
            }
        }
    }
}

#[cfg(test)]
mod ps97_parameterized_descriptor_tests {
    use super::*;
    use crate::container::ArchiveWriter;
    use std::io::{Cursor, Write};

    #[test]
    fn multiplicative_recurrence_roundtrips_through_the_container() {
        // x[i] = 3 * x[i-1] mod 256 is exactly representable by the algebraic
        // recurrence model, the only built-in model that carries parameters.
        let mut x = 1u8;
        let input: Vec<u8> = (0..8192).map(|_| { let v = x; x = x.wrapping_mul(3); v }).collect();
        let mut writer = ArchiveWriter::new(Cursor::new(Vec::new()), 4096).unwrap();
        writer.write_all(&input).unwrap();
        let archive = writer.finish().unwrap().into_inner();
        let mut reader = ArchiveReader::new(Cursor::new(archive)).unwrap();
        let mut restored = Vec::new();
        reader.read_to_end(&mut restored).unwrap();
        assert_eq!(restored, input);
    }
}

#[cfg(test)]
mod ps98_record_container_tests {
    use super::*;
    use crate::container::ArchiveWriter;
    use std::io::{Cursor, Write};

    #[test]
    fn fixed_size_records_roundtrip_through_the_container() {
        let mut x = 0x1234_5678u32;
        let mut next = move || { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x };
        let mut input = Vec::with_capacity(4096 * 16);
        for id in 0..4096u32 {
            input.extend_from_slice(&id.to_le_bytes());
            input.push(0xA5);
            input.push(((next() >> 24) as u8) & 3);
            input.extend_from_slice(&(((next() >> 16) as u16) % 500).to_le_bytes());
            input.extend_from_slice(&1_700_000_000u32.wrapping_add(id.wrapping_mul(3)).to_le_bytes());
            input.extend_from_slice(&next().to_le_bytes());
        }
        let mut writer = ArchiveWriter::new(Cursor::new(Vec::new()), 65_536).unwrap();
        writer.write_all(&input).unwrap();
        let archive = writer.finish().unwrap().into_inner();
        let mut reader = ArchiveReader::new(Cursor::new(archive)).unwrap();
        let mut restored = Vec::new();
        reader.read_to_end(&mut restored).unwrap();
        assert_eq!(restored, input);
    }
}
