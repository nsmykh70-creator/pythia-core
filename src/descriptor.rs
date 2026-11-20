//! Canonical, deterministic serialization of a model-chain descriptor.
use crate::{chain::{ModelChain, Stage}, domain::Domain, UmceError};

const MAGIC: &[u8; 4] = b"PY6D";
pub const DESCRIPTOR_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainDescriptor {
    pub input_domain: Domain,
    pub output_domain: Domain,
    pub stages: Vec<Stage>,
}

impl ChainDescriptor {
    pub fn from_chain(chain: &ModelChain, input_domain: Domain, output_domain: Domain) -> Self {
        Self { input_domain, output_domain, stages: chain.stages().to_vec() }
    }

    /// Format: magic[4], version[1], domains[2], stage_count[ULEB128], then
    /// (model_id[ULEB128], model_version[ULEB128], parameter_len[ULEB128], params).
    /// No maps, padding, native-endian integers, or non-canonical varints.
    pub fn encode(&self) -> Result<Vec<u8>, UmceError> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(DESCRIPTOR_VERSION);
        out.push(self.input_domain.tag());
        out.push(self.output_domain.tag());
        put_varint(self.stages.len() as u64, &mut out);
        for stage in &self.stages {
            put_varint(stage.model_id as u64, &mut out);
            put_varint(stage.model_version as u64, &mut out);
            put_varint(stage.parameters.len() as u64, &mut out);
            out.extend_from_slice(&stage.parameters);
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8], max_stages: usize, max_parameter_bytes: usize) -> Result<Self, UmceError> {
        if bytes.len() < 7 || &bytes[..4] != MAGIC { return Err(UmceError::InvalidDescriptor("bad magic or truncated header")); }
        if bytes[4] != DESCRIPTOR_VERSION { return Err(UmceError::InvalidDescriptor("unsupported descriptor version")); }
        let input_domain = Domain::from_tag(bytes[5])?;
        let output_domain = Domain::from_tag(bytes[6])?;
        let mut pos = 7usize;
        let count = usize::try_from(get_varint(bytes, &mut pos)?).map_err(|_| UmceError::LengthOverflow)?;
        if count > max_stages { return Err(UmceError::InvalidDescriptor("stage limit exceeded")); }
        let mut stages = Vec::with_capacity(count);
        let mut total_params = 0usize;
        for _ in 0..count {
            let id = u32::try_from(get_varint(bytes, &mut pos)?).map_err(|_| UmceError::InvalidDescriptor("model id overflow"))?;
            let version = u16::try_from(get_varint(bytes, &mut pos)?).map_err(|_| UmceError::InvalidDescriptor("model version overflow"))?;
            let len = usize::try_from(get_varint(bytes, &mut pos)?).map_err(|_| UmceError::LengthOverflow)?;
            total_params = total_params.checked_add(len).ok_or(UmceError::LengthOverflow)?;
            if total_params > max_parameter_bytes { return Err(UmceError::InvalidDescriptor("parameter limit exceeded")); }
            let end = pos.checked_add(len).ok_or(UmceError::LengthOverflow)?;
            let params = bytes.get(pos..end).ok_or(UmceError::InvalidDescriptor("truncated parameters"))?.to_vec();
            pos = end;
            stages.push(Stage { model_id: id, model_version: version, parameters: params });
        }
        if pos != bytes.len() { return Err(UmceError::InvalidDescriptor("trailing bytes")); }
        Ok(Self { input_domain, output_domain, stages })
    }

    pub fn serialized_len(&self) -> Result<usize, UmceError> { Ok(self.encode()?.len()) }
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 { out.push((value as u8) | 0x80); value >>= 7; }
    out.push(value as u8);
}

fn get_varint(input: &[u8], pos: &mut usize) -> Result<u64, UmceError> {
    let start = *pos;
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *input.get(*pos).ok_or(UmceError::InvalidDescriptor("truncated varint"))?;
        *pos += 1;
        if shift >= 64 || (shift == 63 && byte > 1) { return Err(UmceError::InvalidDescriptor("varint overflow")); }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 { break; }
        shift += 7;
        if *pos - start > 10 { return Err(UmceError::InvalidDescriptor("varint too long")); }
    }
    let mut canonical = Vec::new(); put_varint(value, &mut canonical);
    if canonical.as_slice() != &input[start..*pos] { return Err(UmceError::InvalidDescriptor("non-canonical varint")); }
    Ok(value)
}
