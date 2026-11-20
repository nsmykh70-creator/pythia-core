//! Deterministic shared model bank. Entries are deduplicated by exact canonical bytes.
use std::collections::BTreeMap;
use crate::{descriptor::ChainDescriptor, UmceError};

pub type BankEntryId = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelBankEntry {
    pub id: BankEntryId,
    pub model_id: u32,
    pub model_version: u16,
    pub parameters: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelBank {
    entries: BTreeMap<BankEntryId, ModelBankEntry>,
    reverse: BTreeMap<(u32, u16, Vec<u8>), BankEntryId>,
}

impl ModelBank {
    pub fn new() -> Self { Self::default() }
    pub fn entries(&self) -> impl Iterator<Item = &ModelBankEntry> { self.entries.values() }
    pub fn get(&self, id: BankEntryId) -> Option<&ModelBankEntry> { self.entries.get(&id) }

    /// Insert or reuse a model instance. The key includes version and exact parameters.
    pub fn intern(&mut self, model_id: u32, model_version: u16, parameters: &[u8]) -> Result<BankEntryId, UmceError> {
        let key = (model_id, model_version, parameters.to_vec());
        if let Some(id) = self.reverse.get(&key) { return Ok(*id); }
        let next = self.entries.keys().next_back().copied().unwrap_or(0).checked_add(1).ok_or(UmceError::LengthOverflow)?;
        let entry = ModelBankEntry { id: next, model_id, model_version, parameters: parameters.to_vec() };
        self.entries.insert(next, entry);
        self.reverse.insert(key, next);
        Ok(next)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, UmceError> {
        let mut out = b"PY6B".to_vec();
        out.push(1); // bank format version
        put_varint(self.entries.len() as u64, &mut out);
        for entry in self.entries.values() {
            put_varint(entry.id as u64, &mut out);
            put_varint(entry.model_id as u64, &mut out);
            put_varint(entry.model_version as u64, &mut out);
            put_varint(entry.parameters.len() as u64, &mut out);
            out.extend_from_slice(&entry.parameters);
        }
        Ok(out)
    }

    /// Map every stage instance to a bank reference. This is a planning helper;
    /// segment-reference serialization is deliberately separate from full chain descriptors.
    pub fn references_for(&mut self, descriptor: &ChainDescriptor) -> Result<Vec<BankEntryId>, UmceError> {
        descriptor.stages.iter().map(|s| self.intern(s.model_id, s.model_version, &s.parameters)).collect()
    }

    pub fn serialized_len(&self) -> Result<usize, UmceError> { Ok(self.canonical_bytes()?.len()) }
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 { out.push((value as u8) | 0x80); value >>= 7; }
    out.push(value as u8);
}
