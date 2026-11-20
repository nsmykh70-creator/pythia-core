//! Exact byte accounting for the archive structures known to PS6.2.
use crate::UmceError;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArchiveCost {
    pub header: u64,
    pub framing: u64,
    pub descriptors: u64,
    pub shared_model_bank: u64,
    pub shared_parameters: u64,
    pub segment_parameters: u64,
    pub auxiliary: u64,
    pub payload: u64,
    pub checksums: u64,
    pub index: u64,
}

impl ArchiveCost {
    pub fn total(self) -> Result<u64, UmceError> {
        [self.header, self.framing, self.descriptors, self.shared_model_bank,
         self.shared_parameters, self.segment_parameters, self.auxiliary,
         self.payload, self.checksums, self.index]
            .into_iter().try_fold(0u64, |sum, n| sum.checked_add(n).ok_or(UmceError::LengthOverflow))
    }

    /// Returns true only when candidate is strictly smaller by complete archive size.
    pub fn beats(self, incumbent: Self) -> Result<bool, UmceError> { Ok(self.total()? < incumbent.total()?) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankEconomics {
    pub standalone_descriptor_bytes: u64,
    pub bank_entry_bytes: u64,
    pub reference_bytes_per_segment: u64,
    pub segment_count: u64,
}

impl BankEconomics {
    pub fn standalone_total(self) -> Result<u64, UmceError> {
        self.standalone_descriptor_bytes.checked_mul(self.segment_count).ok_or(UmceError::LengthOverflow)
    }
    pub fn banked_total(self) -> Result<u64, UmceError> {
        self.bank_entry_bytes.checked_add(self.reference_bytes_per_segment.checked_mul(self.segment_count).ok_or(UmceError::LengthOverflow)?).ok_or(UmceError::LengthOverflow)
    }
    pub fn is_profitable(self) -> Result<bool, UmceError> { Ok(self.banked_total()? < self.standalone_total()?) }
    pub fn savings(self) -> Result<i128, UmceError> { Ok(self.standalone_total()? as i128 - self.banked_total()? as i128) }
}
