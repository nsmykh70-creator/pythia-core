//! CRC-32C (Castagnoli), reflected polynomial 0x82F63B78.

const POLY: u32 = 0x82F6_3B78;

/// Incremental CRC-32C state for bounded-memory serialization and validation.
#[derive(Debug, Clone, Copy)]
pub struct Crc32c { state: u32 }

impl Crc32c {
    pub const fn new() -> Self { Self { state: !0u32 } }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.state ^= byte as u32;
            for _ in 0..8 {
                self.state = (self.state >> 1) ^ (POLY & (0u32.wrapping_sub(self.state & 1)));
            }
        }
    }

    pub const fn finalize(self) -> u32 { !self.state }
}

/// Computes CRC-32C using the incremental implementation.
pub fn checksum(data: &[u8]) -> u32 {
    let mut crc = Crc32c::new();
    crc.update(data);
    crc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn castagnoli_check_value() {
        assert_eq!(checksum(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let mut crc = Crc32c::new();
        crc.update(b"123");
        crc.update(b"456");
        crc.update(b"789");
        assert_eq!(crc.finalize(), checksum(b"123456789"));
    }

    #[test]
    fn empty_is_zero() { assert_eq!(checksum(b""), 0); }
}
