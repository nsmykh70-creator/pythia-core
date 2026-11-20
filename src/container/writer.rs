use std::io::{self, Write};
use super::pipeline::select_block_representation;
use super::{crc32c, format::{write_superblock, encode_footer, BlockHeader, Superblock, MAX_DESCRIPTOR_LEN}};

/// Buffered block writer. Peak input-buffer memory is bounded by block_size;
/// one encoded block is additionally held while it is written.
pub struct ArchiveWriter<W: Write> {
    inner: W,
    block_size: usize,
    buffer: Vec<u8>,
    sequence: u64,
    total_input: u64,
    finished: bool,
}

impl<W: Write> ArchiveWriter<W> {
    pub fn new(mut inner: W, block_size: u32) -> io::Result<Self> {
        let superblock = Superblock::new(block_size)?;
        write_superblock(&mut inner, superblock)?;
        Ok(Self { inner, block_size: block_size as usize, buffer: Vec::with_capacity(block_size as usize), sequence: 0, total_input: 0, finished: false })
    }

    pub fn with_default_block_size(inner: W) -> io::Result<Self> {
        Self::new(inner, super::format::DEFAULT_BLOCK_SIZE)
    }

    pub fn write_all_data(&mut self, mut data: &[u8]) -> io::Result<()> {
        if self.finished { return Err(io::Error::new(io::ErrorKind::BrokenPipe, "archive writer already finished")); }
        while !data.is_empty() {
            let take = (self.block_size - self.buffer.len()).min(data.len());
            self.buffer.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buffer.len() == self.block_size { self.flush_block()?; }
        }
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() { return Ok(()); }
        let (descriptor, frame) = select_block_representation(&self.buffer)?;
        let raw_len = u32::try_from(self.buffer.len()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "block too large"))?;
        let frame_len = u32::try_from(frame.len()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "encoded block too large"))?;
        let descriptor_len = u32::try_from(descriptor.len()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "transform descriptor too large"))?;
        if descriptor_len > MAX_DESCRIPTOR_LEN { return Err(io::Error::new(io::ErrorKind::InvalidData, "transform descriptor exceeds format limit")); }
        let header = BlockHeader { sequence: self.sequence, raw_len, frame_len, descriptor_len, raw_crc32c: crc32c::checksum(&self.buffer) };
        self.inner.write_all(&header.encode())?;
        self.inner.write_all(&descriptor)?;
        self.inner.write_all(&frame)?;
        self.total_input = self.total_input.checked_add(raw_len as u64).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "archive length overflow"))?;
        self.sequence = self.sequence.checked_add(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "too many blocks"))?;
        self.buffer.clear();
        Ok(())
    }

    /// Flushes the final partial block and the underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.flush_block()?;
        self.inner.write_all(&encode_footer(self.sequence, self.total_input))?;
        self.inner.flush()?;
        self.finished = true;
        Ok(self.inner)
    }

    pub fn blocks_written(&self) -> u64 { self.sequence }
    pub fn uncompressed_bytes_written(&self) -> u64 { self.total_input + self.buffer.len() as u64 }
}

impl<W: Write> Write for ArchiveWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_all_data(buf)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flush_block()?;
        self.inner.flush()
    }
}

