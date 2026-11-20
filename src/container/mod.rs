//! PYTHIA streaming single-file and multi-file containers.
pub mod crc32c;
pub mod format;
mod reader;
mod writer;
mod pipeline;
pub use reader::{ArchiveReader, BlockInfo};
pub use writer::ArchiveWriter;

/// Multi-file archive format with per-entry paths, independently checked blocks and a persistent v3 seek index.
pub mod multi;
pub use multi::{ArchiveCostReport, IndexedFileInfo, MultiArchiveIndex, MultiArchiveReader, MultiArchiveWriter, MultiFileInfo};
