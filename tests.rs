//! Cross-module regression tests for the PS6 entropy layer.
//! These tests are source-level additions; they have not been executed here.

use crate::{
    entropy::{self, EntropyCodec},
    ArchiveCost,
};

#[test]
fn all_selected_entropy_frames_roundtrip() {
    let cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        vec![0],
        vec![0; 16_384],
        b"abracadabra abracadabra".repeat(512),
        (0..=255).cycle().take(32_768).collect(),
        (0..32_768).map(|i| ((i * 73 + i / 19) & 255) as u8).collect(),
    ];
    for input in cases {
        let selected = entropy::encode_best(&input).expect("encode candidate set");
        assert_eq!(
            entropy::decode_best(&selected.archive, input.len()).expect("decode selected frame"),
            input
        );
        let minimum = selected.candidates.iter().map(|c| c.archive_bytes).min().unwrap();
        assert_eq!(selected.archive.len(), minimum);
    }
}

#[test]
fn selected_cost_api_charges_exact_frame_without_reencoding() {
    let input = b"context-aware entropy accounting".repeat(256);
    let selected = entropy::encode_best(&input).unwrap();
    let (cost, codec) = entropy::add_selection_to_archive_cost(
        input.len(),
        &selected,
        ArchiveCost::default(),
    )
    .unwrap();
    assert_eq!(codec, selected.codec);
    assert_eq!(cost.total().unwrap(), selected.archive.len() as u64);
}

#[test]
fn corrupt_and_truncated_frames_are_rejected() {
    let input = b"PYTHIA corruption regression".repeat(128);
    let selected = entropy::encode_best(&input).unwrap();
    for cut in 0..selected.archive.len() {
        assert!(entropy::decode_best(&selected.archive[..cut], input.len()).is_err());
    }
    let mut bad_magic = selected.archive.clone();
    bad_magic[0] ^= 0x80;
    assert!(entropy::decode_best(&bad_magic, input.len()).is_err());
}

#[test]
fn raw_is_never_reported_as_larger_than_its_complete_frame_baseline() {
    let input: Vec<u8> = (0..65_536)
        .map(|i| ((i * 149 + i / 13) & 255) as u8)
        .collect();
    let selected = entropy::encode_best(&input).unwrap();
    assert!(selected.archive.len() <= 14 + input.len());
    assert!(matches!(
        selected.codec,
        EntropyCodec::Raw
            | EntropyCodec::AdaptiveArithmetic
            | EntropyCodec::Rans
            | EntropyCodec::ContextRans
            | EntropyCodec::ContextRansBackoff
    ));
}

#[test]
fn container_stream_roundtrip_multiple_blocks_and_empty_input() {
    use crate::container::{ArchiveReader, ArchiveWriter};
    use std::io::{Cursor, Read, Write};
    for input in [Vec::new(), vec![7u8; 9000], (0..40_000).map(|i| ((i * 31 + i / 17) & 255) as u8).collect()] {
        let mut writer = ArchiveWriter::new(Vec::new(), 4096).unwrap();
        writer.write_all(&input).unwrap();
        let archive = writer.finish().unwrap();
        let mut reader = ArchiveReader::new(Cursor::new(archive)).unwrap();
        let mut output = Vec::new();
        reader.read_to_end(&mut output).unwrap();
        assert_eq!(output, input);
    }
}

#[test]
fn container_rejects_truncated_footer_and_corrupted_block() {
    use crate::container::{ArchiveReader, ArchiveWriter};
    use std::io::{Cursor, Read, Write};
    let input = b"container corruption check".repeat(300);
    let mut writer = ArchiveWriter::new(Vec::new(), 4096).unwrap();
    writer.write_all(&input).unwrap();
    let archive = writer.finish().unwrap();
    let truncated = archive[..archive.len() - 1].to_vec();
    let mut truncated_reader = ArchiveReader::new(Cursor::new(truncated)).unwrap();
    let mut discarded = Vec::new();
    assert!(truncated_reader.read_to_end(&mut discarded).is_err(), "footer truncation must be detected after the data block");

    let mut corrupt = archive.clone();
    const SUPERBLOCK_LEN: usize = crate::container::format::SUPERBLOCK_LEN;
    const BLOCK_HEADER_LEN: usize = crate::container::format::BLOCK_HEADER_LEN;
    let descriptor_len_offset = SUPERBLOCK_LEN + 20;
    let descriptor_len = u32::from_le_bytes(corrupt[descriptor_len_offset..descriptor_len_offset + 4].try_into().unwrap()) as usize;
    let first_frame_byte = SUPERBLOCK_LEN + BLOCK_HEADER_LEN + descriptor_len;
    corrupt[first_frame_byte] ^= 0x20;
    assert!(ArchiveReader::new(Cursor::new(corrupt)).unwrap().next_block().is_err());
}

#[test]
fn reader_keeps_ps7_1_major_one_minor_one_compatibility() {
    use crate::container::{ArchiveReader, ArchiveWriter};
    use std::io::{Cursor, Read, Write};
    let original = b"legacy PS7.1 32-byte block header".repeat(128);
    let mut writer = ArchiveWriter::new(Vec::new(), 4096).unwrap();
    writer.write_all(&original).unwrap();
    let mut archive = writer.finish().unwrap();
    // Convert the v2.0 superblock into a v1.1 superblock. The block layout is
    // unchanged between these versions; only the major version differs.
    archive[8..10].copy_from_slice(&1u16.to_le_bytes());
    archive[10..12].copy_from_slice(&1u16.to_le_bytes());
    let header_crc = crate::container::crc32c::checksum(&archive[..32]);
    archive[32..36].copy_from_slice(&header_crc.to_le_bytes());
    let mut reader = ArchiveReader::new(Cursor::new(archive)).unwrap();
    let mut restored = Vec::new();
    reader.read_to_end(&mut restored).unwrap();
    assert_eq!(restored, original);
}

#[test]
fn container_crc32c_known_vector() {
    assert_eq!(crate::container::crc32c::checksum(b"123456789"), 0xE306_9283);
}

#[test]
fn transformed_container_blocks_roundtrip_and_report_descriptors() {
    use crate::container::{ArchiveReader, ArchiveWriter};
    use std::io::{Cursor, Write};
    let input: Vec<u8> = (0..65_536).map(|i| ((i * 17 + i / 256) & 255) as u8).collect();
    let mut writer = ArchiveWriter::new(Vec::new(), 16_384).unwrap();
    writer.write_all(&input).unwrap();
    let archive = writer.finish().unwrap();
    let mut reader = ArchiveReader::new(Cursor::new(archive)).unwrap();
    let mut restored = Vec::new();
    let mut saw_valid_metadata = false;
    while let Some((info, block)) = reader.next_block().unwrap() {
        assert!(info.transform_stages <= 2);
        assert_eq!(info.descriptor_len == 0, info.transform_stages == 0);
        saw_valid_metadata |= info.transform_stages > 0;
        restored.extend_from_slice(&block);
    }
    assert_eq!(restored, input);
    assert!(saw_valid_metadata, "the exact-size selector should choose Delta8 for this low-delta stream");
}

#[test]
fn reader_keeps_ps7_0_minor_zero_compatibility() {
    use crate::container::{crc32c, ArchiveReader};
    use crate::container::format::{encode_footer, Superblock, SUPERBLOCK_LEN};
    use crate::entropy::encode_best;
    use std::io::{Cursor, Read};

    let original = b"legacy PS7.0 block layout".repeat(64);
    let frame = encode_best(&original).unwrap().archive;
    let mut archive = Vec::new();
    let superblock = Superblock { major: 1, minor: 0, flags: 0, block_size: 4096, declared_size: u64::MAX };
    archive.extend_from_slice(&superblock.encode());

    let mut header = [0u8; 28];
    header[..4].copy_from_slice(b"PYBK");
    header[4..12].copy_from_slice(&0u64.to_le_bytes());
    header[12..16].copy_from_slice(&(original.len() as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(frame.len() as u32).to_le_bytes());
    header[20..24].copy_from_slice(&crc32c::checksum(&original).to_le_bytes());
    let header_crc = crc32c::checksum(&header[..24]);
    header[24..28].copy_from_slice(&header_crc.to_le_bytes());
    archive.extend_from_slice(&header);
    archive.extend_from_slice(&frame);
    archive.extend_from_slice(&encode_footer(1, original.len() as u64));

    assert_eq!(archive.len() > SUPERBLOCK_LEN, true);
    let mut reader = ArchiveReader::new(Cursor::new(archive)).unwrap();
    let mut restored = Vec::new();
    reader.read_to_end(&mut restored).unwrap();
    assert_eq!(restored, original);
}

#[cfg(test)]
mod multi_archive_tests {
    use std::io::Cursor;
    use crate::container::{MultiArchiveReader, MultiArchiveWriter};

    #[test]
    fn multi_file_roundtrip_files_empty_files_and_directories() {
        let mut writer = MultiArchiveWriter::new(Vec::new(), 1024).unwrap();
        writer.add_directory("docs").unwrap();
        writer.add_file("docs/a.bin", 4097, Cursor::new(vec![0x5a; 4097])).unwrap();
        writer.add_file("empty", 0, Cursor::new(Vec::<u8>::new())).unwrap();
        let bytes = writer.finish().unwrap();
        let mut reader = MultiArchiveReader::new(Cursor::new(bytes)).unwrap();
        let d = reader.next_entry().unwrap().unwrap();
        assert!(d.is_directory);
        assert_eq!(d.path, "docs");
        assert_eq!(reader.copy_current_to(Vec::new()).unwrap(), 0);
        let f = reader.next_entry().unwrap().unwrap();
        assert_eq!(f.path, "docs/a.bin");
        let mut out = Vec::new();
        assert_eq!(reader.copy_current_to(&mut out).unwrap(), 4097);
        assert_eq!(out, vec![0x5a; 4097]);
        let e = reader.next_entry().unwrap().unwrap();
        assert_eq!(e.path, "empty");
        assert_eq!(reader.copy_current_to(Vec::new()).unwrap(), 0);
        assert!(reader.next_entry().unwrap().is_none());
        reader.finish_check().unwrap();
        assert_eq!(reader.files_read(), 3);
        assert_eq!(reader.total_uncompressed_bytes(), 4097);
    }

    #[test]
    fn multi_archive_rejects_traversal_paths() {
        let mut writer = MultiArchiveWriter::new(Vec::new(), 1024).unwrap();
        assert!(writer.add_file("../escape", 0, Cursor::new(Vec::<u8>::new())).is_err());
        assert!(writer.add_directory("C:/escape").is_err());
    }
}

#[cfg(test)]
mod multi_archive_corruption_tests {
    use std::io::Cursor;
    use crate::container::{MultiArchiveReader, MultiArchiveWriter};

    #[test]
    fn multi_archive_rejects_truncated_global_footer() {
        let mut writer = MultiArchiveWriter::new(Vec::new(), 1024).unwrap();
        writer.add_file("x", 1, Cursor::new(vec![7u8])).unwrap();
        let mut archive = writer.finish().unwrap();
        archive.truncate(archive.len() - 3);
        let mut reader = MultiArchiveReader::new(Cursor::new(archive)).unwrap();
        let entry = reader.next_entry().unwrap().unwrap();
        assert_eq!(entry.path, "x");
        reader.copy_current_to(std::io::sink()).unwrap();
        assert!(reader.next_entry().is_err());
    }

    #[test]
    fn multi_archive_rejects_truncated_file_footer() {
        let mut writer = MultiArchiveWriter::new(Vec::new(), 1024).unwrap();
        writer.add_file("x", 1, Cursor::new(vec![7u8])).unwrap();
        let mut archive = writer.finish().unwrap();
        // Remove one byte from the per-file footer while retaining index and global footer.
        let footer_start = archive.len() - 40;
        let index_offset = u64::from_le_bytes(archive[footer_start + 20..footer_start + 28].try_into().unwrap()) as usize;
        let file_footer = index_offset - 32;
        archive.remove(file_footer + 31);
        let mut reader = MultiArchiveReader::new(Cursor::new(archive)).unwrap();
        reader.next_entry().unwrap().unwrap();
        assert!(reader.copy_current_to(std::io::sink()).is_err());
    }
}

#[cfg(test)]
mod multi_archive_writer_state_tests {
    use std::io::Cursor;
    use crate::container::MultiArchiveWriter;

    #[test]
    fn failed_entry_poisoning_prevents_finalizing_partial_archive() {
        let mut writer = MultiArchiveWriter::new(Vec::new(), 1024).unwrap();
        assert!(writer.add_file("wrong-size", 1, Cursor::new(vec![1u8, 2u8])).is_err());
        assert!(writer.add_directory("later").is_err());
        assert!(writer.finish().is_err());
    }
}

#[cfg(test)]
mod multi_transform_pipeline_tests {
    use std::io::Cursor;
    use crate::container::{MultiArchiveReader, MultiArchiveWriter};
    use crate::container::crc32c;

    #[test]
    fn multi_file_blocks_use_shared_transform_pipeline_and_roundtrip() {
        // Delta8 turns this monotonic byte sequence into a highly compressible stream.
        let data: Vec<u8> = (0..32768u32).map(|i| (i & 0xff) as u8).collect();
        let mut writer = MultiArchiveWriter::new(Vec::new(), 32768).unwrap();
        writer.add_file("pattern.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
        let archive = writer.finish().unwrap();

        // Multi-file format 3 announces indexed transform-descriptor support.
        assert_eq!(u16::from_le_bytes([archive[8], archive[9]]), 3);
        let path_len = u16::from_le_bytes([archive[20], archive[21]]) as usize;
        let block_header = 16 + 16 + path_len + 4;
        let descriptor_len = u32::from_le_bytes(archive[block_header + 20..block_header + 24].try_into().unwrap()) as usize;
        assert!(descriptor_len > 0, "expected a transform descriptor for the patterned input");

        let mut reader = MultiArchiveReader::new(Cursor::new(archive)).unwrap();
        let entry = reader.next_entry().unwrap().unwrap();
        assert_eq!(entry.path, "pattern.bin");
        let mut restored = Vec::new();
        reader.copy_current_to(&mut restored).unwrap();
        assert_eq!(restored, data);
        assert!(reader.next_entry().unwrap().is_none());
        reader.finish_check().unwrap();
    }

    #[test]
    fn multi_file_reader_keeps_version_one_empty_archive_compatibility() {
        let mut archive = vec![0u8;16];
        archive[..8].copy_from_slice(b"PYTHM\r\n\x1a");
        archive[8..10].copy_from_slice(&1u16.to_le_bytes());
        let header_crc = crc32c::checksum(&archive[..12]);
        archive[12..16].copy_from_slice(&header_crc.to_le_bytes());
        let mut footer=[0u8;24]; footer[..4].copy_from_slice(b"PYME");
        let footer_crc=crc32c::checksum(&footer[..20]); footer[20..24].copy_from_slice(&footer_crc.to_le_bytes());
        archive.extend_from_slice(&footer);
        let mut reader = MultiArchiveReader::new(Cursor::new(archive)).unwrap();
        assert!(reader.next_entry().unwrap().is_none());
        reader.finish_check().unwrap();
    }
}

#[cfg(test)]
mod ps74_parallel_tests {
    use std::io::Cursor;
    use crate::container::{MultiArchiveReader, MultiArchiveWriter};

    #[test]
    fn parallel_multifile_encoding_is_byte_deterministic() {
        let data: Vec<u8> = (0..180_000usize)
            .map(|i| ((i * 37 + i / 11 + (i >> 7)) & 255) as u8)
            .collect();
        let mut serial = MultiArchiveWriter::new(Vec::new(), 16_384).unwrap();
        serial.add_file("data.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
        let serial_bytes = serial.finish().unwrap();

        let mut parallel = MultiArchiveWriter::new(Vec::new(), 16_384).unwrap()
            .with_parallelism(4).unwrap();
        parallel.add_file("data.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
        let parallel_bytes = parallel.finish().unwrap();
        assert_eq!(serial_bytes, parallel_bytes, "parallel completion order must not affect archive bytes");

        let mut reader = MultiArchiveReader::new(Cursor::new(parallel_bytes)).unwrap();
        assert_eq!(reader.next_entry().unwrap().unwrap().path, "data.bin");
        let mut restored = Vec::new();
        reader.copy_current_to(&mut restored).unwrap();
        assert_eq!(restored, data);
        assert!(reader.next_entry().unwrap().is_none());
        reader.finish_check().unwrap();
    }

    #[test]
    fn adaptive_block_mode_roundtrips_low_and_high_entropy_files() {
        let cases = [
            vec![0u8; 5 * 1024 * 1024],
            (0..900_000usize).map(|i| ((i * 73 + i / 17) & 255) as u8).collect(),
        ];
        for data in cases {
            let mut writer = MultiArchiveWriter::new(Vec::new(), 1024 * 1024).unwrap()
                .with_adaptive_blocks(true);
            writer.add_file("adaptive.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
            let archive = writer.finish().unwrap();
            let mut reader = MultiArchiveReader::new(Cursor::new(archive)).unwrap();
            reader.next_entry().unwrap().unwrap();
            let mut restored = Vec::new();
            reader.copy_current_to(&mut restored).unwrap();
            assert_eq!(restored, data);
            assert!(reader.next_entry().unwrap().is_none());
            reader.finish_check().unwrap();
        }
    }

    #[test]
    fn rejects_zero_or_excessive_parallelism() {
        assert!(MultiArchiveWriter::new(Vec::new(), 1024).unwrap().with_parallelism(0).is_err());
        assert!(MultiArchiveWriter::new(Vec::new(), 1024).unwrap().with_parallelism(65).is_err());
    }
}


#[cfg(test)]
mod ps75_index_tests {
    use std::io::Cursor;
    use crate::container::{MultiArchiveIndex, MultiArchiveReader, MultiArchiveWriter};

    #[test]
    fn indexed_reader_extracts_one_file_by_offsets() {
        let first=vec![0x31u8;90_000];
        let second:Vec<u8>=(0..140_000usize).map(|i|((i*19+i/13)&255) as u8).collect();
        let mut w=MultiArchiveWriter::new(Vec::new(),16_384).unwrap();
        w.add_file("alpha.bin",first.len() as u64,Cursor::new(first)).unwrap();
        w.add_directory("folder").unwrap();
        w.add_file("folder/target.bin",second.len() as u64,Cursor::new(second.clone())).unwrap();
        let archive=w.finish().unwrap();
        let mut index=MultiArchiveIndex::open(Cursor::new(archive.clone())).unwrap();
        let listed:Vec<String>=index.entries().map(|e|e.path.clone()).collect();
        assert_eq!(listed,vec![String::from("alpha.bin"),String::from("folder"),String::from("folder/target.bin")]);
        let mut restored=Vec::new();
        assert_eq!(index.extract_file_to("folder/target.bin",&mut restored).unwrap(),second.len() as u64);
        assert_eq!(restored,second);
        let mut sequential=MultiArchiveReader::new(Cursor::new(archive)).unwrap();
        while let Some(_) = sequential.next_entry().unwrap(){sequential.copy_current_to(std::io::sink()).unwrap();}
        sequential.finish_check().unwrap();
    }

    #[test]
    fn indexed_range_extraction_decodes_only_overlapping_blocks() {
        let source:Vec<u8>=(0..250_000usize).map(|i|((i*37+i/17+11)&255) as u8).collect();
        let mut w=MultiArchiveWriter::new(Vec::new(),32*1024).unwrap();
        w.add_file("range.bin",source.len() as u64,Cursor::new(source.clone())).unwrap();
        let archive=w.finish().unwrap();
        let mut index=MultiArchiveIndex::open(Cursor::new(archive)).unwrap();
        let start=31_000u64; let length=70_000u64;
        let mut part=Vec::new();
        let n=index.extract_file_range_to("range.bin",start,length,&mut part).unwrap();
        assert_eq!(n,length);
        assert_eq!(part,&source[start as usize..(start+length) as usize]);
        let mut tail=Vec::new();
        let n=index.extract_file_range_to("range.bin",source.len() as u64-123,1000,&mut tail).unwrap();
        assert_eq!(n,123);
        assert_eq!(tail,&source[source.len()-123..]);
        assert!(index.extract_file_range_to("range.bin",source.len() as u64+1,1,Vec::new()).is_err());
        assert!(index.extract_file_range_to("range.bin",1,u64::MAX,Vec::new()).is_err());
    }

    #[test]
    fn random_access_does_not_decode_unrequested_files() {
        let first:Vec<u8>=(0..80_000usize).map(|i|((i*71+i/9)&255) as u8).collect();
        let target=b"target file remains readable".repeat(3000);
        let mut w=MultiArchiveWriter::new(Vec::new(),64*1024).unwrap();
        w.add_file("broken.bin",first.len() as u64,Cursor::new(first)).unwrap();
        w.add_file("target.bin",target.len() as u64,Cursor::new(target.clone())).unwrap();
        let mut archive=w.finish().unwrap();
        let first_path_len=u16::from_le_bytes(archive[20..22].try_into().unwrap()) as usize;
        let first_block=16+16+first_path_len+4;
        let descriptor_len=u32::from_le_bytes(archive[first_block+20..first_block+24].try_into().unwrap()) as usize;
        let payload=first_block+32+descriptor_len;
        archive[payload]^=0x40;
        let mut index=MultiArchiveIndex::open(Cursor::new(archive)).unwrap();
        let mut restored=Vec::new();
        index.extract_file_to("target.bin",&mut restored).unwrap();
        assert_eq!(restored,target);
    }

    #[test]
    fn index_crc_and_duplicate_paths_are_rejected() {
        let mut w=MultiArchiveWriter::new(Vec::new(),1024).unwrap();
        w.add_file("same",1,Cursor::new(vec![1])).unwrap();
        assert!(w.add_file("same",1,Cursor::new(vec![2])).is_err());
        assert!(w.finish().is_err());
        let mut w=MultiArchiveWriter::new(Vec::new(),1024).unwrap();
        w.add_file("x",1,Cursor::new(vec![1])).unwrap();
        let mut archive=w.finish().unwrap();
        let footer=archive.len()-40;
        let index_offset=u64::from_le_bytes(archive[footer+20..footer+28].try_into().unwrap()) as usize;
        archive[index_offset+12]^=1;
        assert!(MultiArchiveIndex::open(Cursor::new(archive)).is_err());
    }
}


#[cfg(test)]
mod ps77_tests {
    use std::io::Cursor;
    use crate::container::{ArchiveCostReport, MultiArchiveIndex, MultiArchiveWriter};

    #[test]
    fn streaming_index_writer_roundtrips_and_cost_sums_to_physical_size() {
        let mut writer=MultiArchiveWriter::new(Vec::new(), 4096).unwrap();
        writer.add_directory("docs").unwrap();
        writer.add_file("docs/a.bin", 12000, Cursor::new(vec![0x41;12000])).unwrap();
        writer.add_file("b.bin", 3, Cursor::new(b"xyz".to_vec())).unwrap();
        let archive=writer.finish().unwrap();
        let mut index=MultiArchiveIndex::open(Cursor::new(archive.clone())).unwrap();
        let report:ArchiveCostReport=index.cost_report();
        assert_eq!(report.archive_bytes,archive.len() as u64);
        assert_eq!(report.accounted_bytes().unwrap(),archive.len() as u64);
        assert_eq!(report.source_bytes,12003);
        let mut restored=Vec::new();
        index.extract_file_to("docs/a.bin",&mut restored).unwrap();
        assert_eq!(restored,vec![0x41;12000]);
    }
}


#[cfg(test)]
mod ps78_budget_tests {
    use std::io::Cursor;
    use crate::container::{MultiArchiveIndex, MultiArchiveWriter};

    #[test]
    fn raw_memory_budget_caps_batches_and_archive_roundtrips() {
        let source: Vec<u8> = (0..50_000).map(|i| ((i * 31) % 251) as u8).collect();
        let mut writer = MultiArchiveWriter::new(Vec::new(), 16 * 1024).unwrap()
            .with_parallelism(8).unwrap()
            .with_adaptive_blocks(true)
            .with_raw_memory_budget(4096).unwrap();
        writer.add_file("budget/data.bin", source.len() as u64, Cursor::new(source.clone())).unwrap();
        let archive = writer.finish().unwrap();
        let mut index = MultiArchiveIndex::open(Cursor::new(archive)).unwrap();
        let mut restored = Vec::new();
        index.extract_file_to("budget/data.bin", &mut restored).unwrap();
        assert_eq!(restored, source);
    }

    #[test]
    fn zero_raw_memory_budget_is_rejected() {
        assert!(MultiArchiveWriter::new(Vec::new(), 1024).unwrap().with_raw_memory_budget(0).is_err());
    }
}
