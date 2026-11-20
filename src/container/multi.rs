//! Multi-file PYTHIA container. Paths are UTF-8 relative paths using `/` separators.
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{
    chain::{ChainResult, ModelChain},
    domain::Domain,
    entropy::{decode_best},
};
use super::pipeline::{adaptive_block_size, select_block_representation, select_block_representations_parallel};
use super::{crc32c, format::{decode_block_descriptor, max_transformed_block_len, BlockHeader, BLOCK_HEADER_LEN}};

const MAGIC: &[u8; 8] = b"PYTHM\r\n\x1a";
const FILE_MAGIC: &[u8; 4] = b"PYFE";
const BLOCK_MAGIC: &[u8; 4] = b"PYBK";
const FILE_END_MAGIC: &[u8; 4] = b"PYFF";
const END_MAGIC: &[u8; 4] = b"PYME";
const INDEX_MAGIC: &[u8; 4] = b"PYIX";
const VERSION: u16 = 5;
const V4_VERSION: u16 = 4;
const V3_VERSION: u16 = 3;
const V2_VERSION: u16 = 2;
const LEGACY_VERSION: u16 = 1;
const MAX_INDEX_BYTES: u64 = 256 * 1024 * 1024;
const MAX_INDEX_ENTRIES: u64 = 1_000_000;
const MAX_PATH: usize = 4096;
const MAX_BLOCK: usize = 16 * 1024 * 1024;

fn invalid(s: &'static str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }
fn invalid_owned(s: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }
fn put_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn get_u32<R: Read>(r: &mut R) -> io::Result<u32> { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiFileInfo { pub path: String, pub size: u64, pub is_directory: bool }

struct CountedWriter<W: Write> { inner: W, position: u64 }
impl<W: Write> CountedWriter<W> {
    fn new(inner: W) -> Self { Self { inner, position: 0 } }
    fn position(&self) -> u64 { self.position }
    fn into_inner(self) -> W { self.inner }
}
impl<W: Write> Write for CountedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.position = self.position.checked_add(n as u64).ok_or_else(|| invalid("archive offset overflow"))?;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

#[derive(Debug, Clone)]
struct IndexEntry {
    info: MultiFileInfo,
    entry_offset: u64,
    end_offset: u64,
    block_count: u32,
    spool_offset: u64,
}

/// Disk-backed store for block offsets. Only a compact per-file descriptor is
/// retained in RAM; one u64 per data block is appended to this private file.
struct IndexSpool { file: Option<File>, path: PathBuf, len: u64 }
impl IndexSpool {
    fn create() -> io::Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir();
        for _ in 0..128 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("pythia-index-{}-{}.tmp", std::process::id(), id));
            match OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
                Ok(file) => return Ok(Self { file: Some(file), path, len: 0 }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(io::ErrorKind::AlreadyExists, "could not allocate a unique PYTHIA index spool"))
    }
    fn position(&self) -> u64 { self.len }
    fn push(&mut self, offset: u64) -> io::Result<()> {
        let next = self.len.checked_add(8).ok_or_else(|| invalid("index spool length overflow"))?;
        if next > MAX_INDEX_BYTES - 24 { return Err(invalid("index offset spool exceeds index safety limit")); }
        self.file.as_mut().ok_or_else(|| invalid("index spool is closed"))?.write_all(&offset.to_le_bytes())?;
        self.len = next;
        Ok(())
    }
    fn read_offsets(&mut self, start: u64, count: u32, mut emit: impl FnMut(u64) -> io::Result<()>) -> io::Result<()> {
        let bytes = (count as u64).checked_mul(8).ok_or_else(|| invalid("index spool range overflow"))?;
        let end = start.checked_add(bytes).ok_or_else(|| invalid("index spool range overflow"))?;
        if end > self.len { return Err(invalid("index spool range exceeds written data")); }
        let file = self.file.as_mut().ok_or_else(|| invalid("index spool is closed"))?;
        file.seek(SeekFrom::Start(start))?;
        for _ in 0..count {
            let mut raw = [0u8; 8];
            file.read_exact(&mut raw)?;
            emit(u64::from_le_bytes(raw))?;
        }
        Ok(())
    }
}
impl Drop for IndexSpool {
    fn drop(&mut self) { drop(self.file.take()); let _ = fs::remove_file(&self.path); }
}

pub struct MultiArchiveWriter<W: Write> {
    out: CountedWriter<W>,
    block_size: usize,
    files: u64,
    total: u64,
    seq: u64,
    failed: bool,
    workers: usize,
    adaptive_blocks: bool,
    raw_memory_budget: Option<usize>,
    index: Vec<IndexEntry>,
    index_spool: IndexSpool,
    names: std::collections::HashSet<String>,
}
impl<W: Write> MultiArchiveWriter<W> {
    pub fn new(out: W, block_size: usize) -> io::Result<Self> {
        if block_size == 0 || block_size > MAX_BLOCK { return Err(io::Error::new(io::ErrorKind::InvalidInput,"block size outside 1..=16 MiB")); }
        let index_spool = IndexSpool::create()?;
        let mut h=[0u8;16]; h[..8].copy_from_slice(MAGIC); h[8..10].copy_from_slice(&VERSION.to_le_bytes());
        let c=crc32c::checksum(&h[..12]); h[12..16].copy_from_slice(&c.to_le_bytes()); let mut out=CountedWriter::new(out); out.write_all(&h)?;
        Ok(Self{out,block_size,files:0,total:0,seq:0,failed:false,workers:1,adaptive_blocks:false,raw_memory_budget:None,index:Vec::new(),index_spool,names:std::collections::HashSet::new()})
    }
    /// Sets the maximum number of independent blocks encoded concurrently.
    /// Memory use is bounded approximately by `workers * block_size`, plus encoded frames.
    pub fn with_parallelism(mut self, workers: usize) -> io::Result<Self> {
        if workers == 0 || workers > 64 { return Err(io::Error::new(io::ErrorKind::InvalidInput, "parallelism must be in 1..=64")); }
        self.workers = workers;
        Ok(self)
    }

    /// Enables deterministic per-file block sizing based on a 64 KiB prefix sample.
    pub fn with_adaptive_blocks(mut self, enabled: bool) -> Self {
        self.adaptive_blocks = enabled;
        self
    }

    /// Caps the aggregate raw input buffers staged for one compression batch.
    /// Encoded frames and transform workspaces add implementation-dependent overhead.
    pub fn with_raw_memory_budget(mut self, bytes: usize) -> io::Result<Self> {
        if bytes == 0 { return Err(io::Error::new(io::ErrorKind::InvalidInput, "raw memory budget must be nonzero")); }
        self.raw_memory_budget = Some(bytes);
        Ok(self)
    }

    pub fn add_file<R: Read>(&mut self, path: &str, input_size: u64, input: R) -> io::Result<()> {
        if self.failed { return Err(io::Error::new(io::ErrorKind::BrokenPipe,"archive writer is poisoned by an earlier error")); }
        let result = self.add_file_inner(path, input_size, input);
        if result.is_err() { self.failed = true; }
        result
    }
    fn add_file_inner<R: Read>(&mut self, path: &str, size: u64, mut input: R) -> io::Result<()> {
        validate_path(path)?;
        if self.names.contains(path) { return Err(io::Error::new(io::ErrorKind::AlreadyExists,"duplicate archive path")); }
        if path.len()>MAX_PATH { return Err(io::Error::new(io::ErrorKind::InvalidInput,"path too long")); }
        let entry_offset=self.out.position();
        let spool_offset=self.index_spool.position();
        let mut fixed=[0u8;16]; fixed[..4].copy_from_slice(FILE_MAGIC); fixed[4..6].copy_from_slice(&(path.len() as u16).to_le_bytes()); fixed[6..8].copy_from_slice(&0u16.to_le_bytes()); fixed[8..16].copy_from_slice(&size.to_le_bytes());
        let mut crc_data=fixed.to_vec(); crc_data.extend_from_slice(path.as_bytes());
        self.out.write_all(&fixed)?; self.out.write_all(path.as_bytes())?; put_u32(&mut self.out,crc32c::checksum(&crc_data))?;
        let mut remaining=size;
        let mut file_total=0u64;
        let mut blocks=0u64;
        let mut probe=vec![0u8;remaining.min(64*1024) as usize];
        input.read_exact(&mut probe)?;
        let mut chosen_block_size=if self.adaptive_blocks { adaptive_block_size(&probe,self.block_size) } else { self.block_size };
        if let Some(budget) = self.raw_memory_budget { chosen_block_size = chosen_block_size.min(budget).max(1); }
        let batch_workers = self.raw_memory_budget
            .map(|budget| self.workers.min((budget / chosen_block_size).max(1)))
            .unwrap_or(self.workers);
        let mut source=std::io::Cursor::new(probe).chain(input);
        remaining=size;
        while remaining>0 {
            let mut raw_batch=Vec::with_capacity(batch_workers);
            for _ in 0..batch_workers {
                if remaining==0 { break; }
                let want=remaining.min(chosen_block_size as u64) as usize;
                let mut raw=vec![0u8;want];
                source.read_exact(&mut raw)?;
                remaining-=want as u64;
                raw_batch.push(raw);
            }
            let encoded_batch=if batch_workers==1 {
                raw_batch.into_iter().map(|raw| {
                    let encoded=select_block_representation(&raw)?;
                    Ok::<_,io::Error>((raw,encoded))
                }).collect::<io::Result<Vec<_>>>()?
            } else {
                select_block_representations_parallel(raw_batch,batch_workers)?
            };
            for (raw,(descriptor,frame)) in encoded_batch.into_iter() {
                let frame_len=u32::try_from(frame.len()).map_err(|_|invalid("frame too large"))?;
                let descriptor_len=u32::try_from(descriptor.len()).map_err(|_|invalid("descriptor too large"))?;
                if descriptor_len > super::format::MAX_DESCRIPTOR_LEN { return Err(invalid("descriptor exceeds format limit")); }
                let header=BlockHeader{sequence:self.seq,raw_len:raw.len() as u32,frame_len,descriptor_len,raw_crc32c:crc32c::checksum(&raw)};
                if blocks >= u32::MAX as u64 { return Err(invalid("too many blocks in one indexed file")); }
                self.index_spool.push(self.out.position())?;
                self.out.write_all(&header.encode())?;
                self.out.write_all(&descriptor)?;
                self.out.write_all(&frame)?;
                self.seq=self.seq.checked_add(1).ok_or_else(||invalid("block sequence overflow"))?;
                file_total=file_total.checked_add(raw.len() as u64).ok_or_else(||invalid("file size overflow"))?;
                blocks=blocks.checked_add(1).ok_or_else(||invalid("block count overflow"))?;
            }
        }
        let mut extra=[0u8;1]; if source.read(&mut extra)?!=0 { return Err(io::Error::new(io::ErrorKind::InvalidInput,"input contains more bytes than declared size")); }
        let end_offset=self.out.position();
        let mut end=[0u8;32]; end[..4].copy_from_slice(FILE_END_MAGIC); end[4..12].copy_from_slice(&self.files.to_le_bytes()); end[12..20].copy_from_slice(&file_total.to_le_bytes()); end[20..28].copy_from_slice(&blocks.to_le_bytes()); let c=crc32c::checksum(&end[..28]); end[28..32].copy_from_slice(&c.to_le_bytes()); self.out.write_all(&end)?;
        self.names.insert(path.to_owned());
        self.index.push(IndexEntry { info: MultiFileInfo { path: path.to_owned(), size: file_total, is_directory: false }, entry_offset, end_offset, block_count: u32::try_from(blocks).map_err(|_|invalid("too many blocks in index entry"))?, spool_offset });
        self.files=self.files.checked_add(1).ok_or_else(||invalid("file count overflow"))?; self.total=self.total.checked_add(file_total).ok_or_else(||invalid("total size overflow"))?; Ok(())
    }
    pub fn add_directory(&mut self,path:&str)->io::Result<()> {
        if self.failed { return Err(io::Error::new(io::ErrorKind::BrokenPipe,"archive writer is poisoned by an earlier error")); }
        let result = self.add_directory_inner(path);
        if result.is_err() { self.failed = true; }
        result
    }
    fn add_directory_inner(&mut self,path:&str)->io::Result<()> {
        validate_path(path)?; if self.names.contains(path){return Err(io::Error::new(io::ErrorKind::AlreadyExists,"duplicate archive path"));} if path.len()>MAX_PATH{return Err(io::Error::new(io::ErrorKind::InvalidInput,"path too long"));}
        let entry_offset=self.out.position();
        let mut fixed=[0u8;16]; fixed[..4].copy_from_slice(FILE_MAGIC); fixed[4..6].copy_from_slice(&(path.len() as u16).to_le_bytes()); fixed[6..8].copy_from_slice(&1u16.to_le_bytes());
        let mut crc_data=fixed.to_vec();crc_data.extend_from_slice(path.as_bytes());self.out.write_all(&fixed)?;self.out.write_all(path.as_bytes())?;put_u32(&mut self.out,crc32c::checksum(&crc_data))?;
        let end_offset=self.out.position();
        let mut end=[0u8;32];end[..4].copy_from_slice(FILE_END_MAGIC);end[4..12].copy_from_slice(&self.files.to_le_bytes());let c=crc32c::checksum(&end[..28]);end[28..32].copy_from_slice(&c.to_le_bytes());self.out.write_all(&end)?;
        self.names.insert(path.to_owned());
        self.index.push(IndexEntry { info: MultiFileInfo { path: path.to_owned(), size: 0, is_directory: true }, entry_offset, end_offset, block_count: 0, spool_offset: self.index_spool.position() });
        self.files=self.files.checked_add(1).ok_or_else(||invalid("file count overflow"))?;Ok(())
    }
    pub fn finish(mut self)->io::Result<W>{
        if self.failed { return Err(io::Error::new(io::ErrorKind::BrokenPipe,"cannot finish an archive after a write error")); }
        let index_offset=self.out.position();
        let index_len=index_encoded_len(&self.index)?;
        write_index(&mut self.out, &self.index, &mut self.index_spool, index_len)?;
        let mut footer=[0u8;40]; footer[..4].copy_from_slice(END_MAGIC); footer[4..12].copy_from_slice(&self.files.to_le_bytes()); footer[12..20].copy_from_slice(&self.total.to_le_bytes()); footer[20..28].copy_from_slice(&index_offset.to_le_bytes()); footer[28..36].copy_from_slice(&index_len.to_le_bytes()); let c=crc32c::checksum(&footer[..36]); footer[36..40].copy_from_slice(&c.to_le_bytes());
        self.out.write_all(&footer)?; self.out.flush()?; Ok(self.out.into_inner())
    }
}

pub struct MultiArchiveReader<R: Read>{input:R,version:u16,expected_seq:u64,files:u64,total:u64,ended:bool,current:Option<(MultiFileInfo,u64,u64)>}
impl<R:Read> MultiArchiveReader<R>{
 pub fn new(mut input:R)->io::Result<Self>{let mut h=[0u8;16];input.read_exact(&mut h)?;if &h[..8]!=MAGIC{return Err(invalid("not a PYTHIA multi-file archive"));}let version=u16::from_le_bytes([h[8],h[9]]);if (version!=VERSION&&version!=V4_VERSION&&version!=V3_VERSION&&version!=V2_VERSION&&version!=LEGACY_VERSION)||h[10..12]!=[0,0]{return Err(invalid("unsupported multi-file version or flags"));}if crc32c::checksum(&h[..12])!=u32::from_le_bytes(h[12..16].try_into().unwrap()){return Err(invalid("multi-file header CRC mismatch"));}Ok(Self{input,version,expected_seq:0,files:0,total:0,ended:false,current:None})}
 pub fn next_entry(&mut self)->io::Result<Option<MultiFileInfo>>{if self.current.is_some(){return Err(io::Error::new(io::ErrorKind::InvalidInput,"consume current entry first"));}if self.ended{return Ok(None);}let mut prefix=[0u8;4];self.input.read_exact(&mut prefix)?;if &prefix==INDEX_MAGIC && self.version>=V3_VERSION {
    let mut lenbuf=[0u8;8]; self.input.read_exact(&mut lenbuf)?; let index_len=u64::from_le_bytes(lenbuf);
    if index_len<24 || index_len>MAX_INDEX_BYTES { return Err(invalid("index length outside safety limit")); }
    let parsed=read_index_infos_stream(&mut self.input,index_len,&lenbuf)?; let mut indexed_total=0u64; for item in &parsed { indexed_total=indexed_total.checked_add(item.size).ok_or_else(||invalid("index total overflow"))?; } if parsed.len() as u64!=self.files||indexed_total!=self.total{return Err(invalid("index counters do not match archive contents"));}
    let mut footer=[0u8;40]; self.input.read_exact(&mut footer)?;
    if &footer[..4]!=END_MAGIC || u64::from_le_bytes(footer[4..12].try_into().unwrap())!=self.files || u64::from_le_bytes(footer[12..20].try_into().unwrap())!=self.total || u64::from_le_bytes(footer[28..36].try_into().unwrap())!=index_len || crc32c::checksum(&footer[..36])!=u32::from_le_bytes(footer[36..40].try_into().unwrap()) { return Err(invalid("indexed archive footer mismatch")); }
    let mut trailing=[0u8;1]; if self.input.read(&mut trailing)?!=0{return Err(invalid("trailing data after archive footer"));} self.ended=true; return Ok(None);
}
if &prefix==END_MAGIC&&self.version<V3_VERSION{let mut tail=[0u8;20];self.input.read_exact(&mut tail)?;let mut e=[0u8;24];e[..4].copy_from_slice(&prefix);e[4..].copy_from_slice(&tail);if u64::from_le_bytes(e[4..12].try_into().unwrap())!=self.files||u64::from_le_bytes(e[12..20].try_into().unwrap())!=self.total||crc32c::checksum(&e[..20])!=u32::from_le_bytes(e[20..24].try_into().unwrap()){return Err(invalid("archive footer mismatch"));}let mut trailing=[0u8;1];if self.input.read(&mut trailing)?!=0{return Err(invalid("trailing data after archive footer"));}self.ended=true;return Ok(None);}if &prefix!=FILE_MAGIC{return Err(invalid("expected file entry or archive footer"));}let mut fixed=[0u8;16];fixed[..4].copy_from_slice(&prefix);self.input.read_exact(&mut fixed[4..])?;let plen=u16::from_le_bytes(fixed[4..6].try_into().unwrap()) as usize;let flags=u16::from_le_bytes(fixed[6..8].try_into().unwrap());let size=u64::from_le_bytes(fixed[8..16].try_into().unwrap());if plen==0||plen>MAX_PATH||flags>1||(flags==1&&size!=0){return Err(invalid("invalid file entry metadata"));}let mut p=vec![0;plen];self.input.read_exact(&mut p)?;let path=String::from_utf8(p.clone()).map_err(|_|invalid("path is not UTF-8"))?;validate_path(&path)?;let got=get_u32(&mut self.input)?;let mut c=fixed.to_vec();c.extend_from_slice(&p);if crc32c::checksum(&c)!=got{return Err(invalid("file entry CRC mismatch"));}let info=MultiFileInfo{path,size,is_directory:flags==1};self.current=Some((info.clone(),0,0));Ok(Some(info))}
 pub fn copy_current_to<W:Write>(&mut self,mut out:W)->io::Result<u64>{let(info,mut bytes,mut blocks)=self.current.take().ok_or_else(||io::Error::new(io::ErrorKind::InvalidInput,"no current entry"))?;if info.is_directory{self.read_file_end(self.files,0,0)?;self.files=self.files.checked_add(1).ok_or_else(||invalid("file count overflow"))?;return Ok(0);}while bytes<info.size{let mut prefix=[0u8;4];self.input.read_exact(&mut prefix)?;if &prefix!=BLOCK_MAGIC{return Err(invalid("expected data block"));}let mut hb=[0u8;BLOCK_HEADER_LEN];hb[..4].copy_from_slice(&prefix);self.input.read_exact(&mut hb[4..])?;let h=BlockHeader::decode(&hb,MAX_BLOCK as u32)?;if h.sequence!=self.expected_seq{return Err(invalid("invalid multi-file block sequence"));}
 let mut descriptor=vec![0u8;h.descriptor_len as usize];self.input.read_exact(&mut descriptor)?;
 if self.version==LEGACY_VERSION&&!descriptor.is_empty(){return Err(invalid("legacy multi-file archive cannot contain transform descriptors"));}
 let mut frame=vec![0;h.frame_len as usize];self.input.read_exact(&mut frame)?;
 let transformed=decode_best(&frame,max_transformed_block_len(h.raw_len)).map_err(|e|invalid_owned(format!("entropy frame rejected: {e}")))?;
 let raw=if descriptor.is_empty(){transformed}else{
  let desc=decode_block_descriptor(&descriptor).map_err(|e|invalid_owned(format!("transform descriptor rejected: {e}")))?;
  if desc.input_domain!=Domain::Bytes||desc.output_domain!=Domain::Bytes||desc.stages.is_empty(){return Err(invalid("unsupported transform descriptor"));}
  let count=desc.stages.len();let chain=ModelChain::new(desc.stages);
  chain.validate_domains(Domain::Bytes,Domain::Bytes).map_err(|e|invalid_owned(format!("transform chain rejected: {e}")))?;
  chain.inverse(&ChainResult{data:transformed,auxiliary:vec![Vec::new();count]}).map_err(|e|invalid_owned(format!("inverse transform failed: {e}")))?
 };
 if raw.len()!=h.raw_len as usize{return Err(invalid("decoded block length mismatch"));}
 if crc32c::checksum(&raw)!=h.raw_crc32c{return Err(invalid("block data CRC mismatch"));}bytes=bytes.checked_add(raw.len() as u64).ok_or_else(||invalid("file size overflow"))?;if bytes>info.size{return Err(invalid("block data exceeds declared file size"));}out.write_all(&raw)?;blocks+=1;self.expected_seq=self.expected_seq.checked_add(1).ok_or_else(||invalid("block sequence overflow"))?;}
 self.read_file_end(self.files,bytes,blocks)?;self.total=self.total.checked_add(bytes).ok_or_else(||invalid("archive size overflow"))?;self.files=self.files.checked_add(1).ok_or_else(||invalid("file count overflow"))?;Ok(bytes)}
 fn read_file_end(&mut self,file_id:u64,bytes:u64,blocks:u64)->io::Result<()>{let mut e=[0u8;32];self.input.read_exact(&mut e)?;if &e[..4]!=FILE_END_MAGIC{return Err(invalid("missing file footer"));}if u64::from_le_bytes(e[4..12].try_into().unwrap())!=file_id||u64::from_le_bytes(e[12..20].try_into().unwrap())!=bytes||u64::from_le_bytes(e[20..28].try_into().unwrap())!=blocks||crc32c::checksum(&e[..28])!=u32::from_le_bytes(e[28..32].try_into().unwrap()){return Err(invalid("file footer mismatch"));}Ok(())}
 pub fn finish_check(&mut self)->io::Result<()>{if self.current.is_some(){return Err(invalid("unconsumed entry"));}if !self.ended{return Err(invalid("archive footer has not been consumed"));}Ok(())}
 pub fn files_read(&self)->u64{self.files} pub fn total_uncompressed_bytes(&self)->u64{self.total}
}

pub fn validate_path(path:&str)->io::Result<()>{
    if path.is_empty()||path.starts_with('/')||path.starts_with('\\')||path.contains('\\')||path.as_bytes().contains(&0){return Err(invalid("unsafe archive path"));}
    for part in path.split('/') {
        if part.is_empty()||part=="."||part==".."||part.contains(':')||part.ends_with(' ')||part.ends_with('.')||part.chars().any(|c|c.is_control()||"<>\"|?*".contains(c)){return Err(invalid("unsafe archive path component"));}
        let stem=part.split('.').next().unwrap_or("").to_ascii_uppercase();
        if matches!(stem.as_str(),"CON"|"PRN"|"AUX"|"NUL"|"COM1"|"COM2"|"COM3"|"COM4"|"COM5"|"COM6"|"COM7"|"COM8"|"COM9"|"LPT1"|"LPT2"|"LPT3"|"LPT4"|"LPT5"|"LPT6"|"LPT7"|"LPT8"|"LPT9"){return Err(invalid("reserved platform path component"));}
    }
    Ok(())
}

fn validate_index_hierarchy<'a>(items: impl Iterator<Item = (&'a str, bool)>) -> io::Result<()> {
    let mut kinds=std::collections::HashMap::<String,bool>::new();
    for (path,is_dir) in items { kinds.insert(path.to_owned(),is_dir); }
    for path in kinds.keys() {
        let mut prefix=String::new();
        let parts:Vec<&str>=path.split('/').collect();
        for part in parts.iter().take(parts.len().saturating_sub(1)) {
            if !prefix.is_empty(){prefix.push('/');}
            prefix.push_str(part);
            if matches!(kinds.get(&prefix),Some(false)){return Err(invalid("a file entry cannot be a parent directory"));}
        }
    }
    Ok(())
}

fn index_encoded_len(entries: &[IndexEntry]) -> io::Result<u64> {
    if entries.len() as u64 > MAX_INDEX_ENTRIES { return Err(invalid("too many index entries")); }
    validate_index_hierarchy(entries.iter().map(|e|(e.info.path.as_str(),e.info.is_directory)))?;
    let mut len=24u64; // magic + length + count + trailing CRC
    for e in entries {
        validate_path(&e.info.path)?;
        let path_len=u64::try_from(e.info.path.len()).map_err(|_|invalid("indexed path too long"))?;
        let offsets=(e.block_count as u64).checked_mul(8).ok_or_else(||invalid("index length overflow"))?;
        len=len.checked_add(32).and_then(|v|v.checked_add(path_len)).and_then(|v|v.checked_add(offsets)).ok_or_else(||invalid("index length overflow"))?;
    }
    if len>MAX_INDEX_BYTES{return Err(invalid("index exceeds 256 MiB safety limit"));}
    Ok(len)
}

fn write_index<W: Write>(out: &mut W, entries: &[IndexEntry], spool: &mut IndexSpool, total_len: u64) -> io::Result<()> {
    use super::crc32c::Crc32c;
    if index_encoded_len(entries)? != total_len { return Err(invalid("index length changed during serialization")); }
    let mut crc=Crc32c::new();
    let mut emit=|bytes:&[u8]| -> io::Result<()> { out.write_all(bytes)?; crc.update(bytes); Ok(()) };
    emit(INDEX_MAGIC)?;
    emit(&total_len.to_le_bytes())?;
    emit(&(entries.len() as u64).to_le_bytes())?;
    for e in entries {
        let path=e.info.path.as_bytes();
        let path_len=u16::try_from(path.len()).map_err(|_|invalid("indexed path too long"))?;
        let flags=if e.info.is_directory {1u16}else{0u16};
        emit(&path_len.to_le_bytes())?;
        emit(&flags.to_le_bytes())?;
        emit(&e.info.size.to_le_bytes())?;
        emit(&e.entry_offset.to_le_bytes())?;
        emit(&e.end_offset.to_le_bytes())?;
        emit(&e.block_count.to_le_bytes())?;
        emit(path)?;
        spool.read_offsets(e.spool_offset, e.block_count, |offset| emit(&offset.to_le_bytes()))?;
    }
    let checksum=crc.finalize();
    out.write_all(&checksum.to_le_bytes())?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedFileInfo { pub path:String, pub size:u64, pub is_directory:bool, pub block_count:u32 }
#[derive(Debug, Clone)]
struct ParsedIndexEntry {
    info: IndexedFileInfo,
    entry_offset: u64,
    end_offset: u64,
    offsets_offset: u64,
    first_sequence: u64,
}

/// Reads an index scalar while preserving the streaming CRC state.
fn read_index_bytes<R: Read>(input: &mut R, crc: &mut crc32c::Crc32c, n: usize) -> io::Result<Vec<u8>> {
    let mut bytes = vec![0u8; n];
    input.read_exact(&mut bytes)?;
    crc.update(&bytes);
    Ok(bytes)
}
fn read_index_u16<R: Read>(input: &mut R, crc: &mut crc32c::Crc32c) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_index_bytes(input, crc, 2)?.try_into().unwrap()))
}
fn read_index_u32<R: Read>(input: &mut R, crc: &mut crc32c::Crc32c) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_index_bytes(input, crc, 4)?.try_into().unwrap()))
}
fn read_index_u64<R: Read>(input: &mut R, crc: &mut crc32c::Crc32c) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_index_bytes(input, crc, 8)?.try_into().unwrap()))
}

/// Validates a terminal index for the sequential reader without allocating its byte size.
/// `INDEX_MAGIC` and its length field have already been consumed by the caller.
fn read_index_infos_stream<R:Read>(input:&mut R,index_len:u64,lenbuf:&[u8;8])->io::Result<Vec<IndexedFileInfo>>{
    let mut crc=crc32c::Crc32c::new();crc.update(INDEX_MAGIC);crc.update(lenbuf);
    let mut count_raw=[0u8;8];input.read_exact(&mut count_raw)?;crc.update(&count_raw);
    let count=u64::from_le_bytes(count_raw);
    if count>MAX_INDEX_ENTRIES||count>usize::MAX as u64{return Err(invalid("index entry count exceeds limit"));}
    let mut remaining=index_len.checked_sub(24).ok_or_else(||invalid("index length underflow"))?;
    if count>remaining/32{return Err(invalid("index entry table is truncated"));}
    let mut entries=Vec::with_capacity(count as usize);let mut names=std::collections::HashSet::with_capacity(count as usize);
    let mut skip=[0u8;64*1024];
    for _ in 0..count{
        if remaining<32{return Err(invalid("index entry table is truncated"));}
        let mut fixed=[0u8;32];input.read_exact(&mut fixed)?;crc.update(&fixed);remaining-=32;
        let plen=u16::from_le_bytes(fixed[0..2].try_into().unwrap()) as usize;
        let flags=u16::from_le_bytes(fixed[2..4].try_into().unwrap());
        let size=u64::from_le_bytes(fixed[4..12].try_into().unwrap());
        let blocks=u32::from_le_bytes(fixed[28..32].try_into().unwrap());
        if plen==0||plen>MAX_PATH||flags>1||(flags==1&&(size!=0||blocks!=0))||(flags==0&&((size==0&&blocks!=0)||(size>0&&blocks==0))){return Err(invalid("invalid index entry metadata"));}
        if remaining<plen as u64{return Err(invalid("truncated index path"));}
        let mut path_bytes=vec![0u8;plen];input.read_exact(&mut path_bytes)?;crc.update(&path_bytes);remaining-=plen as u64;
        let path=String::from_utf8(path_bytes).map_err(|_|invalid("indexed path is not UTF-8"))?;
        validate_path(&path)?;if !names.insert(path.clone()){return Err(invalid("duplicate path in archive index"));}
        let offset_bytes=(blocks as u64).checked_mul(8).ok_or_else(||invalid("index offset table overflow"))?;
        if offset_bytes>remaining{return Err(invalid("index block-offset table is truncated"));}
        let mut to_skip=offset_bytes;
        while to_skip>0{let n=to_skip.min(skip.len() as u64) as usize;input.read_exact(&mut skip[..n])?;crc.update(&skip[..n]);to_skip-=n as u64;}
        remaining-=offset_bytes;
        entries.push(IndexedFileInfo{path,size,is_directory:flags==1,block_count:blocks});
    }
    if remaining!=0{return Err(invalid("trailing bytes in index"));}
    let mut stored=[0u8;4];input.read_exact(&mut stored)?;
    if crc.finalize()!=u32::from_le_bytes(stored){return Err(invalid("index CRC32C mismatch"));}
    validate_index_hierarchy(entries.iter().map(|e|(e.path.as_str(),e.is_directory)))?;
    Ok(entries)
}

/// Exact, non-overlapping byte accounting for a v3 multi-file archive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArchiveCostReport {
    pub source_bytes:u64, pub archive_bytes:u64, pub archive_header:u64,
    pub file_metadata:u64, pub block_headers:u64, pub descriptors:u64,
    pub entropy_payload:u64, pub file_footers:u64, pub index_data:u64,
    pub checksums:u64, pub archive_footer:u64,
}
impl ArchiveCostReport {
    pub fn accounted_bytes(self)->io::Result<u64>{
        [self.archive_header,self.file_metadata,self.block_headers,self.descriptors,self.entropy_payload,self.file_footers,self.index_data,self.checksums,self.archive_footer].into_iter().try_fold(0u64,|a,b|a.checked_add(b).ok_or_else(||invalid("archive cost overflow")))
    }
    pub fn overhead_bytes(self)->io::Result<i128>{Ok(self.archive_bytes as i128-self.source_bytes as i128)}
}

/// Random-access index for v3/v4 archives. The index is validated in streaming
/// passes; block offsets are not retained in memory. Path lookup is a bounded-
/// memory linear scan over compact per-entry metadata.
pub struct MultiArchiveIndex<R: Read+Seek>{ input:R, entries:Vec<ParsedIndexEntry>, index_offset:u64, index_len:u64, cost:ArchiveCostReport }
impl<R:Read+Seek> MultiArchiveIndex<R>{
    pub fn open(mut input:R)->io::Result<Self>{
        let file_len=input.seek(SeekFrom::End(0))?;
        if file_len<56{return Err(invalid("archive is too short for indexed format"));}
        input.seek(SeekFrom::Start(0))?; let mut header=[0u8;16]; input.read_exact(&mut header)?;
        let version=u16::from_le_bytes(header[8..10].try_into().unwrap());
        if &header[..8]!=MAGIC||(version!=VERSION&&version!=V4_VERSION&&version!=V3_VERSION)||header[10..12]!=[0,0]||crc32c::checksum(&header[..12])!=u32::from_le_bytes(header[12..16].try_into().unwrap()){return Err(invalid("not a valid PYTHIA indexed multi-file v3/v4/v5 archive"));}
        input.seek(SeekFrom::End(-40))?; let mut footer=[0u8;40]; input.read_exact(&mut footer)?;
        if &footer[..4]!=END_MAGIC||crc32c::checksum(&footer[..36])!=u32::from_le_bytes(footer[36..40].try_into().unwrap()){return Err(invalid("indexed archive footer CRC mismatch"));}
        let files=u64::from_le_bytes(footer[4..12].try_into().unwrap());
        let declared_total=u64::from_le_bytes(footer[12..20].try_into().unwrap());
        let index_offset=u64::from_le_bytes(footer[20..28].try_into().unwrap());
        let index_len=u64::from_le_bytes(footer[28..36].try_into().unwrap());
        if files>MAX_INDEX_ENTRIES||index_len<24||index_len>MAX_INDEX_BYTES||index_offset<16||index_offset.checked_add(index_len)!=Some(file_len-40){return Err(invalid("indexed archive bounds are invalid"));}

        // Pass 1: verify index CRC with a fixed 64 KiB buffer, never allocating index_len.
        input.seek(SeekFrom::Start(index_offset))?;
        let mut crc=crc32c::Crc32c::new();
        let mut remaining=index_len-4;
        let mut buf=[0u8;64*1024];
        while remaining>0 {
            let n=remaining.min(buf.len() as u64) as usize;
            input.read_exact(&mut buf[..n])?;
            crc.update(&buf[..n]);
            remaining-=n as u64;
        }
        let mut stored_crc=[0u8;4]; input.read_exact(&mut stored_crc)?;
        if crc.finalize()!=u32::from_le_bytes(stored_crc){return Err(invalid("index CRC32C mismatch"));}

        // Pass 2: parse sequentially, validate each block, and retain only file metadata.
        input.seek(SeekFrom::Start(index_offset))?;
        let mut index_crc=crc32c::Crc32c::new();
        let magic=read_index_bytes(&mut input,&mut index_crc,4)?;
        if magic.as_slice()!=INDEX_MAGIC{return Err(invalid("invalid index magic"));}
        let declared_len=read_index_u64(&mut input,&mut index_crc)?;
        let count=read_index_u64(&mut input,&mut index_crc)?;
        if declared_len!=index_len||count!=files||count>MAX_INDEX_ENTRIES||count>(index_len-24)/32{return Err(invalid("index header does not match archive footer"));}

        let mut entries=Vec::with_capacity(count as usize);
        let mut kinds=std::collections::HashMap::<String,bool>::with_capacity(count as usize);
        let mut last_end=16u64; let mut expected_seq=0u64; let mut sum_raw=0u64;
        let mut file_metadata=0u64; let mut block_headers=0u64; let mut descriptors=0u64; let mut entropy_payload=0u64; let mut file_footers=0u64;
        for file_id in 0..count {
            let plen=read_index_u16(&mut input,&mut index_crc)? as usize;
            let flags=read_index_u16(&mut input,&mut index_crc)?;
            let size=read_index_u64(&mut input,&mut index_crc)?;
            let entry_offset=read_index_u64(&mut input,&mut index_crc)?;
            let end_offset=read_index_u64(&mut input,&mut index_crc)?;
            let blocks=read_index_u32(&mut input,&mut index_crc)?;
            if plen==0||plen>MAX_PATH||flags>1{return Err(invalid("invalid index entry metadata"));}
            if flags==1&&(size!=0||blocks!=0){return Err(invalid("directory index entry contains data"));}
            if (flags==0&&size==0&&blocks!=0)||(flags==0&&size>0&&blocks==0){return Err(invalid("inconsistent index block count"));}
            let path_bytes=read_index_bytes(&mut input,&mut index_crc,plen)?;
            let path=String::from_utf8(path_bytes).map_err(|_|invalid("indexed path is not UTF-8"))?;
            validate_path(&path)?;
            if kinds.insert(path.clone(),flags==1).is_some(){return Err(invalid("duplicate path in archive index"));}
            file_metadata=file_metadata.checked_add(20+plen as u64).ok_or_else(||invalid("cost metadata overflow"))?;
            file_footers=file_footers.checked_add(32).ok_or_else(||invalid("cost footer overflow"))?;
            sum_raw=sum_raw.checked_add(size).ok_or_else(||invalid("indexed total size overflow"))?;
            if entry_offset!=last_end||entry_offset>=end_offset||end_offset>=index_offset{return Err(invalid("index entry offsets overlap or leave unindexed data"));}
            let offsets_offset=input.seek(SeekFrom::Current(0))?;
            let index_data_end=index_offset.checked_add(index_len-4).ok_or_else(||invalid("index end overflow"))?;
            let offset_bytes=(blocks as u64).checked_mul(8).ok_or_else(||invalid("index offset table overflow"))?;
            if offsets_offset.checked_add(offset_bytes).map_or(true,|v|v>index_data_end){return Err(invalid("index block-offset table is truncated"));}

            input.seek(SeekFrom::Start(entry_offset))?;
            let mut fixed=[0u8;16]; input.read_exact(&mut fixed)?;
            if &fixed[..4]!=FILE_MAGIC||u16::from_le_bytes(fixed[4..6].try_into().unwrap()) as usize!=plen||u16::from_le_bytes(fixed[6..8].try_into().unwrap())!=flags||u64::from_le_bytes(fixed[8..16].try_into().unwrap())!=size{return Err(invalid("indexed entry header does not match index"));}
            let mut archive_path=vec![0u8;plen]; input.read_exact(&mut archive_path)?;
            if archive_path!=path.as_bytes(){return Err(invalid("indexed entry path mismatch"));}
            let path_crc=get_u32(&mut input)?; let mut entry_crc_data=fixed.to_vec(); entry_crc_data.extend_from_slice(&archive_path);
            if crc32c::checksum(&entry_crc_data)!=path_crc{return Err(invalid("indexed entry header CRC mismatch"));}
            let mut expected_offset=input.seek(SeekFrom::Current(0))?; let mut raw_sum=0u64;
            // Consume offsets from the index stream one at a time. Each is checked
            // against its block, then the reader returns to the next index scalar.
            input.seek(SeekFrom::Start(offsets_offset))?;
            let first_sequence=expected_seq;
            for _ in 0..blocks {
                let index_cursor_before=input.seek(SeekFrom::Current(0))?;
                if index_cursor_before.checked_add(8).map_or(true,|v|v>index_data_end){return Err(invalid("index block-offset table is truncated"));}
                let off=read_index_u64(&mut input,&mut index_crc)?;
                if off!=expected_offset||off>=end_offset{return Err(invalid("indexed block offsets do not exactly cover file data"));}
                let index_cursor=input.seek(SeekFrom::Current(0))?;
                input.seek(SeekFrom::Start(off))?;
                let mut hb=[0u8;BLOCK_HEADER_LEN]; input.read_exact(&mut hb)?;
                let h=BlockHeader::decode(&hb,MAX_BLOCK as u32)?;
                if h.sequence!=expected_seq{return Err(invalid("indexed block sequence is not contiguous"));}
                expected_seq=expected_seq.checked_add(1).ok_or_else(||invalid("block sequence overflow"))?;
                raw_sum=raw_sum.checked_add(h.raw_len as u64).ok_or_else(||invalid("indexed file size overflow"))?;
                block_headers=block_headers.checked_add(28).ok_or_else(||invalid("cost block header overflow"))?;
                descriptors=descriptors.checked_add(h.descriptor_len as u64).ok_or_else(||invalid("cost descriptor overflow"))?;
                entropy_payload=entropy_payload.checked_add(h.frame_len as u64).ok_or_else(||invalid("cost payload overflow"))?;
                expected_offset=off.checked_add(BLOCK_HEADER_LEN as u64).and_then(|v|v.checked_add(h.descriptor_len as u64)).and_then(|v|v.checked_add(h.frame_len as u64)).ok_or_else(||invalid("indexed block extent overflow"))?;
                if expected_offset>end_offset{return Err(invalid("indexed block payload exceeds file extent"));}
                input.seek(SeekFrom::Start(index_cursor))?;
            }
            if expected_offset!=end_offset||raw_sum!=size{return Err(invalid("indexed file block extents do not match file metadata"));}
            // Continue the index stream after the offset array.
            let after_offsets=input.seek(SeekFrom::Current(0))?;
            input.seek(SeekFrom::Start(end_offset))?;
            let mut file_footer=[0u8;32]; input.read_exact(&mut file_footer)?;
            if &file_footer[..4]!=FILE_END_MAGIC||u64::from_le_bytes(file_footer[4..12].try_into().unwrap())!=file_id||u64::from_le_bytes(file_footer[12..20].try_into().unwrap())!=size||u64::from_le_bytes(file_footer[20..28].try_into().unwrap())!=blocks as u64||crc32c::checksum(&file_footer[..28])!=u32::from_le_bytes(file_footer[28..32].try_into().unwrap()){return Err(invalid("indexed per-file footer mismatch"));}
            last_end=end_offset.checked_add(32).ok_or_else(||invalid("file extent overflow"))?;
            entries.push(ParsedIndexEntry{info:IndexedFileInfo{path,size,is_directory:flags==1,block_count:blocks},entry_offset,end_offset,offsets_offset,first_sequence});
            // The index cursor is exactly after this entry's offset array.
            input.seek(SeekFrom::Start(after_offsets))?;
        }
        if last_end!=index_offset{return Err(invalid("archive data area does not end at index offset"));}
        if sum_raw!=declared_total{return Err(invalid("indexed total size does not match footer"));}
        if input.seek(SeekFrom::Current(0))?!=index_offset+index_len-4{return Err(invalid("index contains trailing or missing bytes"));}
        if index_crc.finalize()!=u32::from_le_bytes(stored_crc){return Err(invalid("index CRC changed during parse"));}
        validate_index_hierarchy(kinds.iter().map(|(p,d)|(p.as_str(),*d)))?;

        let index_data=index_len-4;
        let checksums=12u64.checked_add(files.checked_mul(8).ok_or_else(||invalid("checksum accounting overflow"))?).and_then(|v|v.checked_add(expected_seq.checked_mul(4)?)).ok_or_else(||invalid("checksum accounting overflow"))?;
        let cost=ArchiveCostReport{source_bytes:declared_total,archive_bytes:file_len,archive_header:12,file_metadata:file_metadata.saturating_sub(files*4),block_headers,descriptors,entropy_payload,file_footers:file_footers.saturating_sub(files*4),index_data,checksums,archive_footer:36};
        if cost.accounted_bytes()?!=file_len{return Err(invalid("archive cost components do not equal physical archive length"));}
        Ok(Self{input,entries,index_offset,index_len,cost})
    }

    pub fn entries(&self)->impl Iterator<Item=&IndexedFileInfo>{self.entries.iter().map(|e|&e.info)}
    pub fn index_offset(&self)->u64{self.index_offset}
    pub fn cost_report(&self)->ArchiveCostReport{self.cost}

    fn find_entry(&self,path:&str)->io::Result<usize>{
        self.entries.iter().position(|e|e.info.path==path).ok_or_else(||io::Error::new(io::ErrorKind::NotFound,"file not found in archive index"))
    }
    fn read_block_offset(&mut self, entry:&ParsedIndexEntry, block_index:u32)->io::Result<u64>{
        if block_index>=entry.info.block_count{return Err(invalid("block index outside indexed entry"));}
        let pos=entry.offsets_offset.checked_add((block_index as u64).checked_mul(8).ok_or_else(||invalid("block offset position overflow"))?).ok_or_else(||invalid("block offset position overflow"))?;
        let end=pos.checked_add(8).ok_or_else(||invalid("block offset position overflow"))?;
        if end>self.index_offset+self.index_len-4{return Err(invalid("block offset read exceeds index bounds"));}
        self.input.seek(SeekFrom::Start(pos))?; let mut raw=[0u8;8]; self.input.read_exact(&mut raw)?; Ok(u64::from_le_bytes(raw))
    }
    pub fn extract_file_to<W:Write>(&mut self,path:&str,mut output:W)->io::Result<u64>{
        let idx=self.find_entry(path)?; let e=self.entries[idx].clone();
        if e.info.is_directory{return Err(io::Error::new(io::ErrorKind::InvalidInput,"requested archive entry is a directory"));}
        self.input.seek(SeekFrom::Start(e.entry_offset))?; let mut fixed=[0u8;16]; self.input.read_exact(&mut fixed)?;
        if &fixed[..4]!=FILE_MAGIC||u16::from_le_bytes(fixed[4..6].try_into().unwrap()) as usize!=e.info.path.len()||u16::from_le_bytes(fixed[6..8].try_into().unwrap())!=0||u64::from_le_bytes(fixed[8..16].try_into().unwrap())!=e.info.size{return Err(invalid("indexed file header does not match index"));}
        let mut name=vec![0u8;e.info.path.len()]; self.input.read_exact(&mut name)?; if name!=e.info.path.as_bytes(){return Err(invalid("indexed path does not match file header"));}
        let got=get_u32(&mut self.input)?; let mut crc_data=fixed.to_vec();crc_data.extend_from_slice(&name);if crc32c::checksum(&crc_data)!=got{return Err(invalid("indexed file header CRC mismatch"));}
        let mut total=0u64;
        for block_index in 0..e.info.block_count {
            let off=self.read_block_offset(&e,block_index)?; self.input.seek(SeekFrom::Start(off))?;
            let mut hb=[0u8;BLOCK_HEADER_LEN];self.input.read_exact(&mut hb)?;let h=BlockHeader::decode(&hb,MAX_BLOCK as u32)?;
            let expected=e.first_sequence.checked_add(block_index as u64).ok_or_else(||invalid("block sequence overflow"))?;
            if h.sequence!=expected{return Err(invalid("indexed block sequence mismatch"));}
            let mut descriptor=vec![0u8;h.descriptor_len as usize];self.input.read_exact(&mut descriptor)?;let mut frame=vec![0u8;h.frame_len as usize];self.input.read_exact(&mut frame)?;
            if self.input.seek(SeekFrom::Current(0))? > e.end_offset{return Err(invalid("indexed block payload crosses file extent"));}
            let raw=decode_indexed_payload(&h,&descriptor,&frame)?;
            total=total.checked_add(raw.len() as u64).ok_or_else(||invalid("file output size overflow"))?; if total>e.info.size{return Err(invalid("indexed blocks exceed declared file size"));} output.write_all(&raw)?;
        }
        if total!=e.info.size{return Err(invalid("indexed blocks do not match declared file size"));}
        self.input.seek(SeekFrom::Start(e.end_offset))?;let mut end=[0u8;32];self.input.read_exact(&mut end)?;
        if &end[..4]!=FILE_END_MAGIC||u64::from_le_bytes(end[4..12].try_into().unwrap())!=idx as u64||u64::from_le_bytes(end[12..20].try_into().unwrap())!=total||u64::from_le_bytes(end[20..28].try_into().unwrap())!=e.info.block_count as u64||crc32c::checksum(&end[..28])!=u32::from_le_bytes(end[28..32].try_into().unwrap()){return Err(invalid("indexed file footer mismatch"));}
        Ok(total)
    }

    pub fn extract_file_range_to<W:Write>(&mut self,path:&str,start:u64,length:u64,mut output:W)->io::Result<u64>{
        let idx=self.find_entry(path)?;let e=self.entries[idx].clone();
        if e.info.is_directory{return Err(io::Error::new(io::ErrorKind::InvalidInput,"requested archive entry is a directory"));}
        if start>e.info.size{return Err(io::Error::new(io::ErrorKind::InvalidInput,"range start exceeds file size"));}
        let end=start.checked_add(length).ok_or_else(||invalid("requested range overflow"))?.min(e.info.size);if start==end{return Ok(0);}
        let mut raw_position=0u64;let mut written=0u64;
        for block_index in 0..e.info.block_count {
            let off=self.read_block_offset(&e,block_index)?;self.input.seek(SeekFrom::Start(off))?;
            let mut hb=[0u8;BLOCK_HEADER_LEN];self.input.read_exact(&mut hb)?;let h=BlockHeader::decode(&hb,MAX_BLOCK as u32)?;
            let expected=e.first_sequence.checked_add(block_index as u64).ok_or_else(||invalid("block sequence overflow"))?;
            if h.sequence!=expected{return Err(invalid("indexed block sequence mismatch"));}
            let block_end=raw_position.checked_add(h.raw_len as u64).ok_or_else(||invalid("file range offset overflow"))?;
            let payload_end=off.checked_add(BLOCK_HEADER_LEN as u64).and_then(|v|v.checked_add(h.descriptor_len as u64)).and_then(|v|v.checked_add(h.frame_len as u64)).ok_or_else(||invalid("indexed block extent overflow"))?;
            if payload_end>e.end_offset{return Err(invalid("indexed block payload crosses file extent"));}
            if block_end>start&&raw_position<end {
                let mut descriptor=vec![0u8;h.descriptor_len as usize];self.input.read_exact(&mut descriptor)?;let mut frame=vec![0u8;h.frame_len as usize];self.input.read_exact(&mut frame)?;
                if self.input.seek(SeekFrom::Current(0))? > e.end_offset{return Err(invalid("indexed block payload crosses file extent"));}
                let raw=decode_indexed_payload(&h,&descriptor,&frame)?;
                let local_start=start.saturating_sub(raw_position) as usize;let local_end=(end.min(block_end)-raw_position) as usize;
                if local_start>local_end||local_end>raw.len(){return Err(invalid("computed file range is outside decoded block"));}
                output.write_all(&raw[local_start..local_end])?;written=written.checked_add((local_end-local_start) as u64).ok_or_else(||invalid("range output length overflow"))?;
            }
            raw_position=block_end;if raw_position>=end{break;}
        }
        if raw_position<end||written!=end-start{return Err(invalid("indexed blocks do not cover requested byte range"));}Ok(written)
    }
}

fn decode_indexed_payload(h:&BlockHeader,descriptor:&[u8],frame:&[u8])->io::Result<Vec<u8>>{
    let transformed=decode_best(frame,max_transformed_block_len(h.raw_len)).map_err(|x|invalid_owned(format!("entropy frame rejected: {x}")))?;
    let raw=if descriptor.is_empty(){transformed}else{
        let d=decode_block_descriptor(descriptor).map_err(|x|invalid_owned(format!("transform descriptor rejected: {x}")))?;
        if d.input_domain!=Domain::Bytes||d.output_domain!=Domain::Bytes||d.stages.is_empty(){return Err(invalid("unsupported transform descriptor"));}
        let n=d.stages.len();let chain=ModelChain::new(d.stages);
        chain.validate_domains(Domain::Bytes,Domain::Bytes).map_err(|x|invalid_owned(format!("transform chain rejected: {x}")))?;
        chain.inverse(&ChainResult{data:transformed,auxiliary:vec![Vec::new();n]}).map_err(|x|invalid_owned(format!("inverse transform failed: {x}")))?
    };
    if raw.len()!=h.raw_len as usize||crc32c::checksum(&raw)!=h.raw_crc32c{return Err(invalid("indexed block checksum or length mismatch"));}Ok(raw)
}


#[cfg(test)]
mod ps78_spool_tests {
    use super::IndexSpool;
    use std::io;

    #[test]
    fn disk_spool_preserves_offset_order_and_removes_temp_file() -> io::Result<()> {
        let path;
        {
            let mut spool = IndexSpool::create()?;
            path = spool.path.clone();
            for value in [17u64, 0, u64::MAX, 4096, 8192] { spool.push(value)?; }
            let mut actual = Vec::new();
            spool.read_offsets(0, 5, |v| { actual.push(v); Ok(()) })?;
            assert_eq!(actual, vec![17, 0, u64::MAX, 4096, 8192]);
            assert!(spool.read_offsets(8, 5, |_| Ok(())).is_err());
            assert!(path.exists());
        }
        assert!(!path.exists());
        Ok(())
    }
}


#[cfg(test)]
mod ps79_lazy_index_tests {
    use super::{MultiArchiveIndex, MultiArchiveWriter};
    use std::io::Cursor;

    #[test]
    fn lazy_index_reads_offsets_on_demand_for_full_and_range_extract() {
        let first: Vec<u8> = (0..4097).map(|i| (i * 31) as u8).collect();
        let second: Vec<u8> = (0..8193).map(|i| (i * 17 + 9) as u8).collect();
        let mut writer = MultiArchiveWriter::new(Cursor::new(Vec::new()), 512).unwrap();
        writer.add_file("a/first.bin", first.len() as u64, Cursor::new(first.clone())).unwrap();
        writer.add_file("b/second.bin", second.len() as u64, Cursor::new(second.clone())).unwrap();
        let archive = writer.finish().unwrap().into_inner();

        let mut index = MultiArchiveIndex::open(Cursor::new(archive)).unwrap();
        let names: Vec<String> = index.entries().map(|e| e.path.clone()).collect();
        assert_eq!(names, vec!["a/first.bin", "b/second.bin"]);
        assert!(index.entries.iter().all(|e| e.info.block_count > 0));
        let mut restored = Vec::new();
        assert_eq!(index.extract_file_to("b/second.bin", &mut restored).unwrap(), second.len() as u64);
        assert_eq!(restored, second);

        let start = 701u64;
        let length = 1800u64;
        let mut range = Vec::new();
        assert_eq!(index.extract_file_range_to("a/first.bin", start, length, &mut range).unwrap(), length);
        assert_eq!(range.as_slice(), &first[start as usize..(start + length) as usize]);
    }

    #[test]
    fn lazy_index_rejects_corrupt_index_crc_without_loading_index_buffer() {
        let data = vec![42u8; 2048];
        let mut writer = MultiArchiveWriter::new(Cursor::new(Vec::new()), 256).unwrap();
        writer.add_file("payload.bin", data.len() as u64, Cursor::new(data)).unwrap();
        let mut archive = writer.finish().unwrap().into_inner();
        let footer = archive.len() - 40;
        let index_offset = u64::from_le_bytes(archive[footer + 20..footer + 28].try_into().unwrap()) as usize;
        archive[index_offset + 20] ^= 0x40;
        assert!(MultiArchiveIndex::open(Cursor::new(archive)).is_err());
    }
}


#[cfg(test)]
mod ps710_hardening_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn v5_writer_and_v4_v3_reader_compatibility() {
        let data = b"xyz".to_vec();
        let mut writer = MultiArchiveWriter::new(Cursor::new(Vec::new()), 512).unwrap();
        writer.add_file("compat.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
        let archive = writer.finish().unwrap().into_inner();
        assert_eq!(u16::from_le_bytes([archive[8], archive[9]]), VERSION);

        // The indexed payload layout is unchanged between v3, v4 and v5;
        // update only the version field and header CRC to test old versions.
        for legacy_version in [V4_VERSION, V3_VERSION] {
            let mut legacy_archive = archive.clone();
            legacy_archive[8..10].copy_from_slice(&legacy_version.to_le_bytes());
            let crc = crc32c::checksum(&legacy_archive[..12]);
            legacy_archive[12..16].copy_from_slice(&crc.to_le_bytes());

            let mut reader = MultiArchiveReader::new(Cursor::new(legacy_archive.clone())).unwrap();
            let entry = reader.next_entry().unwrap().unwrap();
            assert_eq!(entry.path, "compat.bin");
            let mut restored = Vec::new();
            reader.copy_current_to(&mut restored).unwrap();
            assert_eq!(restored, data);
            assert!(reader.next_entry().unwrap().is_none());
            reader.finish_check().unwrap();

            let index = MultiArchiveIndex::open(Cursor::new(legacy_archive)).unwrap();
            assert_eq!(index.entries().next().unwrap().path, "compat.bin");
        }
    }

    #[test]
    fn sequential_reader_rejects_indexed_archive_with_legacy_footer() {
        let data: Vec<u8> = (0..3000).map(|i| (i * 7) as u8).collect();
        let mut writer = MultiArchiveWriter::new(Cursor::new(Vec::new()), 1024).unwrap();
        writer.add_file("a.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
        let archive = writer.finish().unwrap().into_inner();

        let footer_at = archive.len() - 40;
        let files = u64::from_le_bytes(archive[footer_at + 4..footer_at + 12].try_into().unwrap());
        let total = u64::from_le_bytes(archive[footer_at + 12..footer_at + 20].try_into().unwrap());
        let index_offset = u64::from_le_bytes(archive[footer_at + 20..footer_at + 28].try_into().unwrap()) as usize;

        // Drop the index and append a legacy 24-byte footer: an indexed archive must
        // not be downgradable into an index-less archive.
        let mut forged = archive[..index_offset].to_vec();
        let mut legacy = [0u8; 24];
        legacy[..4].copy_from_slice(END_MAGIC);
        legacy[4..12].copy_from_slice(&files.to_le_bytes());
        legacy[12..20].copy_from_slice(&total.to_le_bytes());
        let crc = crc32c::checksum(&legacy[..20]);
        legacy[20..24].copy_from_slice(&crc.to_le_bytes());
        forged.extend_from_slice(&legacy);

        let mut reader = MultiArchiveReader::new(Cursor::new(forged)).unwrap();
        loop {
            match reader.next_entry() {
                Ok(Some(_)) => { reader.copy_current_to(io::sink()).unwrap(); }
                Ok(None) => panic!("a v3/v4 archive without its index must be rejected"),
                Err(_) => break,
            }
        }
    }
}


#[cfg(test)]
mod ps97_parameterized_descriptor_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn multiplicative_recurrence_roundtrips_through_both_multi_file_readers() {
        let mut x = 1u8;
        let data: Vec<u8> = (0..8192).map(|_| { let v = x; x = x.wrapping_mul(3); v }).collect();
        let mut writer = MultiArchiveWriter::new(Cursor::new(Vec::new()), 4096).unwrap();
        writer.add_file("geo.bin", data.len() as u64, Cursor::new(data.clone())).unwrap();
        let archive = writer.finish().unwrap().into_inner();

        let mut sequential = MultiArchiveReader::new(Cursor::new(archive.clone())).unwrap();
        assert!(sequential.next_entry().unwrap().is_some());
        let mut restored = Vec::new();
        sequential.copy_current_to(&mut restored).unwrap();
        assert_eq!(restored, data);

        let mut index = MultiArchiveIndex::open(Cursor::new(archive)).unwrap();
        let mut restored = Vec::new();
        index.extract_file_to("geo.bin", &mut restored).unwrap();
        assert_eq!(restored, data);
    }
}
