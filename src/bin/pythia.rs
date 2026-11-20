use std::{env, fs::{self, File, OpenOptions}, io::{self, Read, Write}, path::{Path, PathBuf}};
use pythia_core::container::{ArchiveReader, ArchiveWriter, MultiArchiveIndex, MultiArchiveReader, MultiArchiveWriter};

fn usage() {
    eprintln!("PYTHIA research CLI\n  pythia create <input> <archive> [block_size]\n  pythia extract <archive> <output>\n  pythia list <archive>\n  pythia test <archive>\n  pythia create-dir <input_directory> <archive> [block_size] [workers] [adaptive] [raw_memory_budget_bytes]\n  pythia extract-dir <archive> <output_directory>\n  pythia list-dir <archive>\n  pythia test-dir <archive>\n  pythia list-index <archive>\n  pythia extract-file <archive> <internal_path> <output>\n  pythia extract-range <archive> <internal_path> <start> <length> <output>\n  pythia archive-cost <archive>");
}
fn invalid(s: &'static str) -> io::Error { io::Error::new(io::ErrorKind::InvalidInput, s) }

/// Creates `path` exclusively (an existing file is never overwritten or truncated),
/// runs `f`, and removes the new file again if `f` fails, so an interrupted run
/// never leaves a truncated archive or output behind.
fn with_new_file<T>(path: &Path, f: impl FnOnce(File) -> io::Result<T>) -> io::Result<T> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    match f(file) {
        Ok(value) => Ok(value),
        Err(error) => { let _ = fs::remove_file(path); Err(error) }
    }
}

fn collect_tree(root: &Path, current: &Path, out: &mut Vec<(PathBuf, bool, u64)>) -> io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(current)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_symlink() { return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("symbolic links are not archived: {}", path.display()))); }
        let rel = path.strip_prefix(root).map_err(|_| invalid("tree path escaped root"))?.to_path_buf();
        if ty.is_dir() { out.push((rel.clone(), true, 0)); collect_tree(root, &path, out)?; }
        else if ty.is_file() { out.push((rel, false, entry.metadata()?.len())); }
        else { return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unsupported filesystem entry: {}", path.display()))); }
    }
    Ok(())
}
fn rel_string(path: &Path) -> io::Result<String> {
    let mut parts=Vec::new();
    for c in path.components() { match c { std::path::Component::Normal(s)=>parts.push(s.to_str().ok_or_else(||invalid("non-UTF-8 paths are not supported"))?), _=>return Err(invalid("unsafe relative path")) } }
    Ok(parts.join("/"))
}
fn create_dir_archive(source:&Path,archive:&Path,block:usize,workers:usize,adaptive:bool,memory_budget:Option<usize>)->io::Result<()> {
    if !source.is_dir(){return Err(invalid("input must be a directory"));}
    let root=source.canonicalize()?;
    let archive_abs=if archive.exists(){archive.canonicalize()?}else{let parent=archive.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or_else(||Path::new(".")).canonicalize()?;parent.join(archive.file_name().ok_or_else(||invalid("archive path has no filename"))?)};
    if archive_abs.starts_with(&root){return Err(invalid("archive destination must be outside the source directory"));}
    let mut items=Vec::new();collect_tree(&root,&root,&mut items)?;
    with_new_file(archive,|output|{
        let mut writer=MultiArchiveWriter::new(output,block)?.with_parallelism(workers)?.with_adaptive_blocks(adaptive);if let Some(bytes)=memory_budget{writer=writer.with_raw_memory_budget(bytes)?;}
        for (rel,is_dir,size) in items {let name=rel_string(&rel)?;if is_dir{writer.add_directory(&name)?;}else{let file=File::open(root.join(&rel))?;writer.add_file(&name,size,file)?;}}
        writer.finish()?;Ok(())
    })
}
fn ensure_safe_dirs(root:&Path, parts:&[&str])->io::Result<PathBuf>{
    let mut current=root.to_path_buf();
    for part in parts {
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(meta) => { if meta.file_type().is_symlink() || !meta.is_dir() { return Err(io::Error::new(io::ErrorKind::PermissionDenied,"extraction path contains a symlink or non-directory")); } },
            Err(e) if e.kind()==io::ErrorKind::NotFound => fs::create_dir(&current)?,
            Err(e)=>return Err(e),
        }
    }
    Ok(current)
}
fn extract_dir_archive(archive:&Path,dest:&Path)->io::Result<()> {
    fs::create_dir_all(dest)?;let root=dest.canonicalize()?;let input=File::open(archive)?;let mut reader=MultiArchiveReader::new(input)?;
    while let Some(entry)=reader.next_entry()? {
        pythia_core::container::multi::validate_path(&entry.path)?;
        let parts:Vec<&str>=entry.path.split('/').collect();
        if entry.is_directory {ensure_safe_dirs(&root,&parts)?;reader.copy_current_to(io::sink())?;}
        else {let parent=ensure_safe_dirs(&root,&parts[..parts.len()-1])?;let target=parent.join(parts[parts.len()-1]);let mut file=OpenOptions::new().write(true).create_new(true).open(&target)?;reader.copy_current_to(&mut file)?;file.flush()?;}
    }
    reader.finish_check()?;Ok(())
}
fn run() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len()<3{usage();return Err(invalid("missing command or path"));}
    match args[1].as_str(){
        "create"=>{if args.len()<4||args.len()>5{usage();return Err(invalid("create expects input, archive and optional block size"));}let bs=if args.len()==5{args[4].parse::<u32>().map_err(|_|invalid("invalid block size"))?}else{1<<20};let mut input=File::open(&args[2])?;with_new_file(Path::new(&args[3]),|output|{let mut w=ArchiveWriter::new(output,bs)?;io::copy(&mut input,&mut w)?;w.finish()?;Ok(())})?;}
        "extract"=>{if args.len()!=4{return Err(invalid("extract expects archive and output"));}let input=File::open(&args[2])?;let mut r=ArchiveReader::new(input)?;with_new_file(Path::new(&args[3]),|mut output|{io::copy(&mut r,&mut output)?;output.flush()?;Ok(())})?;}
        "list"|"test"=>{if args.len()!=3{return Err(invalid("list/test expects archive"));}let input=File::open(&args[2])?;let mut r=ArchiveReader::new(input)?;let mut raw=0u64;while let Some((info,_))=r.next_block()?{raw+=info.uncompressed_len as u64;if args[1]=="list"{println!("block={} raw={} frame={} stages={} descriptor={}",info.sequence,info.uncompressed_len,info.encoded_len,info.transform_stages,info.descriptor_len);}}if args[1]=="test"{println!("OK: {} blocks, {} bytes",r.blocks_read(),raw);}}
        "create-dir"=>{if args.len()<4||args.len()>8{return Err(invalid("create-dir expects directory, archive, optional block size, workers, adaptive flag and raw memory budget"));}let bs=if args.len()>=5{args[4].parse::<usize>().map_err(|_|invalid("invalid block size"))?}else{1<<20};let workers=if args.len()>=6{args[5].parse::<usize>().map_err(|_|invalid("invalid worker count"))?}else{1};let adaptive=if args.len()>=7{match args[6].as_str(){"true"|"1"=>true,"false"|"0"=>false,_=>return Err(invalid("adaptive must be true/false or 1/0"))}}else{false};let memory_budget=if args.len()==8{Some(args[7].parse::<usize>().map_err(|_|invalid("invalid raw memory budget"))?)}else{None};create_dir_archive(Path::new(&args[2]),Path::new(&args[3]),bs,workers,adaptive,memory_budget)?;}
        "list-index"=>{if args.len()!=3{return Err(invalid("list-index expects archive"));}let index=MultiArchiveIndex::open(File::open(&args[2])?)?;for e in index.entries(){println!("{}\t{}\t{}\t{}",if e.is_directory{"DIR"}else{"FILE"},e.size,e.block_count,e.path);}}
        "archive-cost"=>{if args.len()!=3{return Err(invalid("archive-cost expects archive"));}let index=MultiArchiveIndex::open(File::open(&args[2])?)?;let c=index.cost_report();println!("source_bytes={}\narchive_bytes={}\noverhead_bytes={}\narchive_header={}\nfile_metadata={}\nblock_headers={}\ndescriptors={}\nentropy_payload={}\nfile_footers={}\nindex_data={}\nchecksums={}\narchive_footer={}",c.source_bytes,c.archive_bytes,c.overhead_bytes()?,c.archive_header,c.file_metadata,c.block_headers,c.descriptors,c.entropy_payload,c.file_footers,c.index_data,c.checksums,c.archive_footer);}
        "extract-file"=>{if args.len()!=5{return Err(invalid("extract-file expects archive, internal path and output"));}let mut index=MultiArchiveIndex::open(File::open(&args[2])?)?;with_new_file(Path::new(&args[4]),|mut out|{index.extract_file_to(&args[3],&mut out)?;out.flush()?;Ok(())})?;}
        "extract-range"=>{if args.len()!=7{return Err(invalid("extract-range expects archive, internal path, start, length and output"));}let start=args[4].parse::<u64>().map_err(|_|invalid("invalid range start"))?;let length=args[5].parse::<u64>().map_err(|_|invalid("invalid range length"))?;let mut index=MultiArchiveIndex::open(File::open(&args[2])?)?;let n=with_new_file(Path::new(&args[6]),|mut out|{let n=index.extract_file_range_to(&args[3],start,length,&mut out)?;out.flush()?;Ok(n)})?;println!("OK: extracted {n} bytes");}
        "extract-dir"=>{if args.len()!=4{return Err(invalid("extract-dir expects archive and destination directory"));}extract_dir_archive(Path::new(&args[2]),Path::new(&args[3]))?;}
        "list-dir"|"test-dir"=>{if args.len()!=3{return Err(invalid("list-dir/test-dir expects archive"));}let mut r=MultiArchiveReader::new(File::open(&args[2])?)?;while let Some(e)=r.next_entry()?{if args[1]=="list-dir"{println!("{}\t{}\t{}",if e.is_directory{"DIR"}else{"FILE"},e.size,e.path);}r.copy_current_to(io::sink())?;}r.finish_check()?;if args[1]=="test-dir"{println!("OK: {} entries, {} bytes",r.files_read(),r.total_uncompressed_bytes());}}
        _=>{usage();return Err(invalid("unknown command"));}
    }Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("pythia: {e}");std::process::exit(2);}}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        env::temp_dir().join(format!("pythia-cli-test-{}-{}", std::process::id(), name))
    }

    #[test]
    fn existing_output_is_never_overwritten() {
        let path = temp_path("existing");
        fs::write(&path, b"keep").unwrap();
        let error = with_new_file(&path, |_file| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&path).unwrap(), b"keep");
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn failed_run_removes_its_partial_output() {
        let path = temp_path("failed");
        let _ = fs::remove_file(&path);
        let result: io::Result<()> = with_new_file(&path, |mut file| {
            file.write_all(b"partial")?;
            Err(invalid("simulated failure"))
        });
        assert!(result.is_err());
        assert!(!path.exists());
    }
}
