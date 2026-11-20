//! PS6.8: bounded order-1..4 context rANS with longest-context backoff.
//! Tables are static, canonical and included in the frame; selector charges all bytes.
use crate::UmceError;

use std::collections::BTreeMap;

const SCALE_BITS:u32=12;
 const SCALE:u32=1<<SCALE_BITS;
 const RANS_L:u64=1<<23;

#[derive(Clone)] struct Table {
    freq:[u16;
    256],
    cum:[u32;
    257],
    decode:[u8;
    SCALE as usize],
    wire:Vec<u8>
}

impl Table {
    
     fn from_counts(c:&[u64;
    256])->Result<Self,
    UmceError> {
        let total:u64=c.iter().try_fold(0u64,
        |a,
        &v|a.checked_add(v).ok_or(UmceError::LengthOverflow))?;
        let n=c.iter().filter(|&&v|v>0).count();
        if n==0||n>SCALE as usize {
            return Err(UmceError::InvalidDescriptor("empty PS6.8 context"));
            
        }
        let rem=SCALE as u64-n as u64;
        let mut f=[0u16;
        256];
        let mut assigned=n as u64;
        let mut residues=Vec::new();
        for(s,
        &v)in c.iter().enumerate() {
            if v==0 {
                continue
            }
            let x=v.checked_mul(rem).ok_or(UmceError::LengthOverflow)?;
            let q=x/total;
            f[s]=(1+q)as u16;
            assigned+=q;
            residues.push((x%total,
            s));
            
        }
        residues.sort_by(|a,
        b|b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for&(_,
        s)in residues.iter().take((SCALE as u64-assigned)as usize) {
            f[s]+=1;
            
        }
        let mut cum=[0u32;
        257];
        for i in 0..256 {
            cum[i+1]=cum[i]+f[i]as u32;
            
        }
        if cum[256]!=SCALE {
            return Err(UmceError::InvalidDescriptor("PS6.8 normalization failure"));
            
        }
        let mut wire=Vec::new();
        let n=n as u16;
        wire.extend_from_slice(&n.to_le_bytes());
        for(s,
        &v)in f.iter().enumerate() {
            if v>0 {
                wire.push(s as u8);
                wire.extend_from_slice(&v.to_le_bytes());
                
            }
            
        }
        let mut decode=[0u8;
        SCALE as usize];
        let mut base=0usize;
        for symbol in 0..256 {
            let end=base+f[symbol] as usize;
            decode[base..end].fill(symbol as u8);
            base=end;
            
        }
        Ok(Self {
            freq:f,
            cum,
            decode,
            wire
        })
    }
    
     fn parse(b:&[u8],
    p:&mut usize)->Result<Self,
    UmceError> {
        if b.len().saturating_sub(*p)<2 {
            return Err(UmceError::InvalidDescriptor("truncated PS6.8 table"));
            
        }
        let n=u16::from_le_bytes([b[*p],
        b[*p+1]])as usize;
        *p+=2;
        if n==0||n>256||b.len().saturating_sub(*p)<n*3 {
            return Err(UmceError::InvalidDescriptor("invalid PS6.8 table length"));
            
        }
        let start=*p-2;
        let mut f=[0u16;
        256];
        let mut sum=0u32;
        let mut prev=None;
        for _ in 0..n {
            let s=b[*p];
            let v=u16::from_le_bytes([b[*p+1],
            b[*p+2]]);
            *p+=3;
            if v==0||prev.map_or(false,
            |x|s<=x) {
                return Err(UmceError::InvalidDescriptor("noncanonical PS6.8 frequencies"));
                
            }
            prev=Some(s);
            f[s as usize]=v;
            sum=sum.checked_add(v as u32).ok_or(UmceError::LengthOverflow)?;
            
        }
        if sum!=SCALE {
            return Err(UmceError::InvalidDescriptor("PS6.8 frequency sum mismatch"));
            
        }
        let mut cum=[0u32;
        257];
        for i in 0..256 {
            cum[i+1]=cum[i]+f[i]as u32;
            
        }
        let mut decode=[0u8;
        SCALE as usize];
        let mut base=0usize;
        for symbol in 0..256 {
            let end=base+f[symbol] as usize;
            decode[base..end].fill(symbol as u8);
            base=end;
            
        }
        Ok(Self {
            freq:f,
            cum,
            decode,
            wire:b[start..*p].to_vec()
        })
    }
    
     fn symbol(&self,
    x:u32)->Result<usize,
    UmceError> {
        if x>=SCALE {
            return Err(UmceError::InvalidDescriptor("PS6.8 slot outside table"));
            
        }
        Ok(self.decode[x as usize] as usize)
    }
    
    
}

fn context(input:&[u8],
pos:usize,
order:usize)->Vec<u8> {
    let n=pos.min(order);
    input[pos-n..pos].to_vec()
}

fn train(input:&[u8],
order:usize)->Result<BTreeMap<Vec<u8>,
Table>,
UmceError> {
    let mut counts:BTreeMap<Vec<u8>,
    [u64;
    256]>=BTreeMap::new();
    for(pos,
    &b)in input.iter().enumerate() {
        for o in 0..=order.min(pos) {
            let key=context(input,
            pos,
            o);
            if !counts.contains_key(&key) && counts.len()>=8_192 {
                return Err(UmceError::InvalidDescriptor("PS6.8 context budget exceeded"));
                
            }
            let row=counts.entry(key).or_insert([0;
            256]);
            row[b as usize]=row[b as usize].checked_add(1).ok_or(UmceError::LengthOverflow)?;
            
        }
        
    }
    let mut tables=BTreeMap::new();
    for(k,
    v)in counts {
        tables.insert(k,
        Table::from_counts(&v)?);
        
    }
    Ok(tables)
}


/// Integer log2 approximation in Q20. The normalized mantissa is squared
/// iteratively in Q32, avoiding floating point and keeping model selection
/// deterministic across platforms.
fn log2_q20(value:u32)->u64 {
    debug_assert!(value>0);
    let integer=31-value.leading_zeros();
    let mut x=((value as u128)<<32)>>integer;
    let mut fraction=0u64;
    for bit in 0..20 {
        x=(x*x)>>32;
        if x >= (2u128<<32) {
            x >>= 1;
            fraction |= 1u64 << (19-bit);
        }
    }
    ((integer as u64)<<20)|fraction
}

/// PS9.0: build a sparse suffix model. A context table is retained only when
/// its measured coding gain over the best already-retained suffix exceeds
/// the exact wire cost of the context key and table, plus a small safety margin.
/// This is encoder-side model selection; the chosen tables are serialized, so
/// decoding does not need to repeat model selection.
/// Pack a suffix of at most four bytes into a sortable u64. The high byte
/// stores its length, so empty/short contexts cannot collide with longer ones.
/// This removes a heap allocation for every (position, order) during training.
#[inline]
fn packed_context(input: &[u8], pos: usize, order: usize) -> u64 {
    let n = pos.min(order);
    let mut bytes = 0u32;
    for &byte in &input[pos - n..pos] {
        bytes = (bytes << 8) | byte as u32;
    }
    ((n as u64) << 32) | bytes as u64
}

#[inline]
fn unpack_context(mut key: u64) -> Vec<u8> {
    let n = (key >> 32) as usize;
    let mut bytes = key as u32;
    let mut out = vec![0u8; n];
    for i in (0..n).rev() {
        out[i] = bytes as u8;
        bytes >>= 8;
    }
    out
}

fn train_pruned(input: &[u8], order: usize) -> Result<BTreeMap<Vec<u8>, Table>, UmceError> {
    // A packed integer key avoids allocating Vec<u8> for each byte position.
    // BTreeMap preserves deterministic traversal and serialized table order.
    let mut counts: BTreeMap<u64, [u64; 256]> = BTreeMap::new();
    for (pos, &byte) in input.iter().enumerate() {
        for o in 0..=order.min(pos) {
            let key = packed_context(input, pos, o);
            if !counts.contains_key(&key) && counts.len() >= 8_192 {
                return Err(UmceError::InvalidDescriptor("PS9.0 context budget exceeded"));
            }
            let row = counts.entry(key).or_insert([0; 256]);
            row[byte as usize] = row[byte as usize]
                .checked_add(1).ok_or(UmceError::LengthOverflow)?;
        }
    }

    let mut tables: BTreeMap<Vec<u8>, Table> = BTreeMap::new();
    for (packed_key, row) in counts {
        let key = unpack_context(packed_key);
        if key.is_empty() {
            tables.insert(key, Table::from_counts(&row)?);
            continue;
        }

        // Upper-bound pruning: no symbol can gain more than 12 bits under
        // the 12-bit rANS frequency scale. Skip table construction if that
        // theoretical gain cannot pay the minimum metadata cost.
        let observations = row.iter().try_fold(0u64, |sum, &v| {
            sum.checked_add(v).ok_or(UmceError::LengthOverflow)
        })?;
        let minimum_wire_bytes = 1usize.checked_add(key.len()).and_then(|n| n.checked_add(5))
            .ok_or(UmceError::LengthOverflow)?;
        let minimum_metadata_bits = minimum_wire_bytes.checked_mul(8)
            .and_then(|n| n.checked_add(8)).ok_or(UmceError::LengthOverflow)?;
        let maximum_gain_q20 = (observations as u128)
            .checked_mul(12u128 << 20).ok_or(UmceError::LengthOverflow)?;
        let minimum_cost_q20 = (minimum_metadata_bits as u128)
            .checked_mul(1u128 << 20).ok_or(UmceError::LengthOverflow)?;
        if maximum_gain_q20 <= minimum_cost_q20 { continue; }

        let candidate = Table::from_counts(&row)?;
        let parent = (1..key.len()).rev()
            .find_map(|suffix_start| tables.get(&key[suffix_start..]))
            .or_else(|| tables.get(&Vec::new()))
            .ok_or(UmceError::InvalidDescriptor("PS9.0 global fallback missing"))?;
        let mut gain_q20 = 0i128;
        for (symbol, &count) in row.iter().enumerate() {
            if count == 0 { continue; }
            let parent_log = log2_q20(parent.freq[symbol] as u32) as i128;
            let candidate_log = log2_q20(candidate.freq[symbol] as u32) as i128;
            let contribution = (count as i128).checked_mul(candidate_log - parent_log)
                .ok_or(UmceError::LengthOverflow)?;
            gain_q20 = gain_q20.checked_add(contribution).ok_or(UmceError::LengthOverflow)?;
        }
        let wire_bytes = 1usize.checked_add(key.len()).and_then(|n| n.checked_add(candidate.wire.len()))
            .ok_or(UmceError::LengthOverflow)?;
        let metadata_bits = wire_bytes.checked_mul(8).and_then(|n| n.checked_add(8))
            .ok_or(UmceError::LengthOverflow)?;
        let required_gain = i128::try_from(metadata_bits).map_err(|_| UmceError::LengthOverflow)?
            .checked_mul(1i128 << 20).ok_or(UmceError::LengthOverflow)?;
        if gain_q20 > required_gain { tables.insert(key, candidate); }
    }
    if !tables.contains_key(&Vec::new()) {
        return Err(UmceError::InvalidDescriptor("PS9.0 global fallback missing"));
    }
    Ok(tables)
}

fn choose<'a>(tables:&'a BTreeMap<Vec<u8>,
Table>,
history:&[u8],
order:usize)->Result<&'a Table,
UmceError> {
    for o in(0..=order.min(history.len())).rev() {
        let key=history[history.len()-o..].to_vec();
        if let Some(t)=tables.get(&key) {
            return Ok(t);
            
        }
        
    }
    Err(UmceError::InvalidDescriptor("PS6.8 backoff model missing"))
}

fn payload(input:&[u8],
order:usize)->Result<Vec<u8>,
UmceError> {
    let tables=train(input,
    order)?;
    let mut out=Vec::new();
    out.push(order as u8);
    out.extend_from_slice(&u32::try_from(tables.len()).map_err(|_|UmceError::LengthOverflow)?.to_le_bytes());
    for(k,
    t)in &tables {
        out.push(k.len() as u8);
        out.extend_from_slice(k);
        out.extend_from_slice(&t.wire);
        
    }
    let mut state=RANS_L;
    let mut emitted=Vec::new();
    for i in(0..input.len()).rev() {
        let t=choose(&tables,
        &input[..i],
        order)?;
        let s=input[i]as usize;
        let f=t.freq[s]as u64;
        let c=t.cum[s]as u64;
        let xmax=((RANS_L>>SCALE_BITS)<<8)*f;
        while state>=xmax {
            emitted.push(state as u8);
            state>>=8;
            
        }
        state=((state/f)<<SCALE_BITS)+(state%f)+c;
        
    }
    out.extend_from_slice(&u32::try_from(state).map_err(|_|UmceError::LengthOverflow)?.to_le_bytes());
    emitted.reverse();
    out.extend_from_slice(&emitted);
    Ok(out)
}


fn payload_pruned(input:&[u8],order:usize)->Result<Vec<u8>,UmceError> {
    let tables=train_pruned(input,order)?;
    let mut out=Vec::new();
    out.push(order as u8);
    out.extend_from_slice(&u32::try_from(tables.len()).map_err(|_|UmceError::LengthOverflow)?.to_le_bytes());
    for (k,t) in &tables {
        out.push(k.len() as u8);
        out.extend_from_slice(k);
        out.extend_from_slice(&t.wire);
    }
    let mut state=RANS_L;
    let mut emitted=Vec::new();
    for i in (0..input.len()).rev() {
        let t=choose(&tables,&input[..i],order)?;
        let symbol=input[i] as usize;
        let f=t.freq[symbol] as u64;
        let c=t.cum[symbol] as u64;
        let xmax=((RANS_L>>SCALE_BITS)<<8)*f;
        while state>=xmax { emitted.push(state as u8); state>>=8; }
        state=((state/f)<<SCALE_BITS)+(state%f)+c;
    }
    out.extend_from_slice(&u32::try_from(state).map_err(|_|UmceError::LengthOverflow)?.to_le_bytes());
    emitted.reverse();
    out.extend_from_slice(&emitted);
    Ok(out)
}

pub(super) fn frame_pruned(input:&[u8],order:usize)->Result<Vec<u8>,UmceError> {
    if input.is_empty() { return Err(UmceError::InvalidDescriptor("PS9.0 requires nonempty input")); }
    if !(1..=4).contains(&order) { return Err(UmceError::InvalidDescriptor("PS9.0 order must be 1..4")); }
    let mut out=Vec::new();
    out.extend_from_slice(b"PYAE");
    out.push(6); // new frame version; PS6.8 version 4 remains unchanged
    out.push(order as u8);
    out.extend_from_slice(&u64::try_from(input.len()).map_err(|_|UmceError::LengthOverflow)?.to_le_bytes());
    out.extend_from_slice(&payload_pruned(input,order)?);
    Ok(out)
}

fn decode_payload(b:&[u8],
len:usize)->Result<Vec<u8>,
UmceError> {
    if b.len()<5 {
        return Err(UmceError::InvalidDescriptor("truncated PS6.8 payload"));
        
    }
    let order=b[0]as usize;
    if !(1..=4).contains(&order) {
        return Err(UmceError::InvalidDescriptor("unsupported PS6.8 order"));
        
    }
    let n=u32::from_le_bytes(b[1..5].try_into().map_err(|_|UmceError::InvalidDescriptor("bad PS6.8 count"))?)as usize;
    if n==0||n>8_192 {
        return Err(UmceError::InvalidDescriptor("PS6.8 model count limit"));
        
    }
    let mut p=5;
    let mut tables=BTreeMap::new();
    let mut prev:Option<Vec<u8>>=None;
    for _ in 0..n {
        if p>=b.len() {
            return Err(UmceError::InvalidDescriptor("truncated PS6.8 key"));
            
        }
        let kl=b[p]as usize;
        p+=1;
        if kl>order||b.len().saturating_sub(p)<kl {
            return Err(UmceError::InvalidDescriptor("invalid PS6.8 key length"));
            
        }
        let key=b[p..p+kl].to_vec();
        p+=kl;
        if prev.as_ref().map_or(false,
        |x|x>=&key) {
            return Err(UmceError::InvalidDescriptor("noncanonical PS6.8 key order"));
            
        }
        prev=Some(key.clone());
        let t=Table::parse(b,
        &mut p)?;
        tables.insert(key,
        t);
        
    }
    if !tables.contains_key(&Vec::new()) {
        return Err(UmceError::InvalidDescriptor("PS6.8 global fallback missing"));
        
    }
    if b.len().saturating_sub(p)<4 {
        return Err(UmceError::InvalidDescriptor("missing PS6.8 rANS state"));
        
    }
    let mut state=u32::from_le_bytes(b[p..p+4].try_into().map_err(|_|UmceError::InvalidDescriptor("bad PS6.8 state"))?)as u64;
    p+=4;
    if state<RANS_L {
        return Err(UmceError::InvalidDescriptor("invalid PS6.8 state"));
        
    }
    let renorm=&b[p..];
    let mut rp=0;
    let mut out=Vec::with_capacity(len);
    for _ in 0..len {
        let t=choose(&tables,
        &out,
        order)?;
        let slot=(state as u32)&(SCALE-1);
        let s=t.symbol(slot)?;
        let f=t.freq[s]as u64;
        let c=t.cum[s]as u64;
        state=f*(state>>SCALE_BITS)+(slot as u64-c);
        while state<RANS_L {
            let v=*renorm.get(rp).ok_or(UmceError::InvalidDescriptor("truncated PS6.8 renormalization"))?;
            rp+=1;
            state=(state<<8)|v as u64;
            
        }
        out.push(s as u8);
        
    }
    if rp!=renorm.len()||state!=RANS_L {
        return Err(UmceError::InvalidDescriptor("PS6.8 trailing bytes or terminal state mismatch"));
        
    }
    Ok(out)
}

pub(super) fn frame(input:&[u8],
order:usize)->Result<Vec<u8>,
UmceError> {
    if input.is_empty() {
        return Err(UmceError::InvalidDescriptor("PS6.8 requires nonempty input"));
        
    }
    if !(1..=4).contains(&order) {
        return Err(UmceError::InvalidDescriptor("PS6.8 order must be 1..4"));
        
    }
    let mut out=Vec::new();
    out.extend_from_slice(b"PYAE");
    out.push(4);
    out.push(order as u8);
    out.extend_from_slice(&u64::try_from(input.len()).map_err(|_|UmceError::LengthOverflow)?.to_le_bytes());
    out.extend_from_slice(&payload(input,
    order)?);
    Ok(out)
}

pub(super) fn decode_frame(b:&[u8],
max:usize)->Result<Vec<u8>,
UmceError> {
    if b.len()<14||&b[..4]!=b"PYAE"||!(b[4]==4||b[4]==6)||!(1..=4).contains(&b[5]) {
        return Err(UmceError::InvalidDescriptor("bad PS6.8 frame"));
        
    }
    let len=usize::try_from(u64::from_le_bytes(b[6..14].try_into().map_err(|_|UmceError::InvalidDescriptor("bad PS6.8 length"))?)).map_err(|_|UmceError::LengthOverflow)?;
    if len>max {
        return Err(UmceError::InvalidDescriptor("PS6.8 output limit exceeded"));
        
    }
    let out=decode_payload(&b[14..],
    len)?;
    if out.len()!=len {
        return Err(UmceError::LengthMismatch {
            expected:len,
            actual:out.len()
        });
        
    }
    Ok(out)
}

pub(super) fn order_from_frame(b:&[u8])->Option<usize> {
    if b.len()>=6&&&b[..4]==b"PYAE"&&(b[4]==4||b[4]==6) {
        Some(b[5]as usize)
    }
    else {
        None
    }
    
}

#[cfg(test)]
mod ps92_packed_context_tests {
    use super::{packed_context, unpack_context};

    #[test]
    fn packed_context_roundtrips_all_orders_and_prefix_lengths() {
        let data = [0x00, 0x12, 0x34, 0x56, 0x78, 0xff];
        for pos in 0..=data.len() {
            for order in 0..=4 {
                let n = pos.min(order);
                let expected = &data[pos - n..pos];
                assert_eq!(unpack_context(packed_context(&data, pos, order)), expected);
            }
        }
    }

    #[test]
    fn packed_context_key_orders_by_length_then_bytes() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let empty = packed_context(&data, 0, 4);
        let one = packed_context(&data, 1, 1);
        let two = packed_context(&data, 2, 2);
        let four = packed_context(&data, 4, 4);
        assert!(empty < one && one < two && two < four);
        assert_eq!(unpack_context(four), vec![1, 2, 3, 4]);
    }
}
