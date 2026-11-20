//! PS6.7 Context Modeling Engine: static order-1 byte contexts + rANS.
//! Context tables are serialized canonically and fully charged to the frame.
use crate::UmceError;

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
    bytes:Vec<u8>
}

impl Table {
    
     fn from_counts(c:&[u64;
    256])->Result<Self,
    UmceError> {
        let total:u64=c.iter().sum();
        let n=c.iter().filter(|&&x|x>0).count();
        if n==0||n>SCALE as usize {
            return Err(UmceError::InvalidDescriptor("empty or oversized CME context"));
            
        }
        let rem=SCALE as u64-n as u64;
        let mut f=[0u16;
        256];
        let mut assigned=n as u64;
        let mut rs=Vec::new();
        for(s,
        &v)in c.iter().enumerate() {
            if v==0 {
                continue
            }
            let p=v.checked_mul(rem).ok_or(UmceError::LengthOverflow)?;
            let e=p/total;
            f[s]=(1+e)as u16;
            assigned+=e;
            rs.push((p%total,
            s));
            
        }
        rs.sort_by(|a,
        b|b.0.cmp(&a.0).then_with(||a.1.cmp(&b.1)));
        for&(_,
        s)in rs.iter().take((SCALE as u64-assigned)as usize) {
            f[s]+=1;
            
        }
        let mut cum=[0u32;
        257];
        for i in 0..256 {
            cum[i+1]=cum[i]+f[i]as u32;
            
        }
        if cum[256]!=SCALE {
            return Err(UmceError::InvalidDescriptor("bad CME normalization"));
            
        }
        let mut bytes=Vec::new();
        bytes.extend_from_slice(&(n as u16).to_le_bytes());
        for(s,
        &v)in f.iter().enumerate() {
            if v>0 {
                bytes.push(s as u8);
                bytes.extend_from_slice(&v.to_le_bytes());
                
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
            bytes
        })
    }
    
     fn parse(b:&[u8],
    p:&mut usize)->Result<Self,
    UmceError> {
        if b.len().saturating_sub(*p)<2 {
            return Err(UmceError::InvalidDescriptor("truncated CME table"));
            
        }
        let n=u16::from_le_bytes([b[*p],
        b[*p+1]])as usize;
        *p+=2;
        if n==0||n>256||b.len().saturating_sub(*p)<n*3 {
            return Err(UmceError::InvalidDescriptor("invalid CME table length"));
            
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
                return Err(UmceError::InvalidDescriptor("noncanonical CME table"));
                
            }
            prev=Some(s);
            f[s as usize]=v;
            sum=sum.checked_add(v as u32).ok_or(UmceError::LengthOverflow)?;
            
        }
        if sum!=SCALE {
            return Err(UmceError::InvalidDescriptor("CME frequencies do not sum to scale"));
            
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
            bytes:b[start..*p].to_vec()
        })
    }
    
     fn symbol(&self,
    x:u32)->Result<usize,
    UmceError> {
        if x>=SCALE {
            return Err(UmceError::InvalidDescriptor("CME slot outside table"));
            
        }
        Ok(self.decode[x as usize] as usize)
    }
    
    
}

fn build(input:&[u8])->Result<(Table,
Vec<Option<Table>>,
Vec<u8>),
UmceError> {
    let mut g=[0u64;
    256];
    let mut c=[[0u64;
    256];
    256];
    for&b in input {
        g[b as usize]+=1;
        
    }
    for w in input.windows(2) {
        c[w[0]as usize][w[1]as usize]+=1;
        
    }
    let global=Table::from_counts(&g)?;
    let mut models:Vec<Option<Table>>=(0..256).map(|_|None).collect();
    let mut records=Vec::new();
    let mut count=0u16;
    for i in 0..256 {
        if c[i].iter().any(|&x|x>0) {
            let t=Table::from_counts(&c[i])?;
            records.push(i as u8);
            records.extend_from_slice(&t.bytes);
            models[i]=Some(t);
            count+=1;
            
        }
        
    }
    let mut meta=global.bytes.clone();
    meta.extend_from_slice(&count.to_le_bytes());
    meta.extend_from_slice(&records);
    Ok((global,
    models,
    meta))
}

fn encode_payload(input:&[u8])->Result<Vec<u8>,
UmceError> {
    if input.is_empty() {
        return Err(UmceError::InvalidDescriptor("CME requires nonempty input"));
        
    }
    let(g,
    m,
    mut out)=build(input)?;
    let mut state=RANS_L;
    let mut emitted=Vec::new();
    for i in(0..input.len()).rev() {
        let t=if i==0 {
            &g
        }
        else {
            m[input[i-1]as usize].as_ref().ok_or(UmceError::InvalidDescriptor("missing CME context"))?
        };
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

fn decode_payload(b:&[u8],
len:usize)->Result<Vec<u8>,
UmceError> {
    if len==0 {
        return if b.is_empty() {
            Ok(Vec::new())
        }
        else {
            Err(UmceError::InvalidDescriptor("unexpected CME payload"))
        };
        
    }
    let mut p=0;
    let g=Table::parse(b,
    &mut p)?;
    if b.len().saturating_sub(p)<2 {
        return Err(UmceError::InvalidDescriptor("truncated CME context count"));
        
    }
    let n=u16::from_le_bytes([b[p],
    b[p+1]])as usize;
    p+=2;
    let mut m:Vec<Option<Table>>=(0..256).map(|_|None).collect();
    let mut prev=None;
    for _ in 0..n {
        if p>=b.len() {
            return Err(UmceError::InvalidDescriptor("truncated CME context id"));
            
        }
        let id=b[p];
        p+=1;
        if prev.map_or(false,
        |x|id<=x) {
            return Err(UmceError::InvalidDescriptor("noncanonical CME context order"));
            
        }
        prev=Some(id);
        m[id as usize]=Some(Table::parse(b,
        &mut p)?);
        
    }
    if b.len().saturating_sub(p)<4 {
        return Err(UmceError::InvalidDescriptor("truncated CME state"));
        
    }
    let mut state=u32::from_le_bytes(b[p..p+4].try_into().unwrap())as u64;
    p+=4;
    if state<RANS_L {
        return Err(UmceError::InvalidDescriptor("invalid CME state"));
        
    }
    let renorm=&b[p..];
    let mut cur=0;
    let mut out=Vec::with_capacity(len);
    for i in 0..len {
        let t=if i==0 {
            &g
        }
        else {
            m[out[i-1]as usize].as_ref().ok_or(UmceError::InvalidDescriptor("missing decoded CME context"))?
        };
        let slot=(state as u32)&(SCALE-1);
        let s=t.symbol(slot)?;
        let f=t.freq[s]as u64;
        let c=t.cum[s]as u64;
        state=f*(state>>SCALE_BITS)+(slot as u64-c);
        while state<RANS_L {
            let v=*renorm.get(cur).ok_or(UmceError::InvalidDescriptor("truncated CME renormalization"))?;
            cur+=1;
            state=(state<<8)|v as u64;
            
        }
        out.push(s as u8);
        
    }
    if cur!=renorm.len()||state!=RANS_L {
        return Err(UmceError::InvalidDescriptor("CME trailing data or invalid terminal state"));
        
    }
    Ok(out)
}

pub(super) fn frame(input:&[u8])->Result<Vec<u8>,
UmceError> {
    let len=u64::try_from(input.len()).map_err(|_|UmceError::LengthOverflow)?;
    let payload=encode_payload(input)?;
    let mut out=Vec::with_capacity(14+payload.len());
    out.extend_from_slice(b"PYAE");
    out.push(3);
    out.push(3);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

pub(super) fn decode_frame(b:&[u8],
max:usize)->Result<Vec<u8>,
UmceError> {
    if b.len()<14||&b[..4]!=b"PYAE"||b[4]!=3||b[5]!=3 {
        return Err(UmceError::InvalidDescriptor("bad CME frame"));
        
    }
    let len=usize::try_from(u64::from_le_bytes(b[6..14].try_into().unwrap())).map_err(|_|UmceError::LengthOverflow)?;
    if len>max {
        return Err(UmceError::InvalidDescriptor("CME output limit exceeded"));
        
    }
    decode_payload(&b[14..],
    len)
}
