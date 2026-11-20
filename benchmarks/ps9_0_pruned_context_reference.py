#!/usr/bin/env python3
"""Independent Python reference for the PS9.0 sparse suffix-context/rANS design.
This validates the algorithmic reference, not the Rust build.
"""
from collections import defaultdict
import csv, hashlib
from pathlib import Path
SCALE_BITS=12
SCALE=1<<SCALE_BITS
RANS_L=1<<23

def normalize(counts):
    total=sum(counts); present=[i for i,x in enumerate(counts) if x]
    freq=[0]*256
    if not present: raise ValueError('empty')
    remaining=SCALE-len(present); assigned=len(present); rems=[]
    for s in present:
        product=counts[s]*remaining
        extra,rem=divmod(product,total)
        freq[s]=1+extra; assigned+=extra; rems.append((rem,s))
    rems.sort(key=lambda x:(-x[0],x[1]))
    for _,s in rems[:SCALE-assigned]: freq[s]+=1
    assert sum(freq)==SCALE
    return freq

def contexts(data, order):
    counts=defaultdict(lambda:[0]*256)
    for pos,b in enumerate(data):
        for o in range(min(order,pos)+1):
            key=data[pos-o:pos] if o else b''
            counts[key][b]+=1
    return counts

def table_wire(freq):
    out=bytearray(sum(x>0 for x in freq).to_bytes(2,'little'))
    for s,f in enumerate(freq):
        if f: out.extend((s,)); out.extend(f.to_bytes(2,'little'))
    return bytes(out)

def log2_q20(value):
    integer=value.bit_length()-1
    x=(value<<32)>>integer
    fraction=0
    for bit in range(20):
        x=(x*x)>>32
        if x >= (2<<32):
            x >>= 1
            fraction |= 1 << (19-bit)
    return (integer<<20)|fraction

def train(data,order,pruned):
    counts=contexts(data,order)
    items=sorted(counts.items(),key=lambda kv:(len(kv[0]),kv[0]))
    tables={}
    for key,row in items:
        freq=normalize(row)
        if not pruned or not key:
            tables[key]=freq; continue
        parent=None
        for start in range(1,len(key)):
            if key[start:] in tables:
                parent=tables[key[start:]]; break
        if parent is None: parent=tables[b'']
        gain_q20=0
        for s,n in enumerate(row):
            if n:
                gain_q20 += n*(log2_q20(freq[s])-log2_q20(parent[s]))
        metadata=1+len(key)+len(table_wire(freq))
        if gain_q20 > (metadata*8+8)*(1<<20):
            tables[key]=freq
    return tables

def choose(tables,history,order):
    for o in range(min(order,len(history)),-1,-1):
        key=history[-o:] if o else b''
        if key in tables:return key,tables[key]
    raise AssertionError('no fallback')

def encode(data,order,pruned):
    tables=train(data,order,pruned)
    meta=bytearray([order]); meta.extend(len(tables).to_bytes(4,'little'))
    for key,freq in sorted(tables.items()):
        meta.append(len(key)); meta.extend(key); meta.extend(table_wire(freq))
    cum=[0]*257
    # per-table cumulative distributions
    cums={}
    for key,f in tables.items():
        c=[0]*257
        for i in range(256):c[i+1]=c[i]+f[i]
        cums[key]=c
    state=RANS_L; emitted=bytearray()
    for i in range(len(data)-1,-1,-1):
        hist=data[:i]; key,freq=choose(tables,hist,order); c=cums[key]
        s=data[i]; f=freq[s]; start=c[s]
        xmax=((RANS_L>>SCALE_BITS)<<8)*f
        while state>=xmax:
            emitted.append(state&255); state>>=8
        state=((state//f)<<SCALE_BITS)+(state%f)+start
    emitted.reverse(); meta.extend(state.to_bytes(4,'little')); meta.extend(emitted)
    return bytes(meta),len(tables),sum(len(table_wire(f))+1+len(k) for k,f in tables.items())

def decode(payload,out_len):
    order=payload[0]; n=int.from_bytes(payload[1:5],'little'); p=5; tables={}
    for _ in range(n):
        kl=payload[p];p+=1; key=payload[p:p+kl];p+=kl
        count=int.from_bytes(payload[p:p+2],'little');p+=2
        freq=[0]*256
        for _ in range(count):
            s=payload[p];f=int.from_bytes(payload[p+1:p+3],'little');p+=3;freq[s]=f
        tables[key]=freq
    state=int.from_bytes(payload[p:p+4],'little');p+=4; renorm=payload[p:];rp=0
    cums={}
    decodes={}
    for key,f in tables.items():
        c=[0]*257
        for i in range(256):c[i+1]=c[i]+f[i]
        cums[key]=c
        d=[]
        for s in range(256):d.extend([s]*f[s])
        decodes[key]=d
    out=bytearray()
    for _ in range(out_len):
        key,freq=choose(tables,bytes(out),order)
        slot=state&(SCALE-1); sym=decodes[key][slot]; f=freq[sym]; c=cums[key][sym]
        state=f*(state>>SCALE_BITS)+(slot-c)
        while state<RANS_L:
            state=(state<<8)|renorm[rp];rp+=1
        out.append(sym)
    assert rp==len(renorm) and state==RANS_L
    return bytes(out)

def main():
    cases={
      'linear':bytes(((i*17)&255) for i in range(65536)),
      'periodic':(b'abracadabra|PYTHIA|context-model|'*2048)[:65536],
      'text':(b'The quick brown fox jumps over the lazy dog. '*1500)[:65536],
      'constant':bytes([65])*65536,
      'random_lcg':bytes((((i*1103515245+12345)>>16)&255) for i in range(65536)),
    }
    rows=[]
    for name,data in cases.items():
      for order in range(1,5):
        old,n0,m0=encode(data,order,False)
        new,n1,m1=encode(data,order,True)
        assert encode(data,order,True)[0] == new
        assert decode(old,len(data))==data
        assert decode(new,len(data))==data
        rows.append((name,order,len(old)+14,len(new)+14,n0,n1,m0,m1,hashlib.sha256(data).hexdigest()))
    out=Path(__file__).with_name('PS9_0_REFERENCE_RESULTS.csv')
    with open(out,'w',newline='') as f:
      w=csv.writer(f);w.writerow(['dataset','order','ps68_frame_bytes','ps90_pruned_frame_bytes','ps68_tables','ps90_tables','ps68_metadata_bytes','ps90_metadata_bytes','input_sha256']);w.writerows(rows)
    print('reference_roundtrips:',len(rows)*2)
    for r in rows: print(r[0],r[1],r[2],r[3],r[4],r[5])
if __name__=='__main__':main()
