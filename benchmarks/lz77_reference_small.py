# Independent Python reference for PS8.0 token grammar and bounded hash-chain.
import struct, hashlib
MAGIC=b'PYLZ'; HEADER=9; WINDOW=65535; MIN=4; MAX=130; HS=1<<16; DEPTH=32; NONE=-1

def h3(b,p):
    x,y,z=b[p],b[p+1],b[p+2]
    return ((((x*0x1e35a7bd)&0xffffffff) ^ (y<<11) ^ (z<<3)) >> 16) & (HS-1)
def enc(b):
    out=bytearray(MAGIC+b'\x01'+struct.pack('<I',len(b)))
    head=[NONE]*HS; prev=[NONE]*len(b); lits=bytearray(); p=0
    def flush():
        nonlocal lits
        q=0
        while q<len(lits):
            n=min(128,len(lits)-q); out.append(n-1); out.extend(lits[q:q+n]); q+=n
        lits=bytearray()
    while p<len(b):
        bl=0; bo=0
        if p+MIN<=len(b):
            h=h3(b,p); c=head[h]; probes=0
            while c!=NONE and probes<DEPTH:
                d=p-c
                if d>WINDOW: break
                n=0
                while n<MAX and p+n<len(b) and b[c+n]==b[p+n]: n+=1
                if n>=MIN and n>bl:
                    bl=n; bo=d
                    if n==MAX: break
                c=prev[c]; probes+=1
        if bl>=MIN:
            flush(); out.append(128+bl-3); out.extend(struct.pack('<H',bo)); end=p+bl
            while p<end:
                if p+2<len(b):
                    h=h3(b,p); prev[p]=head[h]; head[h]=p
                p+=1
        else:
            lits.append(b[p])
            if p+2<len(b):
                h=h3(b,p); prev[p]=head[h]; head[h]=p
            p+=1
            if len(lits)==128: flush()
    flush(); return bytes(out)
def dec(x):
    assert x[:5]==MAGIC+b'\x01' and len(x)>=9
    n=struct.unpack('<I',x[5:9])[0]; out=bytearray(); p=9
    while len(out)<n:
        t=x[p]; p+=1
        if t<128:
            k=t+1; out.extend(x[p:p+k]); p+=k
        else:
            k=(t&127)+3; d=struct.unpack('<H',x[p:p+2])[0]; p+=2
            assert k>=4 and 0<d<=len(out)
            for _ in range(k): out.append(out[-d])
        assert len(out)<=n
    assert p==len(x)
    return bytes(out)

samples=[b'',b'a',b'\0'*100000,(b'abcabcabcabcabcabc'),bytes(range(256))*256]
samples.append((b'record:alpha|record:beta|')*4096)
samples.append(bytes((i*73+i//11)&255 for i in range(65536)))
for b in samples:
    z=enc(b); assert dec(z)==b
    print(len(b),len(z),round(len(z)/max(1,len(b)),4),hashlib.sha256(b).hexdigest()[:12])
