#!/usr/bin/env python3
"""PS9.5 experimental reversible candidate discovery. Entropy proxy, not archive size."""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections import Counter

@dataclass(frozen=True)
class Candidate:
    name: str
    params: tuple[int, ...]
    data: bytes
    entropy_bits: float
    overhead_bits: int
    score_bits: float

def entropy_bits(data: bytes) -> float:
    if not data: return 0.0
    n=len(data); return -sum((c/n)*math.log2(c/n) for c in Counter(data).values())*n
def delta(data: bytes, lag: int) -> bytes:
    out=bytearray(len(data))
    for i,b in enumerate(data): out[i]=(b-data[i-lag])&255 if i>=lag else b
    return bytes(out)
def delta_inverse(data: bytes, lag: int) -> bytes:
    out=bytearray(data)
    for i in range(lag,len(out)): out[i]=(data[i]+out[i-lag])&255
    return bytes(out)
def xor_lag(data: bytes, lag: int) -> bytes:
    out=bytearray(data)
    for i in range(lag,len(data)): out[i]=data[i]^data[i-lag]
    return bytes(out)
def xor_lag_inverse(data: bytes, lag: int) -> bytes:
    out=bytearray(data)
    for i in range(lag,len(data)): out[i]=data[i]^out[i-lag]
    return bytes(out)
def transpose_lanes(data: bytes, stride: int) -> bytes:
    # Reorder within fixed blocks; inverse uses same permutation.
    if stride<=1: return data
    out=bytearray()
    for start in range(0,len(data),stride):
        block=data[start:start+stride]
        out.extend(block[i] for i in range(0,len(block),2))
        out.extend(block[i] for i in range(1,len(block),2))
    return bytes(out)
def transpose_inverse(data: bytes, stride: int) -> bytes:
    if stride<=1:return data
    out=bytearray()
    for start in range(0,len(data),stride):
        block=data[start:start+stride]; n=len(block); ev=(n+1)//2
        out.extend(block[i//2] if i%2==0 else block[ev+i//2] for i in range(n))
    return bytes(out)
def bitplane(data: bytes) -> bytes:
    # 8 bit planes, MSB to LSB, packed into bytes; pads final plane to byte boundary.
    out=bytearray(); n=len(data)
    for bit in range(7,-1,-1):
        plane=bytearray((n+7)//8)
        for i,b in enumerate(data): plane[i//8]|=((b>>bit)&1)<<(7-(i%8))
        out.extend(plane)
    return bytes(out)
def bitplane_inverse(data: bytes, n: int) -> bytes:
    plane_len=(n+7)//8
    if len(data)!=8*plane_len: raise ValueError('invalid bitplane payload length')
    out=bytearray(n)
    for bit in range(7,-1,-1):
        plane=data[(7-bit)*plane_len:(8-bit)*plane_len]
        for i in range(n): out[i]|=((plane[i//8]>>(7-(i%8)))&1)<<bit
    return bytes(out)
def discover(data: bytes, max_lag: int=32, strides=(4,8,16,32,64)) -> list[Candidate]:
    candidates=[('identity',(),data,0)]
    for lag in (1,2,3,4,8,16,32):
        if lag<=max_lag and lag<len(data):
            candidates += [(f'delta', (lag,), delta(data,lag), 8),
                          (f'xor-lag',(lag,),xor_lag(data,lag),8)]
    for stride in strides:
        if stride<len(data): candidates.append(('transpose',(stride,),transpose_lanes(data,stride),8))
    bp=bitplane(data); candidates.append(('bitplane',(),bp,16+8*((len(data)+7)//8-len(data)//8)))
    # Shannon entropy estimate is a fast ranking heuristic only. Include descriptor cost.
    return sorted((Candidate(name,payload_params,transformed,entropy_bits(transformed),overhead,
                    entropy_bits(transformed)+overhead) for name,payload_params,transformed,overhead in candidates),key=lambda c:c.score_bits)
def inverse_candidate(c: Candidate, original_len: int) -> bytes:
    if c.name=='identity': return c.data
    if c.name=='delta': return delta_inverse(c.data,c.params[0])
    if c.name=='xor-lag': return xor_lag_inverse(c.data,c.params[0])
    if c.name=='transpose': return transpose_inverse(c.data,c.params[0])
    if c.name=='bitplane': return bitplane_inverse(c.data,original_len)
    raise ValueError('unknown transform')
if __name__=='__main__':
    import sys
    p=sys.argv[1] if len(sys.argv)>1 else '/mnt/data/mixed_10mb_compression_test.bin'
    d=open(p,'rb').read(); results=discover(d[:min(len(d),1<<20)])
    for c in results[:10]: print(f'{c.name:10s} params={c.params!s:10s} proxy={c.score_bits/8/1024:.1f} KiB (entropy+header)')
