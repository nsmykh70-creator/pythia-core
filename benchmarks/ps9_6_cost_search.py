#!/usr/bin/env python3
"""PS9.6 reversible transform-chain discovery, reference implementation.
Uses zlib as a measurable proxy backend; not the PYTHIA production entropy coder.
"""
from __future__ import annotations
import argparse, zlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Transform:
    name: str
    parameter: int
    descriptor_cost: int
    forward: Callable[[bytes, int], bytes]
    inverse: Callable[[bytes, int], bytes]


def delta_f(data: bytes, lag: int) -> bytes:
    if lag <= 0: raise ValueError('lag must be positive')
    return bytes((v - (data[i-lag] if i >= lag else 0)) & 255 for i, v in enumerate(data))
def delta_i(data: bytes, lag: int) -> bytes:
    out=bytearray()
    for i,v in enumerate(data): out.append((v+(out[i-lag] if i>=lag else 0))&255)
    return bytes(out)
def xor_f(data: bytes, lag: int) -> bytes:
    if lag <= 0: raise ValueError('lag must be positive')
    return bytes(v ^ (data[i-lag] if i>=lag else 0) for i,v in enumerate(data))
def xor_i(data: bytes, lag: int) -> bytes:
    out=bytearray()
    for i,v in enumerate(data): out.append(v ^ (out[i-lag] if i>=lag else 0))
    return bytes(out)
def transpose_f(data: bytes, width: int) -> bytes:
    if width <= 1: return data
    out=bytearray()
    for start in range(0,len(data),width):
        block=data[start:start+width]
        out.extend(block[i] for i in range(len(block))) # byte lanes unchanged for partial blocks
    # lane grouping: concatenate same position in each width-sized block
    blocks=[data[i:i+width] for i in range(0,len(data),width)]
    return bytes(blocks[b][lane] for lane in range(width) for b in range(len(blocks)) if lane < len(blocks[b]))
def transpose_i(data: bytes, width: int) -> bytes:
    blocks=[range(min(width, len(data)-i)) for i in range(0,len(data),width)]
    lengths=[len(b) for b in blocks]; out=bytearray(sum(lengths)); pos=0
    for lane in range(width):
        for bi,n in enumerate(lengths):
            if lane<n: out[bi*width+lane]=data[pos]; pos+=1
    return bytes(out)
def bitplane_f(data: bytes, _: int) -> bytes:
    return bytes((data[i]>>bit)&1 for bit in range(8) for i in range(len(data)))
def bitplane_i(data: bytes, _: int) -> bytes:
    if len(data)%8: raise ValueError('bitplane input length must be multiple of 8')
    n=len(data)//8; out=bytearray(n)
    for bit in range(8):
        for i in range(n): out[i]|=(data[bit*n+i]&1)<<bit
    return bytes(out)

TRANSFORMS=[Transform('delta',lag,2,delta_f,delta_i) for lag in (1,2,3,4,8,16,32)] + [Transform('xor',lag,2,xor_f,xor_i) for lag in (1,2,3,4,8,16,32)] + [Transform('transpose',w,2,transpose_f,transpose_i) for w in (4,8,16,32,64)]

def apply_chain(data:bytes, chain:tuple[Transform,...])->bytes:
    cur=data
    for t in chain: cur=t.forward(cur,t.parameter)
    return cur
def inverse_chain(data:bytes, chain:tuple[Transform,...])->bytes:
    cur=data
    for t in reversed(chain): cur=t.inverse(cur,t.parameter)
    return cur
@dataclass
class Candidate:
    chain: tuple[Transform,...]
    encoded: bytes
    score: int
    raw_encoded_size: int

def evaluate(data:bytes, chain:tuple[Transform,...], level:int=9)->Candidate:
    transformed=apply_chain(data,chain)
    # bit-plane is excluded from search currently due to padding/length descriptor needs;
    # it remains a standalone tested transform and can be added with length-aware framing.
    payload=zlib.compress(transformed,level)
    overhead=sum(t.descriptor_cost+1 for t in chain) # type + parameter + chain framing
    score=len(payload)+overhead
    return Candidate(chain,transformed,score,len(payload))
def search(data:bytes, beam_width:int=8, max_depth:int=3, level:int=9, sample_limit:int=262144)->Candidate:
    # bounded prefix search, then full-input validation of the selected chain
    sample=data[:sample_limit]
    best=evaluate(sample,())
    beam=[best]
    # Avoid repeated identical transforms; chain combinations explored by bounded beam.
    for _depth in range(max_depth):
        expanded=[]
        for c in beam:
            for t in TRANSFORMS:
                if t in c.chain: continue
                chain=c.chain+(t,)
                try: cand=evaluate(sample,chain,level)
                except Exception: continue
                expanded.append(cand)
                if cand.score<best.score: best=cand
        if not expanded: break
        expanded.sort(key=lambda x:(x.score,len(x.chain),tuple((t.name,t.parameter) for t in x.chain)))
        beam=expanded[:beam_width]
        # retain identity as a stable candidate if its score is near competitive
    return evaluate(data,best.chain,level)

def selftest()->None:
    import random
    cases=[b'',bytes(range(256)),bytes([7])*8193,bytes((i%13 for i in range(4097))),bytes(random.Random(42).randrange(256) for _ in range(8192))]
    for data in cases:
        for t in TRANSFORMS:
            transformed=t.forward(data,t.parameter)
            assert t.inverse(transformed,t.parameter)==data,(t.name,t.parameter,len(data))
        for chain in [(),(TRANSFORMS[0],),(TRANSFORMS[0],TRANSFORMS[7]),(TRANSFORMS[-1],TRANSFORMS[2])]:
            assert inverse_chain(apply_chain(data,chain),chain)==data
    # bitplane separately
    d=bytes(range(256))*3
    assert bitplane_i(bitplane_f(d,0),0)==d
    print('PS9.6 self-test: PASS; transforms, chains, bitplanes, round-trip')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('file',nargs='?'); ap.add_argument('--depth',type=int,default=3); ap.add_argument('--beam',type=int,default=8); args=ap.parse_args()
    selftest()
    if args.file:
        raw=open(args.file,'rb').read(); winner=search(raw,args.beam,args.depth)
        print(f'input={len(raw)} bytes; zlib-proxy={winner.raw_encoded_size}; score+descriptor={winner.score}')
        print('chain:', ' -> '.join(f'{t.name}({t.parameter})' for t in winner.chain) or 'IDENTITY')
