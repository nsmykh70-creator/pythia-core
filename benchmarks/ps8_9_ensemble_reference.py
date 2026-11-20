#!/usr/bin/env python3
"""Independent Python reference for PYTHIA PS8.9 reversible ensembles.

This validates predictor synchronization and reports zlib diagnostic sizes.
It is not a substitute for Rust cargo test or PYTHIA entropy-frame benchmarks.
"""
from __future__ import annotations
import argparse, hashlib, random, time, zlib

Q = 1 << 24
L = 1 << 16

def dist(a: int, b: int) -> int:
    d = (a - b) & 255
    return min(d, (-d) & 255)

def predictors(p1: int, p2: int, p3: int):
    return (0, p1, (2*p1-p2)&255, (3*p1-3*p2+p3)&255, p1^p2)

def medoid_choice(ps, losses):
    weights = [1 + (1 << 20)//(v+1) for v in losses]
    return min(range(5), key=lambda c: sum(weights[k]*dist(ps[c], ps[k]) for k in range(5)))

def medoid(data: bytes, decode=False) -> bytes:
    losses = [0]*5; p1=p2=p3=0; out=bytearray()
    for b in data:
        ps=predictors(p1,p2,p3); c=medoid_choice(ps,losses)
        x=((b+ps[c])&255) if decode else b
        out.append(x if decode else ((b-ps[c])&255))
        for k in range(5): losses[k]=min((1<<64)-1,losses[k]+dist(x,ps[k]))
        p3,p2,p1=p2,p1,x
    return bytes(out)

def likelihood(d): return max(1,L//(1+d))

def mix_choice(ps, weights):
    scores=[sum(weights[k]*likelihood(dist(ps[c],ps[k])) for k in range(5)) for c in range(5)]
    return max(range(5), key=lambda c:scores[c])

def update(weights, actual, ps):
    raw=[max(1,weights[k]*likelihood(dist(actual,ps[k]))) for k in range(5)]
    total=sum(raw); out=[v*Q//total for v in raw]
    out[max(range(5),key=lambda k:raw[k])] += Q-sum(out)
    return out

def probabilistic(data: bytes, decode=False) -> bytes:
    weights=[Q//5]*5; weights[0]+=Q-sum(weights)
    p1=p2=p3=0; out=bytearray()
    for b in data:
        ps=predictors(p1,p2,p3); c=mix_choice(ps,weights)
        x=((b+ps[c])&255) if decode else b
        out.append(x if decode else ((b-ps[c])&255))
        weights=update(weights,x,ps)
        p3,p2,p1=p2,p1,x
    return bytes(out)

def make_cases():
    cases={
        'linear_64KiB': bytes((i*17)&255 for i in range(1<<16)),
        'quadratic_64KiB': bytes((i*i+3*i+7)&255 for i in range(1<<16)),
        'periodic_text_64KiB': (b'PYTHIA predictive compression pattern; ' * ((1<<16)//38+1))[:1<<16],
        'constant_64KiB': bytes([65])*(1<<16),
    }
    r=random.Random(0x89A)
    cases['random_64KiB']=bytes(r.randrange(256) for _ in range(1<<16))
    return cases

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input'); args=ap.parse_args()
    cases=make_cases()
    if args.input:
        with open(args.input,'rb') as f: cases['external_input']=f.read()
    print('case,model,raw_bytes,zlib_bytes,roundtrip,seconds,sha256')
    for name, raw in cases.items():
        for model, fn in [('identity',lambda x:x),('weighted_medoid',medoid),('probabilistic_mixture',probabilistic)]:
            t=time.perf_counter(); transformed=fn(raw); elapsed=time.perf_counter()-t
            if model=='identity': restored=transformed
            else:
                t2=time.perf_counter(); restored=fn(transformed,decode=True); elapsed += time.perf_counter()-t2
            ok=restored==raw
            print(f'{name},{model},{len(transformed)},{len(zlib.compress(transformed,6))},{ok},{elapsed:.4f},{hashlib.sha256(restored).hexdigest()}')
            if not ok: raise SystemExit(f'roundtrip failed: {name}/{model}')
if __name__=='__main__': main()
