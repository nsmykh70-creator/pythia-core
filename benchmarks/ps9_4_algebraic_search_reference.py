#!/usr/bin/env python3
"""Independent Python reference for PS9.4 causal algebraic predictor; diagnostic only."""
import random, math

TERM_LINEAR_1, TERM_LINEAR_2, TERM_SQUARE_1, TERM_PRODUCT_12 = range(4)
TERM_XOR_12, TERM_AND_12, TERM_OR_12, TERM_CUBE_1 = range(4,8)

def term_value(t, hist):
    p1=hist[-1] if hist else 0
    p2=hist[-2] if len(hist)>1 else 0
    return (p1,p2,(p1*p1)&255,(p1*p2)&255,p1^p2,p1&p2,p1|p2,(p1*p1*p1)&255)[t]

def predict(terms, hist):
    return sum((term_value(t,hist)*c) for t,c in terms) & 255

def forward(data, terms):
    out=[]
    for i,x in enumerate(data): out.append((x-predict(terms,data[:i]))&255)
    return bytes(out)

def inverse(residual, terms):
    out=[]
    for r in residual: out.append((r+predict(terms,out))&255)
    return bytes(out)

def entropy(data):
    if not data: return 0.0
    h=[0]*256
    for b in data:h[b]+=1
    n=len(data)
    return sum(-c/n*math.log2(c/n) for c in h if c)

if __name__=='__main__':
    rng=random.Random(941)
    for n in [0,1,2,17,1024,10000]:
        data=bytes(rng.randrange(256) for _ in range(n))
        model=[(0,11),(2,7),(3,19),(4,5),(7,3)]
        assert inverse(forward(data,model),model)==data
    x=bytes((i*7+(i//3)*19)&255 for i in range(65536))
    m=[(0,1),(1,3),(2,7),(3,11),(4,17)]
    r=forward(x,m)
    print('structured sample bytes:',len(x),'source entropy:',round(entropy(x),4),'residual entropy:',round(entropy(r),4),'roundtrip:',inverse(r,m)==x)
    print('randomized roundtrip cases: PASS')
