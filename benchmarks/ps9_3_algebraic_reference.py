import numpy as np, zlib, hashlib, csv, os, time
src='/mnt/data/mixed_10mb_compression_test.bin'
data=np.frombuffer(open(src,'rb').read(),dtype=np.uint8)
fit=data[:min(4096,len(data))]
a=np.zeros(4,dtype=np.uint16)
def residual(x, coeff):
    pred=np.zeros(len(x),dtype=np.uint16)
    for j,c in enumerate(coeff):
        if j+1<len(x): pred[j+1:]=(pred[j+1:]+int(c)*x[:-(j+1)].astype(np.uint16))&255
    return ((x.astype(np.uint16)-pred)&255).astype(np.uint8)
def score(x, coeff):
    h=np.bincount(residual(x,coeff),minlength=256).astype(np.uint64)
    return int(np.dot(h,h))
start=time.time(); best=score(fit,a)
for _ in range(2):
    for j in range(4):
        win=int(a[j]); win_score=best
        for c in range(256):
            t=a.copy(); t[j]=c; v=score(fit,t)
            if v>win_score or (v==win_score and c<win): win,win_score=c,v
        a[j]=win; best=win_score
trans=residual(data,a)
# inverse recurrence
rec=np.zeros(len(data),dtype=np.uint8)
for i in range(len(data)):
    pred=0
    for j,c in enumerate(a):
        if i>j: pred=(pred+int(c)*int(rec[i-j-1]))&255
    rec[i]=(int(trans[i])+pred)&255
assert np.array_equal(rec,data)
raw=bytes(data); enc=bytes(trans)
print('size',len(raw),'sha256',hashlib.sha256(raw).hexdigest())
print('coefficients',a.tolist(),'fit_score',best)
print('zlib_original',len(zlib.compress(raw,9)))
print('zlib_residual',len(zlib.compress(enc,9)))
print('roundtrip',bool(np.array_equal(rec,data)))
print('fit_seconds',round(time.time()-start,3))
