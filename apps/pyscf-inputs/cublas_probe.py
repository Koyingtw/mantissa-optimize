import cupy as cp
a=cp.random.rand(512,256); b=cp.random.rand(256,384)
c=a@b; d=cp.dot(a,b); e=cp.einsum('ij,jk->ik',a,b); f=cp.matmul(cp.random.rand(4,64,64),cp.random.rand(4,64,64))
z=(a+1j*a)@(b+1j*b); cp.cuda.Device().synchronize()
