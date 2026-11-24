import numpy as np

N = 300
T = 120
errs = {}
for mode in [0.090909, 0.270156, 0.5, 0.75, 1.0, 'running', 'kf']:
    stats = []
    for jump in (False, True):
        errors = []
        for seed in range(N):
            rng=np.random.default_rng(seed)
            x=rng.normal(2,1,5).mean()
            P = .2
            one=[]
            for t in range(T):
                target=5 if jump and t>=50 else 2
                rng.standard_normal(50)
                z=rng.normal(target,1,5).mean()
                if mode=='running':
                    k=1/(t+2)
                elif mode=='kf':
                    Pp=P+.02
                    k=Pp/(Pp+.2)
                    P=(1-k)*Pp
                else:
                    k=mode
                x += k*(z-x)
                one.append(abs(x-target))
            errors.append(one)
        q=np.mean(errors,axis=0)
        post=q[50:]
        where=np.where(post<.3)[0]
        stats.append((q[-1],int(where[0]) if len(where) else 70,post.sum()))
    print(mode,stats[0][0],stats[1][1:])
for c in (.98, 1):
    A=np.array([[.5,c],[0,.5]])
    B=A.T
    print("rho-spectral/transient/switch",np.linalg.eigvals(A),np.linalg.norm(A,2),np.linalg.eigvals(A@B))
