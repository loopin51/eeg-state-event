# Stage 0-a diagnosis: learning curves on TRAINING-SET data only (official Validation set is not touched).
import sys, os, numpy as np, torch, torch.nn.functional as F, json; sys.argv=['x']; sys.path.insert(0,'scripts')
import stage0_intent as S
X, meta, fs, on = S.load('p1'); y=meta['label'].to_numpy(); subj=meta['subject'].to_numpy(); sess=meta['session'].to_numpy(); X=X.astype(np.float32)
out={}
def curve(tag, tr, ho, epochs, seed=0, lr=1e-3):
    rng=np.random.default_rng(seed); net=S.make_net(64,X.shape[2],fs,seed); opt=torch.optim.Adam(net.parameters(),lr); rows=[]
    for ep in range(epochs):
        net.train()
        for b in S.batches(tr,64,rng):
            if len(b)<2: continue
            loss=F.cross_entropy(net(torch.from_numpy(X[b]).to(S.DEV)), torch.from_numpy(y[b]).to(S.DEV)); opt.zero_grad(); loss.backward(); opt.step()
        if ep%10==9 or ep<3:
            lg=S.logits(net,X,ho); hl=float(F.cross_entropy(lg,torch.from_numpy(y[ho]))); ha=float((lg.argmax(1).numpy()==y[ho]).mean())
            ta=float((S.logits(net,X,tr).argmax(1).numpy()==y[tr]).mean()); rows.append((ep+1,ta,ha,hl)); print(tag, ep+1, 'train %.3f holdout acc %.3f loss %.3f'%(ta,ha,hl), flush=True)
    out[tag]=rows
tr_all=np.where(sess==0)[0]; tr,ho=S.strat_holdout(tr_all,y,0.1,np.random.default_rng(0))
curve('pooled', tr, ho, 300)
for s in [0,5,10]:
    idx=np.where((subj==s)&(sess==0))[0]; tr,ho=S.strat_holdout(idx,y,0.2,np.random.default_rng(s)); curve(f'subj{s}', tr, ho, 300)
json.dump(out, open('results/stage0_0a_diag_curves.json','w'))
