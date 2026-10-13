"""ESEModel: L0 front end + L1 heads + JEPA target/predictor + subject probes, and the
Stage-A loss composition (docs/04 §6)."""
import copy, torch, torch.nn as nn, torch.nn.functional as F
from .encoder import FrontEnd
from .heads import FastHead, SlowHead, Predictor, SubjectProbe
from .losses import predict_loss, coherence_loss, stim_align_loss, xcov_loss, make_block_mask, variance_loss


class ESEModel(nn.Module):
    def __init__(self, n_subjects, d=512, layers=6, heads=8, ffn=2048, d_event=256, d_state=32,
                 n_datasets=8, d_tok=32, n_virtual=32, dropout=0.1, path_drop=0.1,
                 mask_ratio=0.5, alphas=(1.0, 1.0, 0.1, 0.1), temp=0.1, var_weight=0.5, center_in_batch=True,
                 split_timescale=True, xcov_mode='spectral', fast_highpass=None, slow_lowpass=None,
                 use_predict=True):
        super().__init__()
        self.front = FrontEnd(d, layers, heads, ffn, n_datasets, d_tok, n_virtual, dropout, path_drop)
        # D-024: the two halves of the split are separable. The high-pass on the FAST branch is what
        # decorrelates the branches; the low-pass on the SLOW branch also strips the within-window
        # dynamics the slow head used to attend to, which cost AUC/ISC. Default: high-pass only.
        hp = split_timescale if fast_highpass is None else fast_highpass
        lp = False if slow_lowpass is None else slow_lowpass
        self.fast = FastHead(d, d_event, highpass=hp)
        self.slow = SlowHead(d, d_state, lowpass_input=lp)
        self.pred = Predictor(d, 256, d_event)
        self.probe_e, self.probe_s = SubjectProbe(d_event, n_subjects), SubjectProbe(d_state, n_subjects)
        self.mask_ratio, self.alphas, self.temp, self.var_weight = mask_ratio, alphas, temp, var_weight
        self.center_in_batch, self.xcov_mode, self.use_predict = center_in_batch, xcov_mode, use_predict
        # EMA target (front + fast), never trained directly
        self.target = copy.deepcopy(nn.ModuleDict(dict(front=self.front, fast=self.fast)))
        for p in self.target.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, m):
        for (n, p), (_, q) in zip(self.target.named_parameters(),
                                  nn.ModuleDict(dict(front=self.front, fast=self.fast)).named_parameters()):
            p.mul_(m).add_(q.detach(), alpha=1 - m)

    @staticmethod
    def _subject_center(z, subject):
        """D-020: subtract each subject's in-batch mean (subjects with >=2 items). Mirrors the per-subject
        centring used by every read-out, so the contrastive geometry is defined in the same space that is
        evaluated, and the (large) per-subject offset never has to be un-learned by the encoder."""
        same = (subject[:, None] == subject[None, :]).to(z.dtype)             # [B,B]
        cnt = same.sum(1, keepdim=True)
        mean = same @ z / cnt
        return torch.where(cnt > 1, z - mean, z)

    def represent(self, x, pos, dataset_id):
        """Inference path: E(t), z_event(t), z_state(τ)."""
        E = self.front(x, pos, dataset_id)
        return E, self.fast(E), self.slow(E)

    def forward(self, batch, grl_lambda=0.0):
        x, pos, ds = batch['x'], batch['positions'], batch['dataset_id']
        B, _, L = x.shape
        merged = self.front.merge(x, pos, ds)                                     # [B,N_p,d]
        Np = merged.shape[1]
        mask = make_block_mask(B, Np, self.mask_ratio, device=x.device)
        # context (masked) path -> predictor
        if self.use_predict:
            E_ctx = self.front.encode(merged, mask)
            pred = self.pred(E_ctx)
            with torch.no_grad():
                E_t = self.target['front'](x, pos, ds); z_t = self.target['fast'](E_t)
            L_pred = predict_loss(pred, z_t, mask)
        else:
            L_pred = merged.new_zeros(())        # CLISA-style ablation: alignment objective only
        # unmasked path -> heads
        E = self.front.encode(merged)
        z_event, z_state = self.fast(E), self.slow(E)                             # [B,N_p,de], [B,N_s,ds]
        ze, zs = z_event.mean(1), z_state.mean(1)
        zc = self._subject_center(zs, batch['subject']) if self.center_in_batch else zs
        L_coh = coherence_loss(zc, batch['trial_key'], batch['pair_id'], batch['subject'], self.temp)
        L_ali = stim_align_loss(zc, batch['subject'], batch['clip_key'], batch['t_onset'], ds, self.temp)
        L_xc = xcov_loss(ze, zs, self.xcov_mode)
        L_sub = F.cross_entropy(self.probe_e(ze, grl_lambda), batch['subject']) + \
                F.cross_entropy(self.probe_s(zs, grl_lambda), batch['subject'])
        L_var = variance_loss(z_event) + variance_loss(z_state)
        a1, a2, a3, a4 = self.alphas
        total = L_pred + a1 * L_coh + a2 * L_ali + a3 * L_xc + a4 * L_sub + self.var_weight * L_var
        with torch.no_grad():
            std_e = z_event.reshape(-1, z_event.shape[-1]).float().std(0).mean().item()
            std_s = z_state.reshape(-1, z_state.shape[-1]).float().std(0).mean().item()
        return total, dict(pred=L_pred.item(), coh=L_coh.item(), align=L_ali.item(), xcov=L_xc.item(), subj=L_sub.item(),
                           var=L_var.item(), std_event=std_e, std_state=std_s, total=total.item())
