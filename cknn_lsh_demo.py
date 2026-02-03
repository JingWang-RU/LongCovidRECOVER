
import os, math, random, time, json
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import hnswlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------------------
# 1) Synthetic sequential data
# ------------------------------

def make_synthetic_longitudinal(N=1000, T=5, d_x=6, seed=1337):
    """
    Generate synthetic longitudinal data with confounding.
    Each patient i has time-varying covariates X_{i,t}, treatment A_{i,t} in {0,1},
    and outcome Y_{i,t}. Treatment depends on history; outcomes depend on both
    latent health and treatment.
    """
    rng = np.random.default_rng(seed)
    # latent "health" state per patient, evolves
    Z = rng.normal(0, 1, size=(N,))  # baseline health
    
    X = []
    A = []
    Y = []
    # create d_x covariates; first few are confounders linked to Z and past Y
    for i in range(N):
        Xi = []
        Ai = []
        Yi = []
        z = Z[i]
        y_prev = 0.0
        for t in range(T):
            # covariates: some depend on z and y_prev
            x = rng.normal(0,1,size=(d_x,))
            x[0] += 0.8*z
            x[1] += 0.6*y_prev
            x[2] += 0.4*z*y_prev
            
            # clinician policy: treat if (z + y_prev + x0) high, plus noise
            logits = 0.8*z + 0.7*y_prev + 0.5*x[0] + rng.normal(0,0.5)
            a = (logits > 0.0).astype(int)
            
            # outcome model: future health depends on current treatment and covariates
            # true treatment effect heterogeneity: beneficial if z is low (sicker)
            tau = 0.8 - 0.6*max(z, 0)  # smaller effect for healthier
            noise = rng.normal(0, 0.5)
            y = (0.6*y_prev + 0.5*x[0] + 0.3*x[1] + tau*a + 0.3*z + noise)
            
            Xi.append(x.astype(np.float32))
            Ai.append(int(a))
            Yi.append(float(y))
            
            y_prev = y
        
        X.append(np.array(Xi, dtype=np.float32))  # (T, d_x)
        A.append(np.array(Ai, dtype=np.int64))    # (T,)
        Y.append(np.array(Yi, dtype=np.float32))  # (T,)
    return np.array(X, dtype=object), np.array(A, dtype=object), np.array(Y, dtype=object)

# -----------------------------------------
# 2) Models: GRU encoder, outcome, policy
# -----------------------------------------

class GRUEncoder(nn.Module):
    def __init__(self, d_x, d_hidden=64, d_latent=16):
        super().__init__()
        self.gru = nn.GRU(input_size=d_x+2, hidden_size=d_hidden, batch_first=True)
        self.proj = nn.Linear(d_hidden, d_latent)
    def forward(self, X, A, Y):
        """
        X: (B,T,d_x), A: (B,T), Y: (B,T)
        Build history embedding for each time step t using prefix 0..t-1.
        We do teacher-forcing: shift A,Y by one with zeros at t=0.
        Returns embeddings Phi of shape (B,T,d_latent).
        """
        B,T,d_x = X.shape
        A_shift = torch.zeros_like(A)
        Y_shift = torch.zeros_like(Y)
        if T>1:
            A_shift[:,1:] = A[:,:-1].float()
            Y_shift[:,1:] = Y[:,:-1].float()
        inp = torch.cat([X, A_shift.unsqueeze(-1), Y_shift.unsqueeze(-1)], dim=-1)
        h, _ = self.gru(inp)  # (B,T,d_hidden)
        phi = self.proj(h)    # (B,T,d_latent)
        return phi

class MLP(nn.Module):
    def __init__(self, d_in, d_out=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)

def outcome_forward(outcome_net, phi, a):
    """
    phi: (B,T,d_latent), a: (B,T) int {0,1}
    returns preds (B,T)
    """
    B,T,d = phi.shape
    a_one = F.one_hot(a, num_classes=2).float()  # (B,T,2)
    x = torch.cat([phi, a_one], dim=-1).view(B*T, d+2)
    y = outcome_net(x).view(B,T)
    return y

def propensity_forward(prop_net, phi):
    """
    phi: (B,T,d_latent)
    returns p(a=1 | phi) in (B,T)
    """
    B,T,d = phi.shape
    x = phi.reshape(B*T, d)
    logit = prop_net(x).view(B,T)
    p1 = torch.sigmoid(logit)
    return p1

# -------------------------------
# 3) Losses: predictive + balance
# -------------------------------

def mmd_loss(res_treated, res_control, sigma=1.0):
    """Simple RBF-kernel MMD between 1D residual samples."""
    # res_*: (n,)
    def rbf(x, y):
        xx = x.unsqueeze(1)
        yy = y.unsqueeze(0)
        return torch.exp(-(xx-yy)**2/(2*sigma**2))
    Ktt = rbf(res_treated, res_treated).mean()
    Kcc = rbf(res_control, res_control).mean()
    Ktc = rbf(res_treated, res_control).mean()
    return Ktt + Kcc - 2*Ktc

# --------------------------------------------
# 4) Training with predictive + balancing loss
# --------------------------------------------

@dataclass
class TrainConfig:
    d_x: int = 6
    T: int = 5
    d_hidden: int = 64
    d_latent: int = 16
    hidden: int = 64
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    lam_mmd: float = 0.1
    seed: int = 1337

def train_cknn_lsh(X, A, Y, cfg: TrainConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    N = len(X)
    idx = np.arange(N)
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=cfg.seed)
    
    # pack into tensors
    def pack(idxs):
        # pad to uniform T if variable-length; here T is fixed
        Xb = torch.tensor(np.stack([X[i] for i in idxs], axis=0)) # (B,T,d_x)
        Ab = torch.tensor(np.stack([A[i] for i in idxs], axis=0)) # (B,T)
        Yb = torch.tensor(np.stack([Y[i] for i in idxs], axis=0)) # (B,T)
        return Xb, Ab, Yb
    Xtr, Atr, Ytr = pack(tr_idx)
    Xte, Ate, Yte = pack(te_idx)
    
    enc = GRUEncoder(cfg.d_x, cfg.d_hidden, cfg.d_latent)
    out_net = MLP(cfg.d_latent+2, 1, cfg.hidden)
    prop_net = MLP(cfg.d_latent, 1, cfg.hidden)
    params = list(enc.parameters()) + list(out_net.parameters()) + list(prop_net.parameters())
    opt = optim.Adam(params, lr=cfg.lr)
    
    B = Xtr.shape[0]
    steps_per_epoch = math.ceil(B / cfg.batch_size)
    
    for epoch in range(cfg.epochs):
        perm = torch.randperm(B)
        Xtr_sh = Xtr[perm]; Atr_sh = Atr[perm]; Ytr_sh = Ytr[perm]
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            s = step*cfg.batch_size; e = min((step+1)*cfg.batch_size, B)
            x = Xtr_sh[s:e]; a = Atr_sh[s:e]; y = Ytr_sh[s:e]
            phi = enc(x,a,y)               # (b,T,d_latent)
            yhat = outcome_forward(out_net, phi, a)  # (b,T)
            p1 = propensity_forward(prop_net, phi)   # (b,T)
            
            # Predictive loss (MSE + BCE for observed actions)
            loss_pred = F.mse_loss(yhat, y)
            # action loglik
            logp = a.float()*torch.log(p1+1e-6) + (1-a).float()*torch.log(1-p1+1e-6)
            loss_act = -logp.mean()
            
            # Balancing: MMD of residuals across treatment groups in the minibatch
            res = (y - yhat).detach()
            # pool across time
            a_vec = a.reshape(-1)
            res_vec = res.reshape(-1)
            if (a_vec==1).sum()>1 and (a_vec==0).sum()>1:
                mmd = mmd_loss(res_vec[a_vec==1], res_vec[a_vec==0])
            else:
                mmd = torch.tensor(0.0)
            
            loss = loss_pred + 0.1*loss_act + cfg.lam_mmd*mmd
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}/{cfg.epochs}] loss={epoch_loss/steps_per_epoch:.4f}")
    
    # Build embeddings for train and test
    with torch.no_grad():
        phi_tr = enc(Xtr, Atr, Ytr).numpy()   # (n_tr,T,d_latent)
        phi_te = enc(Xte, Ate, Yte).numpy()   # (n_te,T,d_latent)
        # also store predictions for outcome models
        yhat_tr = outcome_forward(out_net, torch.tensor(phi_tr), Atr).numpy()
        p1_tr = propensity_forward(prop_net, torch.tensor(phi_tr)).numpy()
    
    model = {"enc": enc, "out_net": out_net, "prop_net": prop_net}
    data = {"Xtr": Xtr, "Atr": Atr, "Ytr": Ytr, "Xte": Xte, "Ate": Ate, "Yte": Yte,
            "phi_tr": phi_tr, "phi_te": phi_te, "yhat_tr": yhat_tr, "p1_tr": p1_tr,
            "tr_idx": tr_idx, "te_idx": te_idx}
    return model, data

# ------------------------------------
# 5) ANN index and DR-kNN estimator
# ------------------------------------

class ANNIndex:
    def __init__(self, dim, space="l2", ef=100, M=32):
        self.index = hnswlib.Index(space=space, dim=dim)
        self.ef = ef
        self.M = M
        self.built = False
    
    def build(self, X, ids=None):
        num = X.shape[0]
        if ids is None:
            ids = np.arange(num)
        self.index.init_index(max_elements=num, ef_construction=self.ef, M=self.M)
        self.index.add_items(X.astype(np.float32), ids.astype(np.int64))
        self.index.set_ef(self.ef)
        self.built = True
    
    def query(self, q, k=50):
        labels, dists = self.index.knn_query(q.astype(np.float32), k=k)
        return labels, dists

def flatten_time(phi, A, Y):
    """
    Flatten sequences into visit-level rows.
    phi: (n,T,d), A,Y: (n,T)
    returns arrays of shape (n*T, d), (n*T,), (n*T,)
    """
    n,T,d = phi.shape
    return phi.reshape(n*T, d), A.reshape(n*T), Y.reshape(n*T)

def dr_knn_mu(enc, out_net, prop_net, phi_tr, A_tr, Y_tr, query_phi, a_query, k=50, temp=1.0):
    """
    Estimate mu(a|H) for each query row using DR-kNN.
    - Build separate ANN per treatment value for clarity.
    """
    # Split train by treatment
    mask_a = (A_tr == a_query)
    phi_a = phi_tr[mask_a]
    Y_a = Y_tr[mask_a]
    
    # ANN on phi_a
    dim = phi_tr.shape[1]
    ann = ANNIndex(dim=dim)
    ann.build(phi_a, np.arange(len(phi_a)))
    
    # outcome and propensity on train
    with torch.no_grad():
        # outcome for treatment a on train
        phi_tr_t = torch.tensor(phi_tr, dtype=torch.float32)
        a_tr = torch.tensor(A_tr, dtype=torch.int64)
        yhat_tr = outcome_forward(out_net, phi_tr_t.view(1,-1,dim), a_tr.view(1,-1)).view(-1).numpy()
        # propensity on train
        p1_tr = propensity_forward(prop_net, phi_tr_t.view(1,-1,dim)).view(-1).numpy()
        # choose propensity for a
        e_tr = p1_tr if a_query==1 else (1-p1_tr)
    
    # Precompute yhat for treatment a on train rows
    # yhat under a differs; recompute with a vector of a's
    with torch.no_grad():
        B = phi_tr.shape[0]
        # create a vector a* of length n_tr*T
        a_star = np.full((phi_tr.shape[0],), a_query, dtype=np.int64)
    
    # For simplicity at visit-level, we treat query_phi as (Q,d)
    labels, dists = ann.query(query_phi, k=min(k, len(phi_a)))
    # kernel weights
    W = np.exp(-dists / max(temp, 1e-6))
    W /= (W.sum(axis=1, keepdims=True) + 1e-8)
    
    mu_hat = np.zeros(query_phi.shape[0], dtype=np.float32)
    for i in range(query_phi.shape[0]):
        idxs = labels[i]                             # indices within phi_a
        # Map back to global indices of train where A=a
        global_idxs = np.where(mask_a)[0][idxs]
        w = W[i]
        # Compute outcome regression for neighbors under a_query
        with torch.no_grad():
            phi_neighbors = torch.tensor(phi_tr[global_idxs], dtype=torch.float32).view(1,-1,dim)
            a_neighbors = torch.full((1,len(global_idxs)), a_query, dtype=torch.int64)
            m_neighbors = outcome_forward(out_net, phi_neighbors, a_neighbors).view(-1).numpy()
        # residuals for neighbors (observed a == a_query)
        res = Y_tr[global_idxs] - m_neighbors
        # propensity weights (stabilized)
        with torch.no_grad():
            p1_neighbors = propensity_forward(prop_net, torch.tensor(phi_tr[global_idxs]).view(1,-1,dim)).view(-1).numpy()
        e_neighbors = p1_neighbors if a_query==1 else (1-p1_neighbors)
        iw = 1.0 / np.clip(e_neighbors, 1e-3, 1-1e-3)
        w_tilde = w * iw
        w_tilde = w_tilde / (w_tilde.sum() + 1e-8)
        
        # outcome regression at query
        with torch.no_grad():
            q_phi = torch.tensor(query_phi[i]).view(1,1,dim)
            q_a = torch.tensor([[a_query]], dtype=torch.int64)
            m_q = outcome_forward(out_net, q_phi, q_a).item()
        
        mu_hat[i] = m_q + np.sum(w_tilde * res)
    return mu_hat

# ---------------------------------
# 6) Main demo: train & evaluate
# ---------------------------------

def main():
    X, A, Y = make_synthetic_longitudinal(N=800, T=5, d_x=6, seed=1337)
    cfg = TrainConfig(epochs=8, batch_size=64, d_x=6, T=5, d_latent=16, lam_mmd=0.1)
    model, data = train_cknn_lsh(X, A, Y, cfg)
    
    enc = model["enc"]; out_net = model["out_net"]; prop_net = model["prop_net"]
    phi_tr, phi_te = data["phi_tr"], data["phi_te"]
    
    # Flatten to visit-level
    phi_tr_flat, A_tr_flat, Y_tr_flat = flatten_time(phi_tr, data["Atr"].numpy(), data["Ytr"].numpy())
    phi_te_flat, A_te_flat, Y_te_flat = flatten_time(phi_te, data["Ate"].numpy(), data["Yte"].numpy())
    
    # Estimate mu(a|H) on test for a in {0,1}
    mu0 = dr_knn_mu(enc, out_net, prop_net, phi_tr_flat, A_tr_flat, Y_tr_flat, phi_te_flat, a_query=0, k=50, temp=1.0)
    mu1 = dr_knn_mu(enc, out_net, prop_net, phi_tr_flat, A_tr_flat, Y_tr_flat, phi_te_flat, a_query=1, k=50, temp=1.0)
    
    # Simple evaluation proxy: If observed A_te==1, compare Y to mu1; else compare to mu0.
    y_pred_obs = np.where(A_te_flat==1, mu1, mu0)
    mse = mean_squared_error(Y_te_flat, y_pred_obs)
    print(f"Test MSE against observed-treatment counterfactual: {mse:.4f}")
    
    # Simple "policy": treat if mu1 > mu0
    policy = (mu1 > mu0).astype(int)
    # Off-policy DR value estimate on test (single-step approximation)
    # Here we just use observed one-step outcomes; for multi-step, apply sequential DR.
    # Propensity on test:
    with torch.no_grad():
        dim = phi_te.shape[-1]
        p1_te = torch.sigmoid(model["prop_net"](torch.tensor(phi_te).view(-1,dim))).view(-1).numpy()
    e_te = np.where(A_te_flat==1, p1_te, 1-p1_te)
    with torch.no_grad():
        # outcome regression at observed A
        B = phi_te_flat.shape[0]
        a_obs = torch.tensor(A_te_flat.reshape(1,-1), dtype=torch.int64)
        m_obs = outcome_forward(out_net, torch.tensor(phi_te_flat).view(1,-1,dim), a_obs).view(-1).numpy()
    dr_value = np.mean( (policy==A_te_flat)/np.clip(e_te,1e-3,1-1e-3) * (Y_te_flat - m_obs) + 
                        ( (policy)*(mu1) + (1-policy)*(mu0) ) )
    print(f"Off-policy DR value (approx.): {dr_value:.4f}")
    
if __name__ == "__main__":
    main()
