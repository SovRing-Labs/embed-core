"""EMBED-CORE-0 E6 fit (runs on the seat's pod cpu16): token table B starts at the distilled table T; minimise mean(1 − cos(X_fit · B, Y_fit)) +
λ‖B − T‖² / V; report held-out cosine per corpus every 25 steps (the honest number); save the table as f16. Usage: fit.py LAMBDA STEPS THREADS"""
import json, sys, time
import numpy as np, torch
lam, steps, th = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); torch.set_num_threads(th)
z = np.load("fitset.npz"); T = np.load("token_table_f32.npy")
X = torch.sparse_csr_tensor(torch.tensor(z["indptr"]), torch.tensor(z["indices"]), torch.tensor(z["data"]), size=tuple(z["shape"]))
Y = torch.nn.functional.normalize(torch.tensor(z["Y"]), dim=1); fit = torch.tensor(z["fit"]); corpus = torch.tensor(z["corpus"])
fi = torch.nonzero(fit).ravel(); ho = torch.nonzero(~fit).ravel()
B = torch.tensor(T.copy(), requires_grad=True); T0 = torch.tensor(T); opt = torch.optim.Adam([B], lr=2e-3); t0 = time.time(); log = []
def cos_rows(idx):
    with torch.no_grad():
        P = torch.nn.functional.normalize(X @ B, dim=1); c = (P * Y).sum(1)
        return {n: round(float(c[idx][corpus[idx] == k].mean()), 4) for k, n in ((0, "vault"), (1, "sessions"))}
for s in range(steps):
    opt.zero_grad(); P = torch.nn.functional.normalize(X @ B, dim=1)
    loss = (1 - (P[fi] * Y[fi]).sum(1)).mean() + lam * ((B - T0) ** 2).sum() / B.shape[0]
    loss.backward(); opt.step()
    if s % 25 == 0 or s == steps - 1:
        r = {"step": s, "loss": round(loss.item(), 4), "fit_cos": cos_rows(fi), "heldout_cos": cos_rows(ho), "secs": round(time.time() - t0)}; log.append(r); print(json.dumps(r), flush=True)
np.save(f"table_l{lam}_s{steps}.npy", B.detach().numpy().astype(np.float16)); json.dump(log, open(f"fitlog_l{lam}_s{steps}.json", "w"), indent=1)
