from dataset import BTTPatchDataset, build_btt_point_features
from torch.utils.data import DataLoader
import torch
from Encoder import VariationalIndependentTimeSeriesTransformer
from VAE import PhysicalHarmonicVAE
from loss import compute_harmonic_elbo


# 1. 实例化
input_dim = 6
num_harmonics = 4
num_probes = 4

epochs = 10

encoder = VariationalIndependentTimeSeriesTransformer(
    input_dim=input_dim,
    output_dim=num_harmonics,
    hidden_dim=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=256,
    hidden_dim_dense=256,
    num_probes=num_probes,
    use_standard_pe=False,
    device=device,
)

model = PhysicalHarmonicVAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

features, t_samples, rev_ids, probe_ids = build_btt_point_features(
    x_observed=x_t_observed,
    t_samples=t_samples,
    rev_ids=rev_ids,
    probe_ids=probe_ids,
    theta_samples=theta_samples,
    freqs_at_samples=freqs_at_samples,
    base_freq=base_x,
    n_revs=n_revs,
)

dataset = BTTPatchDataset(
    features=features,
    t_samples=t_samples,
    rev_ids=rev_ids,
    probe_ids=probe_ids,
    window_revs=64,
    hop_revs=32,
    num_probes=4,
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)

# 2. 训练步
model.train()

for epoch in range(epochs):
    for x_batch, t_batch, probe_ids, rev_ids, target_batch in dataloader:
        x_batch = x_batch.to(device)           # [B, L, 6]
        t_batch = t_batch.to(device)           # [B, L]
        probe_ids = probe_ids.to(device)       # [B, L]
        target_batch = target_batch.to(device) # [B, L, 2]

        optimizer.zero_grad()

        x_hat, dist_params = model(
            x_batch,
            t_batch,
            probe_ids=probe_ids,
        )

        loss, recon, kl = compute_harmonic_elbo(
            x_target=target_batch,
            x_hat=x_hat,
            dist_params=dist_params,
            beta=1e-3,
            prior_A_mu=0.0,
            prior_A_var=1.0,
            prior_a_w=1000.0,
            prior_phi_mu=0.0,
            prior_phi_kappa=0.0,
            use_kl_A=True,
            use_kl_w=False,
            use_kl_phi=False,
        )


        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

    print(
        f"epoch={epoch:04d} "
        f"loss={loss.item():.6f} "
        f"recon={recon.item():.6f} "
        f"kl={kl.item():.6f}"
    )