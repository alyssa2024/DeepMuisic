import torch

from dataset import BTTSingleInstanceDataset
from loss import compute_group_profile_recon_loss
from VAE import PhysicalHarmonicVAE


def _make_dataset(seed=7):
    return BTTSingleInstanceDataset(
        patch_num_cycles=2,
        num_total_patches=10,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        num_probes=4,
        base_freq=150.0,
        fluctuation_delta=0.001,
        probe_angles=[0, 28, 111.08, 166.15],
        freq_lower=[160.0, 330.0],
        freq_upper=[180.0, 350.0],
        true_freq_hz=[167.0, 341.0],
        true_amp_real=[0.6, 0.54],
        true_amp_imag=[0.0, 0.84],
        snr_db=20,
        seed=seed,
        patch_hop_cycles=2,
        allow_patch_overlap=False,
        normalization="group_std",
        parameter_source="fixed",
    )


def test_single_instance_dataset_fixed_frequency_and_splits():
    ds_a = _make_dataset(seed=11)
    ds_b = _make_dataset(seed=11)
    assert len(ds_a) == 1
    item_a = ds_a[0]
    item_b = ds_b[0]
    assert torch.equal(item_a["true_freq_hz"], item_b["true_freq_hz"])
    assert torch.equal(item_a["train"]["x"], item_b["train"]["x"])
    assert item_a["train"]["x"].shape[0] == 6
    assert item_a["val"]["x"].shape[0] == 2
    assert item_a["test"]["x"].shape[0] == 2
    assert torch.equal(item_a["true_freq_hz"], item_a["val"].get("true_freq_hz", item_a["true_freq_hz"]))


def test_single_instance_patches_do_not_overlap_by_default():
    ds = _make_dataset(seed=13)
    t_abs = ds[0]["train"]["t_abs"]
    for p in range(t_abs.shape[0] - 1):
        assert torch.max(t_abs[p]) < torch.min(t_abs[p + 1])


def test_map_ls_solution_satisfies_normal_equation():
    class _Encoder:
        output_dim = 2

    model = PhysicalHarmonicVAE(encoder=_Encoder(), ls_ridge=0.0)
    t = torch.tensor([[0.0, 0.1, 0.2, 0.3]], dtype=torch.float32)
    f = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    c_true = torch.tensor([[0.5 + 0.2j, -0.1 + 0.3j]], dtype=torch.complex64)
    phi = model.build_dictionary(f, t)
    y = (phi * c_true.unsqueeze(1)).sum(dim=-1)
    ridge = torch.tensor([1e-3], dtype=torch.float32)
    _real, _imag, c_hat, _cond = model.solve_amplitudes_ls(
        y_complex=y,
        f=f,
        t=t,
        ridge_lambda=ridge,
        return_condition=True,
    )
    gram = phi.conj().transpose(-2, -1) @ phi
    eye = torch.eye(2, dtype=gram.dtype).unsqueeze(0)
    lhs = (gram + ridge.to(gram.dtype).view(-1, 1, 1) * eye) @ c_hat.unsqueeze(-1)
    rhs = phi.conj().transpose(-2, -1) @ y.unsqueeze(-1)
    assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_coherent_profile_loss_solves_one_global_amplitude():
    class _Encoder:
        output_dim = 2
        freq_lower = torch.tensor([0.5, 1.0], dtype=torch.float32)
        freq_upper = torch.tensor([2.0, 3.0], dtype=torch.float32)

    model = PhysicalHarmonicVAE(encoder=_Encoder(), ls_ridge=0.0)
    f = torch.tensor([[1.25, 2.2]], dtype=torch.float32)
    c_true = torch.tensor([[0.5 + 0.2j, -0.1 + 0.3j]], dtype=torch.complex64)
    t_abs = torch.tensor(
        [
            [
                [0.00, 0.07, 0.15, 0.24, 0.31, 0.42],
                [1.10, 1.18, 1.27, 1.36, 1.44, 1.55],
            ]
        ],
        dtype=torch.float32,
    )
    phi = torch.polar(
        torch.ones(1, 2, 6, 2, dtype=torch.float32),
        2.0 * torch.pi * t_abs.unsqueeze(-1) * f[:, None, None, :],
    )
    y = (phi * c_true[:, None, None, :]).sum(dim=-1)

    loss, diag = compute_group_profile_recon_loss(
        y_complex=y,
        t=t_abs,
        mu_f=f,
        std_f=torch.full_like(f, 0.01),
        model=model,
        sequence_posterior_samples=1,
        ridge_lambda=0.0,
        f_samples=f.unsqueeze(0),
        noise_var_norm=torch.ones(1, 2),
        include_log_const=False,
        profile_mode="coherent",
    )

    assert loss.item() < 1e-8
    assert diag["c_hat_samples"].shape == (1, 1, 1, 2)
    assert torch.allclose(diag["c_hat_samples"][0, 0, 0], c_true[0], atol=1e-4, rtol=1e-4)
    assert diag["profile_num_blocks"].item() == 1
