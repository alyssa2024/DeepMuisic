import torch

from dataset import BTTSingleInstanceDataset
from VAE import PhysicalHarmonicVAE


def _make_dataset(seed=7):
    return BTTSingleInstanceDataset(
        patch_num_cycles=2,
        num_train_patches=3,
        num_val_patches=1,
        num_test_patches=1,
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
    assert item_a["train"]["x"].shape[0] == 3
    assert item_a["val"]["x"].shape[0] == 1
    assert item_a["test"]["x"].shape[0] == 1
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
