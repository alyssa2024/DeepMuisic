import torch


DATASET_STATE_KEYS = [
    "dataset_mode",
    "split_id",
    "use_long_sequence",
    "chronological_split",
    "num_param_sets",
    "sequences_per_param",
    "param_group_id",
    "parent_sequence_id",
    "window_start_cycle",
    "window_hop_cycles",
    "short_num_cycles",
    "long_sequence_num_cycles",
    "train_ratio_x10000",
    "val_ratio_x10000",
    "is_windowed",
    "is_iid_sequence",
]


def extract_dataset_state(batch, device):
    state = {}
    for key in DATASET_STATE_KEYS:
        if key not in batch:
            continue

        value = batch[key]
        if torch.is_tensor(value):
            state[key] = value.to(device)
        else:
            state[key] = torch.as_tensor(value, device=device)

    return state
