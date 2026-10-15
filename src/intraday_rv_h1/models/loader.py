"""Explicit, portable construction of the archived H1 forward graph."""
import torch
from .archived.monthly_demo_attention import MonthlyDemoHARNet
from .archived.structure_ablation_attention import apply_ablation


def build_model(factory, weights, seed, *, training=False):
    model = MonthlyDemoHARNet(
        factory['config'], seed, factory['limits'], factory['descriptors'],
        factory['variant'], factory['reference_state'],
        factory['rank_values'], factory['rank_counts'],
        own_values=factory['own_values'], own_counts=factory['own_counts'],
        own_assets=factory['own_assets'], initial_weights=factory['initial_weights'],
        train_assets=[factory['asset']], numeric_mode=factory['numeric_mode'],
        split_calendar=factory['split_calendar'], local_amplitude=factory['local_amplitude'],
        return_column=factory['return_column'], return_state_count=5,
        return_state_width=3, return_input='rank', price_column=0)
    apply_ablation(model, 'linear_decoder_tanh', output_seed=seed)
    model.load_state_dict(weights, strict=True)
    return model.train() if training else model.eval().requires_grad_(False)
