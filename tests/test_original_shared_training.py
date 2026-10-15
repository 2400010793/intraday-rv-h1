"""Original graph reconstruction must preserve formulas and parameter flags."""
from pathlib import Path
import numpy as np
import torch
from intraday_rv_h1.models.loader import build_model
from intraday_rv_h1.research.original_shared_inputs import OriginalInputs


def test_training_toggle_keeps_original_forward():
    root=Path('artifacts/models/IF');source=Path('/public_space_2/chenxy/volatility')
    if not root.exists():
        import pytest
        pytest.skip('requires local original model assets')
    torch.set_num_threads(1)
    f=torch.load(root/'factory.pt',map_location='cpu',weights_only=True)
    checkpoint=source/'results/intraday_rv/notebook10-onds-three-origin-monthly-adam-shampoo-e6-20261014-v1/tasks/adam/2026-08/IF__seed7/checkpoint_00.pt'
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    frozen=build_model(f,p['model'],7);trainable=build_model(f,p['model'],7,training=True)
    data=OriginalInputs(source,'IF','2026-08');args=data.batch('score',np.array([0,1,120,240]))
    trainable.eval()
    with torch.no_grad():
        for a,b in zip(frozen(*args),trainable(*args)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not any(p.requires_grad for p in frozen.parameters())
    assert any(p.requires_grad and 'average_layers' in n for n,p in trainable.named_parameters())
    loss=trainable.loss(args,torch.tensor(data.frame.y_true.iloc[[0,1,120,240]].to_numpy()),.007513092011263302)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trainable.parameters() if p.requires_grad)
