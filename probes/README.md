# Linear probes

These are ImageNet-1k linear classifiers trained on frozen backbone features.
Each `.safetensors` file holds `weight` [1000, 2C] and `bias` [1000], with the
backbone and readout recorded in its metadata. Load one with
`PrunableClassifier.from_probe` (`src/models/token_pruning/prunable_classifier.py`).
Class indices follow the sorted ImageNet WNIDs.

| File | Backbone (loaded from its original hub) | Readout | ImageNet-1k val top-1 |
| --- | --- | --- | --- |
| `dinov3_vitb16.safetensors` | `timm/vit_base_patch16_dinov3.lvd1689m` | [CLS ; GAP] | 85.25% |
| `siglip2_vitb16.safetensors` | `google/siglip2-base-patch16-384` | [MAP pool ; GAP] | 84.90% |

- **Training:** each probe was trained with the DINOv3 linear-probing protocol
  on ImageNet-1k train at 336 × 336 px, with the backbone frozen. The best
  learning rate was selected by validation top-1.
- **Readout:** GAP averages the patch tokens only. SigLIP 2 has no [CLS]
  token, so its attention-pooling (MAP) head output takes that place.

## Licences

The backbone weights are not redistributed here; they download from their
original sources under their own terms.

- **`dinov3_vitb16.safetensors`**
  - **Built with DINOv3.**
  - Distributed under the DINOv3 License; a copy is in `DINOv3_LICENSE.md`.
  - Its use is subject to that licence, including its trade-control and use
    restrictions.
- **`siglip2_vitb16.safetensors`**
  - Trained on features of the Apache-2.0-licensed SigLIP 2 weights.
  - Distributed under the Apache License 2.0; a copy is in
    `SigLIP2_LICENSE_Apache-2.0.txt`.

Both probes were trained on ImageNet-1k, whose images are available for
non-commercial research and educational use under the ImageNet terms of access.
