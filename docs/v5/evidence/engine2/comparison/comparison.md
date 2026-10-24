## Threshold-free ranking quality

| Model | Provenance | Dev ROC-AUC | Dev AP | Test ROC-AUC | Test AP | CPU ms/text |
|---|---|---|---|---|---|---|
| scratch_int8 | trained-from-scratch-ppaguard | 0.658 | 0.472 | 0.660 | 0.660 | 13.4 |
| roberta6l_int8 | fine-tuned-ppaguard | 0.716 | 0.535 | 0.547 | 0.571 | 150.6 |
| roberta6l_fp32 | fine-tuned-ppaguard | 0.717 | 0.537 | 0.536 | 0.565 | 294.6 |
| protectai_v2 | pretrained-external | 0.823 | 0.722 | 0.396 | 0.497 | 299.5 |
| deepset_deberta | pretrained-external | contaminated | contaminated | 0.586 | 0.602 | 200.2 |
| fmops_distilbert | pretrained-external | contaminated | contaminated | 0.808 | 0.794 | 79.9 |

## Injection class at operating thresholds (frozen test set, evaluated once)

Thresholds: `dev_*` re-selected on the dev set with the FPR-bounded rule; `card_*` from the model card; `p_gt_0.5` the model's argmax.

| Model | Threshold set | t | Precision | Recall | F1 | FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| scratch_int8 | p_gt_0.5 | 0.50 | 0.693 | 0.822 | 0.752 | 51.2% | 493 | 218 | 208 | 107 |
| scratch_int8 | dev_warn | 0.42 | 0.686 | 0.862 | 0.764 | 55.6% | 517 | 237 | 189 | 83 |
| scratch_int8 | dev_block | 0.73 | 0.673 | 0.340 | 0.452 | 23.2% | 204 | 99 | 327 | 396 |
| roberta6l_int8 | p_gt_0.5 | 0.50 | 0.613 | 0.470 | 0.532 | 41.8% | 282 | 178 | 248 | 318 |
| roberta6l_int8 | dev_warn | 0.49 | 0.614 | 0.493 | 0.547 | 43.7% | 296 | 186 | 240 | 304 |
| roberta6l_int8 | dev_block | 0.82 | 0.000 | 0.000 | 0.000 | 0.5% | 0 | 2 | 424 | 600 |
| roberta6l_fp32 | p_gt_0.5 | 0.50 | 0.606 | 0.435 | 0.506 | 39.9% | 261 | 170 | 256 | 339 |
| roberta6l_fp32 | dev_warn | 0.50 | 0.606 | 0.435 | 0.506 | 39.9% | 261 | 170 | 256 | 339 |
| roberta6l_fp32 | dev_block | 0.82 | 0.000 | 0.000 | 0.000 | 0.7% | 0 | 3 | 423 | 600 |
| protectai_v2 | p_gt_0.5 | 0.50 | 0.416 | 0.193 | 0.264 | 38.3% | 116 | 163 | 263 | 484 |
| protectai_v2 | dev_warn | 0.01 | 0.458 | 0.320 | 0.377 | 53.3% | 192 | 227 | 199 | 408 |
| protectai_v2 | dev_block | 0.99 | 0.399 | 0.122 | 0.186 | 25.8% | 73 | 110 | 316 | 527 |
| deepset_deberta | card_warn | 0.50 | 0.628 | 1.000 | 0.771 | 83.6% | 600 | 356 | 70 | 0 |
| deepset_deberta | card_block | 0.50 | 0.628 | 1.000 | 0.771 | 83.6% | 600 | 356 | 70 | 0 |
| deepset_deberta | p_gt_0.5 | 0.50 | 0.628 | 1.000 | 0.771 | 83.6% | 600 | 356 | 70 | 0 |
| fmops_distilbert | card_warn | 0.50 | 0.631 | 1.000 | 0.774 | 82.4% | 600 | 351 | 75 | 0 |
| fmops_distilbert | card_block | 0.50 | 0.631 | 1.000 | 0.774 | 82.4% | 600 | 351 | 75 | 0 |
| fmops_distilbert | p_gt_0.5 | 0.50 | 0.631 | 1.000 | 0.774 | 82.4% | 600 | 351 | 75 | 0 |

## Per-subset behaviour on the test set (WARN-level threshold)

| Model | Subset | n | Injections caught | Benign flagged | Subset ROC-AUC |
|---|---|---|---|---|---|
| scratch_int8 | bipia_code | 200 | 112/150 | 20/50 | 0.723 |
| scratch_int8 | bipia_email | 200 | 105/150 | 36/50 | 0.432 |
| scratch_int8 | bipia_table | 400 | 300/300 | 100/100 | 0.431 |
| scratch_int8 | notinject_three | 113 | 0/0 | 47/113 | – |
| scratch_int8 | notinject_two | 113 | 0/0 | 34/113 | – |
| roberta6l_int8 | bipia_code | 200 | 114/150 | 33/50 | 0.542 |
| roberta6l_int8 | bipia_email | 200 | 26/150 | 21/50 | 0.355 |
| roberta6l_int8 | bipia_table | 400 | 156/300 | 56/100 | 0.483 |
| roberta6l_int8 | notinject_three | 113 | 0/0 | 48/113 | – |
| roberta6l_int8 | notinject_two | 113 | 0/0 | 28/113 | – |
| roberta6l_fp32 | bipia_code | 200 | 107/150 | 34/50 | 0.525 |
| roberta6l_fp32 | bipia_email | 200 | 22/150 | 15/50 | 0.360 |
| roberta6l_fp32 | bipia_table | 400 | 132/300 | 47/100 | 0.486 |
| roberta6l_fp32 | notinject_three | 113 | 0/0 | 49/113 | – |
| roberta6l_fp32 | notinject_two | 113 | 0/0 | 25/113 | – |
| protectai_v2 | bipia_code | 200 | 0/150 | 0/50 | 0.435 |
| protectai_v2 | bipia_email | 200 | 117/150 | 41/50 | 0.462 |
| protectai_v2 | bipia_table | 400 | 75/300 | 57/100 | 0.276 |
| protectai_v2 | notinject_three | 113 | 0/0 | 68/113 | – |
| protectai_v2 | notinject_two | 113 | 0/0 | 61/113 | – |
| deepset_deberta | bipia_code | 200 | 150/150 | 50/50 | 0.648 |
| deepset_deberta | bipia_email | 200 | 150/150 | 50/50 | 0.795 |
| deepset_deberta | bipia_table | 400 | 300/300 | 100/100 | 0.755 |
| deepset_deberta | notinject_three | 113 | 0/0 | 100/113 | – |
| deepset_deberta | notinject_two | 113 | 0/0 | 56/113 | – |
| fmops_distilbert | bipia_code | 200 | 150/150 | 50/50 | 0.571 |
| fmops_distilbert | bipia_email | 200 | 150/150 | 50/50 | 0.675 |
| fmops_distilbert | bipia_table | 400 | 300/300 | 100/100 | 0.749 |
| fmops_distilbert | notinject_three | 113 | 0/0 | 98/113 | – |
| fmops_distilbert | notinject_two | 113 | 0/0 | 53/113 | – |

## Dev set (thresholds chosen here)

| Model | Threshold set | t | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| scratch_int8 | dev_warn | 0.42 | 0.494 | 0.217 | 0.301 | 9.9% |
| scratch_int8 | dev_block | 0.73 | 0.765 | 0.064 | 0.118 | 0.9% |
| scratch_int8 | p_gt_0.5 | 0.50 | 0.571 | 0.197 | 0.293 | 6.6% |
| roberta6l_int8 | dev_warn | 0.49 | 0.638 | 0.365 | 0.464 | 9.2% |
| roberta6l_int8 | dev_block | 0.82 | 1.000 | 0.005 | 0.010 | 0.0% |
| roberta6l_int8 | p_gt_0.5 | 0.50 | 0.637 | 0.355 | 0.456 | 9.0% |
| roberta6l_fp32 | dev_warn | 0.50 | 0.633 | 0.374 | 0.471 | 9.6% |
| roberta6l_fp32 | dev_block | 0.82 | 1.000 | 0.010 | 0.020 | 0.0% |
| roberta6l_fp32 | p_gt_0.5 | 0.50 | 0.633 | 0.374 | 0.471 | 9.6% |
| protectai_v2 | dev_warn | 0.01 | 0.748 | 0.498 | 0.598 | 7.5% |
| protectai_v2 | dev_block | 0.99 | 0.802 | 0.360 | 0.497 | 3.9% |
| protectai_v2 | p_gt_0.5 | 0.50 | 0.750 | 0.429 | 0.545 | 6.4% |
| deepset_deberta | – | – | trained on the dev set's source | | | |
| fmops_distilbert | – | – | trained on the dev set's source | | | |
