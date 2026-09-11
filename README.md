# Two-Tower Recommendation System with Cold-Start

A two-tower neural recommender trained on the MovieLens 1M dataset, built with PyTorch. The user tower fuses a learned user-ID embedding with demographic features (gender, age, occupation) and a self-attention-pooled watch history; the item tower is initialized from content features (TF-IDF over genres + a lexicon-based emotion profile) so it can represent items with little or no interaction data. The notebook covers training, standard ranking/classification evaluation, and dedicated cold-start experiments for both new items and new users.

## Contents

- [`two-tower-recommendation-system-with-cold-start.ipynb`](two-tower-recommendation-system-with-cold-start.ipynb) — the full pipeline: data loading, feature engineering, model, training, evaluation, cold-start analysis, and a standalone model-loading cell.
- `dataset/` — MovieLens 1M (`ratings.dat`, `users.dat`, `movies.dat`); not tracked in git (see `.gitignore`).

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): 1,000,209 ratings from 6,040 users on 3,883 movies. After filtering to items present in both the ratings and metadata (and users with ≥3 ratings), training uses 3,706 items and 6,040 users, expanded into 994,169 next-item-prediction training examples via a sliding window over each user's watch history.

## Model architecture

`TwoTowerWithAttention` (PyTorch `nn.Module`):

- **User tower**: a learned per-user ID embedding is concatenated with small gender/age/occupation embeddings and projected back to `embed_dim`. The user's watch history (up to 20 items, left-padded) is embedded, passed through self-attention to let items in the sequence attend to each other, then mean-pooled. A cross-attention step fuses the pooled history back into the user representation, followed by an MLP projection and layer norm.
- **Item tower**: items are initialized from a 30-dim content feature vector (20 TF-IDF genre features + 10 lexicon-based emotion scores derived from genre text), projected into `embed_dim`. The embedding table is fine-tuned during training (`freeze=False`), so it starts from content but adapts to interaction data — this is what lets a never-seen item still get a reasonable representation from its content features alone.
- **Training objective**: in-batch softmax cross-entropy (sampled softmax / in-batch negatives) — each batch's other targets act as negatives for a given user.

### Hyperparameters used

| | |
|---|---|
| Embedding dim | 128 |
| Attention heads | 4 |
| Demographic embedding dim | 8 |
| Max history length | 20 |
| Batch size | 128 |
| Optimizer | AdamW, lr=1e-3, weight_decay=0.01 |
| LR schedule | ReduceLROnPlateau (mode=max, patience=1, factor=0.5) |
| Epochs | 10 |
| Trainable parameters | ~1.09M |

## Results

### Training (in-batch hit rate)

Best validation Hit Rate@10 (against ~127 in-batch negatives) reached **0.6752** after 10 epochs, checkpointed whenever it improved. This number is an easier, faster training-time signal — not directly comparable to the full-catalog numbers below.

### Full-catalog ranking evaluation (leave-one-out, all 3,706 items)

| Metric | Two-Tower | Popularity baseline |
|---|---|---|
| HR@1 | 0.0192 | 0.0055 |
| HR@5 | 0.0851 | 0.0209 |
| HR@10 | 0.1435 | 0.0376 |
| HR@20 | 0.2320 | 0.0687 |
| HR@50 | 0.3811 | 0.1411 |
| NDCG@10 | 0.0707 | — |
| MRR | 0.0628 | 0.0201 |
| Median rank | 97 / 3,706 | — |

The model beats a pure popularity baseline by ~3–4x at every cutoff, confirming it's using personalization signal rather than just surfacing generically popular movies. Absolute numbers are modest because this ranks against the *entire* catalog, not a small negative sample.

### Classification framing (1 true item vs. 100 random negatives)

| Metric | Value |
|---|---|
| ROC-AUC (pooled) | 0.8813 |
| ROC-AUC (mean/user) | 0.8903 |
| PR-AUC | 0.1682 (positive rate 1%) |
| F1 @ best threshold | 0.2592 |
| Precision / Recall | 0.2355 / 0.2881 |
| Balanced accuracy | 0.6394 |
| Raw accuracy | 0.9837 (misleading — 100:1 class imbalance) |

High ROC-AUC shows strong pairwise separability between watched and random unwatched items; the lower PR-AUC/F1 reflect the severe class imbalance and confirm the model is better suited to *ranking* (top-K retrieval) than to a fixed accept/reject threshold — which is also how it's meant to be served.

### Cold-start: items

Items with ≤5 training occurrences (331 of 3,706) were evaluated using a content-only embedding (no learned interaction signal) vs. the fully trained embedding:

| Catalog used | HR@10 | NDCG@10 | MRR | Median rank |
|---|---|---|---|---|
| Content-only (cold item, no history) | 0.1429 | 0.1429 | 0.1455 | 647 |
| Hybrid (cold → content, warm → trained; serving-realistic) | 0.1429 | 0.0714 | 0.0482 | 1,151 |
| Trained catalog (reference, if item had been warm) | 0.2857 | 0.1981 | 0.1845 | 26 |
| Warm items (reference) | 0.1595 | 0.0778 | 0.0675 | 82 |

Content features alone give a brand-new item a real, non-random shot at being recommended (HR@10 comparable to warm items), even though it never received training gradients directly — this is the core cold-start value proposition of initializing the item tower from content.

### Cold-start: users

Ranking quality as a function of available watch history (`hist_len`), with a known user ID + demographics:

| History length | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| 0 | 0.0325 | 0.0148 | 0.0163 |
| 1 | 0.0046 | 0.0018 | 0.0035 |
| 2 | 0.0129 | 0.0057 | 0.0072 |
| 3 | 0.0200 | 0.0088 | 0.0098 |
| 5 | 0.0382 | 0.0170 | 0.0168 |
| 10 | 0.0733 | 0.0329 | 0.0308 |
| 20 (full) | 0.1435 | 0.0707 | 0.0628 |

And simulating a completely unseen user ID (mean user embedding, i.e. no learned identity signal at all):

| History length | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| 0 (pure demographic cold-start) | 0.0065 | 0.0024 | 0.0044 |
| 5 | 0.0252 | 0.0109 | 0.0116 |
| 20 | 0.1184 | 0.0572 | 0.0514 |

Quality degrades smoothly as history shrinks, and the gap between "known ID" and "unknown ID" rows at the same history length is the value contributed by the learned per-user embedding specifically. Demographics + a handful of watched items already recover a meaningful fraction of full performance, which is the practical cold-start case for a brand-new user.

## Saved artifacts

Training checkpoints the best model (by validation hit rate) to `/kaggle/working/artifacts/`:

- `best_two_tower_model.pth` — model `state_dict` plus config (`embed_dim`, `num_heads`, `demo_dim`, encoder cardinalities, epoch, hit rate)
- `item_matrix.pth` — the (fine-tuned) item content/embedding matrix
- `encoders.pkl` — the fitted `LabelEncoder`s for user ID, item ID, gender, age, and occupation

## Requirements

- Python 3, PyTorch, scikit-learn, pandas, numpy, tqdm

- This notebook was authored for a Kaggle environment (`/kaggle/input`, `/kaggle/working`).
