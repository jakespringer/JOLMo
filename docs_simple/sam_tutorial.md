I read the paper. One small terminology note: the paper introduces **SAM** and later discusses **“m-sharpness”** (the case where SAM is computed on sub-batches of size (m), e.g., per accelerator and then averaged). It does **not** give a separate optimizer name “m-SAM,” but your phrasing is a very reasonable shorthand for that practical variant. 

### Core idea (SAM)

SAM optimizes a *worst-case local loss* around the current weights, instead of just the loss at the current point:

[
L^{SAM}*S(w) = \max*{|\epsilon|_p \le \rho} L_S(w+\epsilon)
]

and then minimizes this w.r.t. (w) (plus optional weight decay). In practice, the paper linearizes the inner maximization and gets a closed-form perturbation (\hat\epsilon(w)), then computes the gradient at the perturbed weights (w+\hat\epsilon(w)). 

For the common case (p=2), the paper notes this simplifies to **rescaling the gradient to norm (\rho)**. 

---

## What “m-SAM” means in practice (from the paper’s m-sharpness discussion)

The paper explicitly says that in multi-accelerator training, they:

1. split the batch across accelerators,
2. compute a SAM gradient **independently on each sub-batch**,
3. average the resulting SAM gradients. 

They then explain this is equivalent to changing the SAM objective so the inner maximization is done **separately on disjoint subsets of size (m)** (the per-accelerator sub-batch size), which they connect to their **m-sharpness** notion. 

So “m-SAM” = **per-sub-batch SAM + average gradients**.

---

## Pseudocode for m-SAM (practical distributed/sub-batch version)

Below is framework-agnostic pseudocode for the (p=2) case (the default in the paper).

```text
Inputs:
  model parameters w
  base optimizer BaseOpt (SGD, AdamW, etc.)
  loss function ℓ
  batch size b
  sub-batch size m   # e.g., per-accelerator shard size
  neighborhood radius ρ > 0
  number of sub-batches K = b / m

Repeat until convergence:
  Sample batch B of size b
  Partition B into K disjoint sub-batches: B1, B2, ..., BK (each size m)

  For each sub-batch Bj:
    # First pass: compute gradient at current weights
    g1_j = ∇w L_Bj(w)

    # Adversarial weight perturbation (p=2 SAM)
    eps_j = ρ * g1_j / (||g1_j||_2 + ε_num)

    # Second pass: gradient at perturbed weights (same sub-batch!)
    g2_j = ∇w L_Bj(w + eps_j)

  # Average SAM gradients across sub-batches / accelerators
  g_sam = (1/K) * Σ_j g2_j

  # Apply base optimizer update using g_sam
  w ← BaseOpt.update(w, g_sam)
```

This matches the paper’s Algorithm 1 logic (compute gradient, form perturbation, compute gradient at perturbed weights, update), but applied per sub-batch and averaged, which is exactly how they describe the multi-accelerator implementation. 

---

## Walkthrough (step by step)

### 1) First gradient pass (find the “sharp” direction)

On each sub-batch (B_j), compute the ordinary gradient:

[
g_j = \nabla_w L_{B_j}(w)
]

This tells you the local direction of increase/decrease in loss. SAM uses it to approximate the direction that most increases loss within a radius-(\rho) ball. The paper derives this from a first-order Taylor approximation of (L(w+\epsilon)). 

---

### 2) Build the perturbation (\hat\epsilon_j)

For general (p), the paper gives a dual-norm formula. For (p=2), it reduces to:

[
\hat\epsilon_j = \rho \cdot \frac{g_j}{|g_j|_2}
]

So you perturb the weights in the local “worst-case” direction, but with fixed norm (\rho). The paper explicitly notes that for (p=2), this is just gradient rescaling. 

---

### 3) Second gradient pass at perturbed weights

Now evaluate the gradient on the **same sub-batch** but at the perturbed weights:

[
g^{SAM}*j \approx \nabla_w L*{B_j}(w + \hat\epsilon_j)
]

This is the key approximation in the paper (their Eq. 3): they drop the second-order term and just use the gradient at (w+\hat\epsilon). They say this works well empirically and even report that including the second-order terms hurt in their ablation. 

---

### 4) Average across sub-batches (the “m” part)

If your batch was split into sub-batches of size (m), average the SAM gradients:

[
\bar g = \frac{1}{K}\sum_{j=1}^K g^{SAM}_j
]

This is the paper’s multi-accelerator recipe: independent SAM gradients per shard, then average. That corresponds to the m-sharpness-style objective (independent inner maximizations on subsets of size (m)). 

---

### 5) Apply your base optimizer update

Use (\bar g) as the gradient input to your optimizer (SGD, AdamW, etc.):

[
w \leftarrow \text{BaseOptStep}(w, \bar g)
]

The paper presents SAM with SGD as the base optimizer in pseudocode, but the mechanism is generally “wrap a base optimizer with the SAM gradient.” 

---

## Implementation notes that matter

* **Two backward passes per step:** SAM needs one gradient to build (\hat\epsilon), and another at the perturbed weights, so it roughly doubles the backward cost. The paper explicitly mentions each SAM update requires two backprop operations. 

* **Use the same sub-batch for both passes:** In m-SAM, compute both passes on the same sub-batch (B_j). This is implied by the paper’s per-accelerator/sub-batch description and is necessary for the derivation to match what you’re optimizing. 

* **(p=2) is the default:** The paper generalizes to (p)-norms, but also reports (p=2) is typically best in their experiments. 

---

If you want, I can also give a **PyTorch-style implementation skeleton** (optimizer wrapper with `first_step` / `second_step`) for this exact m-SAM variant.
