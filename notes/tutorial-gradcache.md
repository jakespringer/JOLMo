This paper is **“Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup”** (Gao et al., 2021). It introduces **gradient caching** so you can train with effectively large contrastive batches (and lots of in-batch negatives) without needing to fit the whole encoder forward/backward graph in memory at once. The key claim is that it reproduces the **exact same gradient update** as full large-batch training while keeping encoder memory usage nearly constant in the batch dimension. ([arXiv][1])

## Core idea in one sentence

They split backprop into:

1. **loss → representations** (depends on the *whole batch*), and
2. **representations → encoder parameters** (can be done *sub-batch by sub-batch* once representation gradients are known).

That lets them cache gradients on embeddings, then replay sub-batches through the encoder with those cached gradients. 

---

## Pseudocode implementation (faithful to the paper)

Below is a clean implementation sketch (PyTorch-style pseudocode) of the paper’s Section 3.3 method.

```python
# Pseudocode for Gradient Cache (GC) training for contrastive learning
# Based on Gao et al. (2021), Section 3.3 and equations (1)-(9)

def train_step_gradient_cache(
    batch_S,              # anchors: [s1, ..., s_|S|]
    batch_T,              # targets: [t1, ..., t_|T|]
    positive_index,       # r_i: for each anchor i, index of its positive target in batch_T
    f_encoder,            # encoder f(s; Θ)
    g_encoder,            # encoder g(t; Λ)
    optimizer,
    subbatch_size_S,
    subbatch_size_T,
    temperature=1.0,
):
    optimizer.zero_grad()

    # ---------------------------------------------------------
    # Step 1: Graph-less Forward (compute and store all reps)
    # ---------------------------------------------------------
    # No autograd graph for the encoders here: just numeric representations.
    with no_grad():
        F = []  # f(s_i) for all anchors
        for S_chunk in chunk(batch_S, subbatch_size_S):
            F_chunk = f_encoder(S_chunk)      # shape: [bs, d]
            F.append(F_chunk)
        F = concat(F, dim=0)                  # shape: [|S|, d]

        G = []  # g(t_j) for all targets
        for T_chunk in chunk(batch_T, subbatch_size_T):
            G_chunk = g_encoder(T_chunk)      # shape: [bt, d]
            G.append(G_chunk)
        G = concat(G, dim=0)                  # shape: [|T|, d]

    # ---------------------------------------------------------
    # Step 2: Representation Gradient Computation + Caching
    # ---------------------------------------------------------
    # Build a small graph ONLY on the representation tensors, not the encoders.
    # Treat F and G as leaf tensors requiring grad.
    F_leaf = detach(F).requires_grad_(True)   # shape: [|S|, d]
    G_leaf = detach(G).requires_grad_(True)   # shape: [|T|, d]

    # Contrastive logits over the full batch (all in-batch negatives included)
    # logits[i, j] = <F_i, G_j> / tau
    logits = (F_leaf @ G_leaf.T) / temperature

    # Cross-entropy over positives: positive_index[i] gives target j for anchor i
    loss = contrastive_cross_entropy(logits, positive_index)

    # Backprop ONLY to F_leaf and G_leaf
    loss.backward()

    # Representation gradient cache
    # u_i = ∂L/∂f(s_i), v_j = ∂L/∂g(t_j)
    U_cache = detach(F_leaf.grad).clone()     # shape: [|S|, d]
    V_cache = detach(G_leaf.grad).clone()     # shape: [|T|, d]

    # Free the representation-graph objects if desired
    del F_leaf, G_leaf, logits, loss

    # ---------------------------------------------------------
    # Step 3: Sub-batch Gradient Accumulation through encoders
    # ---------------------------------------------------------
    # Re-run encoder forwards WITH graph, sub-batch by sub-batch.
    # Use cached representation gradients as "external gradients".
    # This accumulates the exact full-batch parameter gradient.
    s_start = 0
    for S_chunk in chunk(batch_S, subbatch_size_S):
        bs = len(S_chunk)
        F_chunk = f_encoder(S_chunk)                      # graph attached
        U_chunk = U_cache[s_start:s_start + bs]           # cached dL/dF for this chunk

        # Backprop through f_encoder with externally supplied grad
        # Equivalent to summing <U_chunk, F_chunk> and calling backward()
        backward_on_tensor(F_chunk, grad_tensor=U_chunk, retain_graph=False)

        s_start += bs

    t_start = 0
    for T_chunk in chunk(batch_T, subbatch_size_T):
        bt = len(T_chunk)
        G_chunk = g_encoder(T_chunk)                      # graph attached
        V_chunk = V_cache[t_start:t_start + bt]           # cached dL/dG for this chunk

        # Backprop through g_encoder with externally supplied grad
        backward_on_tensor(G_chunk, grad_tensor=V_chunk, retain_graph=False)

        t_start += bt

    # ---------------------------------------------------------
    # Step 4: Optimizer step
    # ---------------------------------------------------------
    optimizer.step()
```

---

## Technical walkthrough

### 1) Why ordinary gradient accumulation fails for contrastive loss

For a standard contrastive setup, each anchor’s loss depends on **all targets in the batch** because the denominator is a softmax over all in-batch negatives (plus positives / hard negatives). In the paper’s notation, they define a dot-product contrastive loss over anchor set (S) and target set (T), with positive mapping (r_i). Each term depends on the entire set (T). 

That means if you split a big batch into small chunks and do normal gradient accumulation, each chunk sees a **different denominator** (fewer negatives), so you are not simulating the big-batch gradient. The paper explicitly points this out and motivates why standard accumulation is not equivalent here. 

---

### 2) The key chain-rule factorization

They write the encoder parameter gradients using the chain rule:

* (\partial L / \partial \Theta = \sum_i (\partial L / \partial f(s_i))(\partial f(s_i)/\partial \Theta))
* (\partial L / \partial \Lambda = \sum_j (\partial L / \partial g(t_j))(\partial g(t_j)/\partial \Lambda))

This is the central trick: the representation gradients (\partial L/\partial f(s_i)) and (\partial L/\partial g(t_j)) contain the global batch coupling, but the Jacobian factors (\partial f(s_i)/\partial \Theta) and (\partial g(t_j)/\partial \Lambda) are local to each example. 

They also define a normalized similarity (p_{ij}) and derive closed-form expressions for (\partial L/\partial f(s_i)) and (\partial L/\partial g(t_j)), making explicit that these terms depend on the full batch representations. That’s why you must compute them from the full batch (or equivalent all-gathered representations). 

---

### 3) The two observations that make gradient caching possible

The paper makes two explicit observations:

1. **Encoder Jacobians are local**
   (\partial f(s_i)/\partial \Theta) depends only on (s_i,\Theta), and similarly for (g).

2. **Representation gradients need only the numeric representations**
   (\partial L/\partial f(s_i)) and (\partial L/\partial g(t_j)) can be computed from the representation values (F, G), without the encoder computation graph.

Together, this means you can first compute all representations numerically, compute/cache the representation gradients, and only then backprop through the encoders in smaller sub-batches. 

---

### 4) Step-by-step what each stage is doing

#### Step 1: Graph-less forward

You run an extra forward pass over all examples to get all representations, but **without storing encoder activations** (no autograd graph). This is cheap in memory and gives you the full-batch embeddings needed for the true contrastive denominator. 

#### Step 2: Representation-gradient computation and caching

Now you treat the stored representations as leaf tensors and compute the contrastive loss on them. A backward pass gives:

* (u_i = \partial L/\partial f(s_i))
* (v_j = \partial L/\partial g(t_j))

These are cached (the “Representation Gradient Cache”). Importantly, the encoder is **not** part of this graph. 

#### Step 3: Sub-batch gradient accumulation through the encoders

Re-run the encoder on one sub-batch at a time **with autograd enabled**. For each sub-batch output tensor, inject the cached representation gradients as the upstream gradient and backprop through the encoder. Accumulate parameter gradients across all sub-batches. This reconstructs the same full-batch gradient by linearity of the chain rule (their Eqs. 8 and 9). 

#### Step 4: Optimizer step

Once all sub-batches have contributed, call `optimizer.step()`. This is equivalent to a single full-batch update, but without needing the full encoder graph resident at once. 

---

### 5) Why this is memory-efficient

Direct full-batch training stores encoder activations for the entire batch, so memory scales roughly with batch size. Here, encoder forward/backward memory is bounded by the **sub-batch** size instead. The main extra persistent memory is just the cached representations (and later their gradients), which are much smaller than storing all transformer activations. The paper summarizes this cache size as roughly ((|S|d + |T|d)) floating-point values (for representation dimension (d)). 

---

### 6) Multi-GPU variant

Their multi-GPU extension is very clean:

* After Step 1, do one **all-gather** so every GPU can see the global representations.
* Step 2 computes the global-batch contrastive loss, but each GPU only caches gradients for its **local** examples.
* Step 3 needs no extra communication (local sub-batch backprops).
* Step 4 uses standard gradient reduction across GPUs.

So the only extra cross-GPU communication beyond normal data parallelism is the all-gather of representations after Step 1. 

---

### 7) Extension beyond dot-product similarity

The paper also extends this to a learned deep interaction / distance function (\Phi(f(s_i), g(t_j))), not just a parameterless dot product. They introduce a second cache:

* **Distance Gradient Cache**: (w_{ij} = \partial L/\partial d_{ij})

Then they can:

* update the interaction model parameters (\Omega) in sub-batches using (w_{ij}), and
* simultaneously accumulate representation gradients (u_i, v_j) from the interaction graph, then proceed with the usual encoder Step 3.

So the same principle applies even when the score function itself is deep. 

---

## What they show experimentally

They plug the method into DPR (dense passage retrieval) on Natural Questions and compare against:

* **Sequential** (small batch that fits)
* **Accumulation** (small batches accumulated)
* **Cache** (their method)
* DPR reference numbers from the original multi-GPU setup

They report that Cache matches or exceeds the large-batch DPR reference in their run, while standard accumulation underperforms because it does not preserve the same number of in-batch negatives. They also show Cache scales to very large examples-per-update with about **20% time overhead** (for representation pre-computation), and mention a practical single-GPU training time (31 hours on a single RTX 2080 Ti in their setup). 

---

## Practical implementation notes (important if you code this)

* **You need two forward passes per step** through the encoder:

  * one no-grad pass (Step 1),
  * one grad-enabled pass in sub-batches (Step 3).
    This is the time-vs-memory tradeoff. 

* In Step 2, representations should be **detached** and marked `requires_grad=True` so autograd computes gradients w.r.t. the embeddings only.

* In Step 3, use the cached gradients as the explicit upstream gradient to `backward` on each sub-batch representation tensor.

* The method is **exact** (not an approximation) for the same batch and loss, because it uses the same full-batch representation-level gradients and just reorganizes the chain rule. 

---

If you want, I can also give you a **full PyTorch implementation template** (not just pseudocode) for a dual-encoder training loop with this method, including the exact `autograd.grad` / `tensor.backward(grad)` calls.

[1]: https://arxiv.org/abs/2101.06983?utm_source=chatgpt.com "Scaling Deep Contrastive Learning Batch Size under Memory ..."
