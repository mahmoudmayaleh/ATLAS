# Visual Explanation: MIRA Architecture

## ❌ WRONG: What I Implemented Before

```
┌─────────────────────────────────────────────────────────┐
│                    WRONG APPROACH                        │
│            (Global Averaging - Not MIRA!)                │
└─────────────────────────────────────────────────────────┘

Step 1: Task Clustering
┌────────┐  ┌────────┐  ┌────────┐
│Client 0│  │Client 1│  │Client 2│  ← Sentiment Analysis
│  w₀    │  │  w₁    │  │  w₂    │
└────────┘  └────────┘  └────────┘
     └──────────┴──────────┘
              ↓
       ┌──────────────┐
       │ AGGREGATE    │
       │ (Average)    │
       └──────────────┘
              ↓
       ┌──────────────┐
       │ Model_Group0 │  ← One model for sentiment group
       └──────────────┘

┌────────┐  ┌────────┐  ┌────────┐
│Client 3│  │Client 4│  │Client 5│  ← Question Answering
│  w₃    │  │  w₄    │  │  w₅    │
└────────┘  └────────┘  └────────┘
     └──────────┴──────────┘
              ↓
       ┌──────────────┐
       │ AGGREGATE    │
       │ (Average)    │
       └──────────────┘
              ↓
       ┌──────────────┐
       │ Model_Group1 │  ← One model for QA group
       └──────────────┘

Step 2: MERGE ALL GROUPS
       ┌──────────────┐
       │ Model_Group0 │
       └──────────────┘
              ↓
       ┌──────────────┐
       │ WEIGHTED     │  ← WRONG: Merges everything!
       │ MERGE        │
       └──────────────┘
              ↑
       ┌──────────────┐
       │ Model_Group1 │
       └──────────────┘
              ↓
       ┌──────────────┐
       │ GLOBAL MODEL │  ← ONE MODEL FOR ALL!
       └──────────────┘

Step 3: Broadcast
       Everyone gets the SAME model

Problem: Why cluster if everyone gets same model?!
```

---

## ✅ CORRECT: MIRA's Actual Approach

```
┌─────────────────────────────────────────────────────────┐
│                   CORRECT APPROACH                       │
│         (Laplacian Regularization - MIRA!)               │
└─────────────────────────────────────────────────────────┘

Step 1: Each Client Maintains OWN Model
┌────────┐  ┌────────┐  ┌────────┐
│Client 0│  │Client 1│  │Client 2│  ← Sentiment Analysis
│  W₀    │  │  W₁    │  │  W₂    │     (Each has own model)
└────────┘  └────────┘  └────────┘

┌────────┐  ┌────────┐  ┌────────┐
│Client 3│  │Client 4│  │Client 5│  ← Question Answering
│  W₃    │  │  W₄    │  │  W₅    │     (Each has own model)
└────────┘  └────────┘  └────────┘

Step 2: Build Task Graph (from clustering)
   ┌────────┐
   │Client 0│──────┐
   │  W₀    │←────┐│
   └────────┘     ││
       ↕          ││  Neighbors!
   ┌────────┐     ││  (Same task group)
   │Client 1│←────┘│
   │  W₁    │──────┤
   └────────┘      │
       ↕           │
   ┌────────┐      │
   │Client 2│←─────┘
   │  W₂    │
   └────────┘

   ┌────────┐
   │Client 3│──────┐
   │  W₃    │←────┐│
   └────────┘     ││
       ↕          ││  Different neighbors!
   ┌────────┐     ││  (Different task group)
   │Client 4│←────┘│
   │  W₄    │──────┤
   └────────┘      │
       ↕           │
   ┌────────┐      │
   │Client 5│←─────┘
   │  W₅    │
   └────────┘

Step 3: Laplacian Regularization (Per Client)

For Client 0:
W₀^(t+1) = W₀^(t) - η[a₀₁(W₀-W₁) + a₀₂(W₀-W₂)]
           └─────────────────────────────────┘
              Pull toward neighbors 1 & 2

For Client 1:
W₁^(t+1) = W₁^(t) - η[a₁₀(W₁-W₀) + a₁₂(W₁-W₂)]
           └─────────────────────────────────┘
              Pull toward neighbors 0 & 2

... (same for all clients)

Step 4: Send Personalized Models
┌────────┐  ┌────────┐  ┌────────┐
│Client 0│  │Client 1│  │Client 2│
│  W₀'   │  │  W₁'   │  │  W₂'   │  ← Each gets OWN model
└────────┘  └────────┘  └────────┘     (specialized but similar)

┌────────┐  ┌────────┐  ┌────────┐
│Client 3│  │Client 4│  │Client 5│
│  W₃'   │  │  W₄'   │  │  W₅'   │  ← Each gets OWN model
└────────┘  └────────┘  └────────┘     (specialized but similar)

Result: 6 DIFFERENT models (not 1 global model!)
```

---

## Key Differences

| Aspect                 | ❌ Wrong (Old)       | ✅ Correct (MIRA)           |
| ---------------------- | -------------------- | --------------------------- |
| **Final # Models**     | 1 (global)           | 6 (personalized)            |
| **Operation**          | Averaging            | Regularization              |
| **Formula**            | `W = Σ wᵢWᵢ`         | `Wₖ' = Wₖ - η Σ aₖₗ(Wₖ-Wₗ)` |
| **Task Mixing**        | All tasks merged     | Tasks stay separate         |
| **Personalization**    | None (everyone same) | Full (each unique)          |
| **Clustering Purpose** | Unclear!             | Defines neighbors           |

---

## The "Pulling" Effect

### Visualization of Laplacian Regularization:

```
Before Regularization:
    W₀ = [1.0]  ←─┐
                  ├─ Far apart!
    W₁ = [0.0]  ←─┘

After Regularization (η=0.1):
    Laplacian term = (W₀ - W₁) = 1.0
    W₀' = W₀ - η*(W₀ - W₁) = 1.0 - 0.1*1.0 = 0.9
    W₁' = W₁ - η*(W₁ - W₀) = 0.0 - 0.1*(-1.0) = 0.1

    W₀' = [0.9]  ←─┐
                   ├─ Closer but still distinct!
    W₁' = [0.1]  ←─┘

After Many Rounds:
    W₀ ≈ [0.6]  ←─┐
                  ├─ Converge but not identical
    W₁ ≈ [0.4]  ←─┘
```

**Key:** Models move toward each other but NEVER become identical!

---

## Analogy

### ❌ Wrong Approach (Averaging):

Like mixing red and blue paint → you get purple paint

- Everyone gets the same purple
- Lost the red and blue specialization

### ✅ MIRA Approach (Regularization):

Like red and blue objects in space with springs

- Red objects pulled slightly toward each other (become similar reds)
- Blue objects pulled slightly toward each other (become similar blues)
- Red and blue stay separate
- Each object remains unique but influenced by neighbors

---

## Why MIRA Is Better

### Task Heterogeneity Example:

**Scenario:** Client 0 = "Movie reviews", Client 3 = "Medical Q&A"

#### Wrong Approach:

```
Average(Movie model, Medical model) = Confused hybrid model
↓
Everyone gets confused model
↓
Bad at movies AND bad at medical
```

#### MIRA Approach:

```
Movie model influenced by other movie tasks only
Medical model influenced by other medical tasks only
↓
Client 0 gets movie-specialized model
Client 3 gets medical-specialized model
↓
Good at movies AND good at medical (separately)
```

---

## Formula Breakdown

```
W_k^(t+1) = W_k^(t,R) - η Σ(ℓ∈N_k) a_kℓ(W_k^(t,R) - W_ℓ^(t,R))
```

**Each component:**

- `W_k^(t+1)`: Client k's updated model
- `W_k^(t,R)`: Client k's model after R local training steps
- `η`: Regularization strength (how much to pull)
- `N_k`: Neighbors of client k (from clustering)
- `a_kℓ`: Adjacency weight (how much client ℓ influences k)
- `(W_k - W_ℓ)`: Difference between models

**Interpretation:**

1. Start with your current model `W_k`
2. For each neighbor `ℓ`:
   - Compute difference `(W_k - W_ℓ)`
   - Weight by `a_kℓ` (importance of neighbor)
3. Sum all weighted differences
4. Move in opposite direction: `-η * sum`
5. Result: Pulled toward neighbors but stay distinct

**Not averaging!** It's gradient descent on a regularization term.

---

## Benefits Summary

### Scientific:

- ✅ Matches MIRA paper exactly
- ✅ Handles task heterogeneity properly
- ✅ Novel combination: MIRA + heterogeneous LoRA ranks

### Performance:

- ✅ Personalized models outperform averaged models
- ✅ Better for diverse tasks (sentiment ≠ QA ≠ summarization)
- ✅ Knowledge sharing without harmful mixing

### Practical:

- ✅ Clear clustering purpose (defines neighbors)
- ✅ Scales to many tasks
- ✅ Privacy-friendly (models stay separate)
