# Topic 1: Definitions of Deep Learning

---

## 1. Topic Header
**"What is Deep Learning?" / "Definitions of Deep Learning"** (Slides 9–10)

---

## 2. What/Why (Layman First)

### Plain-English Intuition

Imagine you want to teach a computer to recognize cats in photos. Traditional programming would require you to manually write rules: "If there are pointy ears, whiskers, and four legs, it's a cat." But this is incredibly hard—cats come in all shapes, colors, and poses!

**Deep Learning** flips this approach. Instead of giving the computer rules, you give it **thousands of examples** (pictures of cats and "not cats"), and the computer **learns the patterns on its own**. It builds up understanding in **layers**—first detecting edges, then shapes, then ears, then faces, and finally "cat vs. dog."

> **Think of it like how a child learns**: A child doesn't learn what a "cat" is by memorizing a rulebook. They see many cats, hear "that's a cat," and eventually their brain forms the concept.

### Real-World Applications
| Domain | Application |
|--------|-------------|
| **Vision** | Face recognition (iPhone Face ID), medical imaging (detecting tumors), self-driving cars |
| **Language** | ChatGPT, Google Translate, voice assistants (Siri, Alexa) |
| **Audio** | Speech-to-text, music generation, noise cancellation |
| **Gaming** | AlphaGo (beat world champion in Go), game AI |
| **Science** | Protein folding (AlphaFold), drug discovery |

### Why Do We Need Deep Learning?

Before deep learning, we used **traditional Machine Learning** (like SVMs, Decision Trees). These required humans to **hand-craft features**—i.e., you had to manually decide what characteristics to extract from data.

| Traditional ML | Deep Learning |
|----------------|---------------|
| Human designs features | Features are **learned automatically** |
| Struggles with unstructured data (images, audio, text) | **Excels** at unstructured data |
| Performance plateaus with more data | Performance **keeps improving** with more data |

**The key breakthrough**: Deep Learning can learn **hierarchical representations** automatically. The "deep" in deep learning refers to the **depth** (number of layers) in the neural network.

### Key Points from Slides (in Layman Terms)

1. **Multiple layers** extract progressively higher-level features (edges → shapes → objects)
2. **Inspired by the human brain**—but not a perfect copy
3. **"Deep"** = 3 or more layers in the neural network
4. **Learns by example**, not by explicit programming

---

## 3. Theory (Rigorous)

### Formal Definition

> **Deep Learning** is a class of machine learning algorithms that uses multiple layers of nonlinear processing units for feature extraction and transformation. Each successive layer uses the output from the previous layer as input.

### Mathematical Framework

A deep neural network can be represented as a **composition of functions**:

```
f(x) = f⁽ᴸ⁾ ∘ f⁽ᴸ⁻¹⁾ ∘ ... ∘ f⁽²⁾ ∘ f⁽¹⁾(x)
```

**Reading this formula:**
- `x` = input vector (e.g., pixel values of an image)
- `f⁽¹⁾(x)` = first layer's transformation of input
- `f⁽²⁾(...)` = second layer applied to output of first layer
- `∘` = "composition" (means "apply one after another")
- `L` = total number of layers

**In plain English:** The output goes through Layer 1, then Layer 2, then Layer 3... up to Layer L.

---

### What Each Layer Computes

```
h⁽ˡ⁾ = σ( W⁽ˡ⁾ · h⁽ˡ⁻¹⁾ + b⁽ˡ⁾ )
```

**Breaking this down:**

| Symbol | Meaning | Example |
|--------|---------|---------|
| `h⁽ˡ⁾` | Output of layer l | Hidden representation at layer 3 |
| `h⁽ˡ⁻¹⁾` | Output from previous layer | Fed as input to current layer |
| `W⁽ˡ⁾` | Weight matrix for layer l | Numbers the network learns |
| `b⁽ˡ⁾` | Bias vector for layer l | Offset values (also learned) |
| `σ(·)` | Activation function | ReLU, Sigmoid, Tanh, etc. |
| `h⁽⁰⁾ = x` | Layer 0 is just the input | Starting point |

**The computation in 3 steps:**
1. **Multiply**: `W⁽ˡ⁾ · h⁽ˡ⁻¹⁾` (matrix times vector)
2. **Add bias**: `+ b⁽ˡ⁾`
3. **Activate**: Apply σ (introduces nonlinearity)

---

### Why "Deep"?

| Network Type | Number of Hidden Layers |
|--------------|-------------------------|
| Shallow Neural Network | 1–2 hidden layers |
| **Deep Neural Network** | **3 or more** hidden layers |

### Universal Approximation Theorem

**Statement:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function, under mild assumptions on the activation function.

> **Important**: This theorem says a shallow network CAN approximate any function, but it may require an **exponentially large** number of neurons. Deep networks can achieve the same with **exponentially fewer** parameters—this is the power of depth!

### Hierarchy of Representations

Deep learning learns a **hierarchy of features**:

```
Layer 1: Edges, gradients
    ↓
Layer 2: Corners, textures
    ↓
Layer 3: Object parts (eyes, wheels)
    ↓
Layer 4+: Whole objects (faces, cars)
```

This is formalized as **representation learning**—the network learns useful representations automatically rather than using hand-engineered features.

---

## 4. Derivations

Since this topic is primarily definitional, there isn't a complex derivation. However, let's show how the layer-by-layer computation works:

### Forward Pass Derivation (Layer l)

**Given**: 
- Input from previous layer: `h⁽ˡ⁻¹⁾`
- Weights: `W⁽ˡ⁾`
- Bias: `b⁽ˡ⁾`

**Step 1**: Compute the linear combination (pre-activation)
```
z⁽ˡ⁾ = W⁽ˡ⁾ · h⁽ˡ⁻¹⁾ + b⁽ˡ⁾
```

**Step 2**: Apply nonlinear activation
```
h⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

**Example with ReLU activation**: `σ(z) = max(0, z)`

For a 2-layer network with 2D input vector `x`:
```
h⁽¹⁾ = σ(W⁽¹⁾ · x + b⁽¹⁾)     ← First layer output
y    = W⁽²⁾ · h⁽¹⁾ + b⁽²⁾       ← Final output
```

---

## 5. Examples (Numeric, Non-Textbook)

### Example 1: Simple 2-Layer Forward Pass

**Given**:
```
Input x = [1, 2]  (column vector)

Layer 1 weights W⁽¹⁾ = | 0.5  -0.5 |
                        | 1.0   0.5 |

Layer 1 bias b⁽¹⁾ = [0.1, -0.1]

Activation: ReLU
```

**Step 1**: Compute pre-activation `z⁽¹⁾ = W⁽¹⁾ · x + b⁽¹⁾`

```
z⁽¹⁾ = | 0.5  -0.5 |   | 1 |   | 0.1  |
        | 1.0   0.5 | × | 2 | + | -0.1 |

     = | (0.5)(1) + (-0.5)(2) |   | 0.1  |
       | (1.0)(1) + (0.5)(2)  | + | -0.1 |

     = | 0.5 - 1.0 |   | 0.1  |   | -0.5 + 0.1 |   | -0.4 |
       | 1.0 + 1.0 | + | -0.1 | = |  2.0 - 0.1 | = |  1.9 |
```

**Step 2**: Apply ReLU (max(0, z) for each element)
```
h⁽¹⁾ = ReLU(z⁽¹⁾) = | max(0, -0.4) |   | 0   |
                     | max(0,  1.9) | = | 1.9 |
```

**Result**: The first layer outputs `[0, 1.9]`

---

### Example 2: Counting Layers

**Question**: Which of these qualifies as "deep learning"?

| Network | Hidden Layers | Deep? |
|---------|---------------|-------|
| Logistic Regression | 0 | ❌ No |
| Single hidden layer | 1 | ❌ No (shallow) |
| 2 hidden layers | 2 | ⚠️ Borderline (some say yes) |
| **3 hidden layers** | **3** | ✅ **Yes (Deep)** |
| ResNet-50 | 50+ | ✅ Yes (Very Deep) |
| GPT-4 | 100+ transformer layers | ✅ Yes (Extremely Deep) |

---

### Example 3: Feature Hierarchy in Image Recognition

Consider classifying a handwritten digit "5":

| Layer | What It Learns | Example Pattern |
|-------|----------------|-----------------|
| 1 | Edges | Horizontal line ─, Vertical line │ |
| 2 | Curves | Arc ⌒, Corner ∟ |
| 3 | Parts | Top curve + Bottom hook |
| 4 | Digit | Complete "5" pattern |

---

## 6. Slide Example(s) Explained

The slides provide multiple definitions. Let's unify them:

| Slide Definition | Explanation |
|------------------|-------------|
| "Multiple layers extract progressively higher-level features" | Each layer builds on the previous—edges → shapes → objects |
| "Inspired by the human brain" | Uses neuron-like processing units, but simplified |
| "Learn by example" | Supervised learning from labeled data |
| "Neural network with 3+ layers" | The depth threshold for "deep" |
| "Deep = more layers to learn from data" | More layers = more abstraction = better learning |

### The Relationship Pyramid (from Slides)

```
         ┌─────────────────┐
         │  Artificial     │
         │  Intelligence   │  ← Broadest: includes rule-based AI, expert systems
         │      (AI)       │
         ├─────────────────┤
         │    Machine      │  ← Learns from data (includes SVMs, trees, etc.)
         │    Learning     │
         ├─────────────────┤
         │     Deep        │  ← Neural networks with 3+ layers
         │    Learning     │
         └─────────────────┘
```

> **Key insight**: All deep learning is machine learning, all ML is AI, but not vice versa!

---

## 7. Checks & Pitfalls

### Common Mistakes

| ❌ What NOT to Do | ✅ What TO Do |
|-------------------|---------------|
| "Deep learning = AI" | Deep learning is a **subset** of AI |
| "Deep learning copies the brain exactly" | It's **inspired** by, not a replica of, the brain |
| "More layers = always better" | Too deep can cause vanishing gradients; architecture matters |
| "Deep learning works for everything" | Works best for unstructured data (images, text, audio) |
| "Deep learning doesn't need data" | Needs **large amounts** of data to work well |

### Edge Cases

1. **Small data**: Deep learning typically needs thousands to millions of examples. With <1000 samples, traditional ML often wins.

2. **Structured/tabular data**: For data in spreadsheets (e.g., financial data with columns), gradient boosting (XGBoost) often beats deep learning.

3. **Interpretability**: Deep learning models are "black boxes." In medical/legal contexts, simpler models may be preferred.

### Sanity Check Questions

- Is my target variable well-defined?
- Do I have enough data (typically 10× the number of parameters)?
- Is my data unstructured (images/text/audio) or structured (tables)?
- Do I need interpretability?

---

## 8. MTech-Level Problems

### Problem 1: Architecture Analysis

**Question**: A neural network has the following architecture:
- Input layer: 784 neurons (28×28 image)
- Hidden layer 1: 256 neurons
- Hidden layer 2: 128 neurons
- Hidden layer 3: 64 neurons
- Output layer: 10 neurons (for digits 0–9)

(a) How many hidden layers does this network have?  
(b) Is this a deep neural network? Justify.  
(c) Calculate the total number of learnable parameters (weights + biases).

**Solution**:

**(a)** The network has **3 hidden layers** (256, 128, 64 neurons).

**(b)** **Yes, it is deep.** By definition, a neural network with 3 or more layers (excluding input) is considered "deep." This network has 3 hidden + 1 output = 4 layers beyond input.

**(c)** Parameter count:

| Connection | Weights | Biases | Total |
|------------|---------|--------|-------|
| Input → Hidden 1 | 784 × 256 = 200,704 | 256 | 200,960 |
| Hidden 1 → Hidden 2 | 256 × 128 = 32,768 | 128 | 32,896 |
| Hidden 2 → Hidden 3 | 128 × 64 = 8,192 | 64 | 8,256 |
| Hidden 3 → Output | 64 × 10 = 640 | 10 | 650 |
| **Total** | | | **242,762** |

**Formula used**: For a layer with `n_prev` inputs and `n_curr` outputs:
```
Parameters = (n_prev × n_curr) + n_curr = n_curr × (n_prev + 1)
```

---

### Problem 2: Conceptual Depth

**Question**: Compare and contrast:
(a) Why can't we just use a very wide shallow network instead of a deep network?  
(b) Given the Universal Approximation Theorem, why do we need depth at all?

**Solution**:

**(a) Width vs Depth**:

| Aspect | Wide Shallow Network | Deep Network |
|--------|---------------------|--------------|
| Parameters | Exponentially many needed | Polynomial in depth |
| Features | Single level of abstraction | Hierarchical features |
| Expressiveness | Limited compositional structure | Can represent complex compositions |
| Efficiency | Computationally expensive | More parameter-efficient |

**Example**: To represent a function like `f(x) = ((x² + 1)² - 2)³`, a deep network can learn this compositionally (layer by layer). A shallow network would need to approximate the entire complex surface at once.

**(b) Why Depth Despite UAT**:

The Universal Approximation Theorem guarantees **existence** of a shallow approximator, but says nothing about **efficiency**. 

**Key insight**: There exist functions that can be represented by a network of depth `k` with polynomial size, but require **exponential** size if depth is restricted to `k-1`.

**Formal statement** (simplified):
```
There exists a function f such that:
  - A depth-d network needs only O(n²) neurons
  - A depth-(d-1) network needs Ω(2ⁿ) neurons
```

This is why depth matters in practice!

---

## 9. Quick Quiz (Socratic)

Answer these questions before we move on:

1. **True or False**: Deep learning is a subset of machine learning.

2. **Fill in the blank**: A neural network is considered "deep" when it has ___ or more layers.

3. **Which of the following is NOT a reason for the rise of deep learning?**
   - (A) Large amounts of data
   - (B) Cheap GPU computation
   - (C) Manual feature engineering
   - (D) Automated feature learning

4. **In your own words**: Why does deep learning work particularly well for images and text, but not always for spreadsheet data?

5. **Quick calculation**: A network has layers with [100, 50, 25, 10] neurons (input to output). How many total weight parameters are there? (Ignore biases)

---

## 10. Key Takeaways

> 📌 **Always Remember**

| Point | Detail |
|-------|--------|
| **Definition** | Deep Learning = Neural networks with 3+ hidden layers |
| **Hierarchy** | AI ⊃ Machine Learning ⊃ **Deep Learning** |
| **Key advantage** | **Automatic feature learning** from raw data |
| **Best suited for** | Unstructured data (images, text, audio, video) |
| **Why "deep"?** | Depth enables hierarchical feature extraction efficiently |
| **Data requirement** | Needs **large datasets** to work well |
| **Compute requirement** | GPUs/TPUs for training; CPUs can work for inference |

### Core Formula
```
h⁽ˡ⁾ = σ( W⁽ˡ⁾ · h⁽ˡ⁻¹⁾ + b⁽ˡ⁾ )
```

### Remember the Pyramid
```
AI > ML > DL
```

---

## 11. Ready for Next Topic?

We've covered **"Definitions of Deep Learning"** including:
- What deep learning is (layman + formal)
- Why we need it (vs traditional ML)
- Mathematical framework
- Worked examples
- Common pitfalls

**Next topic in the slides**: "Where in AI sits DL?" / "Deep Learning Timeline" / "Why Deep Learning?" (We can cover these together as they're related, or separately—your choice!)

---

> **Ready to move to the next topic?**  
> Please answer the quiz questions above and let me know when you're ready! 🎓
