Let's ground this completely in the Machine Learning perspective using the ETTh1 dataset. 

In the ETTh1 dataset, you are feeding the model a sequence of past **Oil Temperatures (OT)** and asking it to predict the future.

Imagine we are passing a sequence of length 3 into the model. To make it simple, let's pretend the multi-dimensional embeddings are just the raw temperature values.

* **Position 0 (10:00 AM):** 25.0°C
* **Position 1 (11:00 AM):** 28.0°C
* **Position 2 (12:00 PM):** 25.0°C

### The Fundamental ML Problem
Self-Attention (the core of the Informer) treats data like an unordered "bag of words." If you don't tell the model about time, it sees two `25.0` tokens and treats them identically. It has no idea that one happened at 10 AM and the other happened at 12 PM. It just sees `[25.0, 28.0, 25.0]`.

The standard approach (Sine/Cosine Positional Encoding) solves this by blindly adding a math wave to the values: `[25.0 + 0.1]`, `[28.0 + 0.5]`, `[25.0 + 0.9]`. 
This makes them unique, but it forces the model's dense layers to mathematically *decode* those arbitrary waves to figure out the shape of the temperature curve.

### The Phase-Space Solution (What your experiments are testing)
Phase-Space says: Let's explicitly calculate the "shape" and "flow" of the temperature sequence and feed *that* directly into the model as the positional encoding.

Let's look at how the three components (Label, Order, Distance) mathematically transform our 3-token example for the ML model:

#### 1. Label (Distinctiveness - Exp 3)
**The Goal:** Guarantee the model knows 10:00 AM is not 12:00 PM.
**What happens:** The model generates mathematically orthogonal (completely independent) vectors using Legendre polynomials.
* **10:00 AM:** 25.0°C + `[Label 0]`
* **12:00 PM:** 25.0°C + `[Label 2]`
**ML Perspective:** The model now has a strict "ID Tag" for every timestamp. It will never confuse the first 25.0°C with the second one. However, IDs alone don't tell the model if the temperature is rising or falling (which is why Label-Only fails).

#### 2. Order (Directionality - Exp 4)
**The Goal:** Tell the model about the *trend* (slope) of the data from the perspective of each token.
**What happens:** The model calculates the signed displacement: `Average(X_i - X_j)`. Let's calculate the "Order" signal for each of our three tokens:

* **For Position 0 (25.0°C):**
  * Diff to Pos 1: 25.0 - 28.0 = -3.0
  * Diff to Pos 2: 25.0 - 25.0 = 0.0
  * *Average Order Signal = **-1.5***
  * **ML Perspective:** The model learns: *"From my perspective, the rest of the sequence is hotter on average. I am in a valley."*

* **For Position 1 (28.0°C):**
  * Diff to Pos 0: 28.0 - 25.0 = +3.0
  * Diff to Pos 2: 28.0 - 25.0 = +3.0
  * *Average Order Signal = **+3.0***
  * **ML Perspective:** The model learns: *"From my perspective, I am at a peak. Everything else is colder."*

* **For Position 2 (25.0°C):**
  * Diff to Pos 0: 25.0 - 25.0 = 0.0
  * Diff to Pos 1: 25.0 - 28.0 = -3.0
  * *Average Order Signal = **-1.5***

**The Flaw in Order-Only (Exp 4):** Notice that Position 0 and Position 2 have the **exact same Order signal (-1.5)**. If you only use Order, the model gets confused again! It thinks 10:00 AM and 12:00 PM are in the exact same trajectory. 

#### 3. Distance (Locality - Exp 1)
**The Goal:** Tell the model that the temperature 1 hour ago matters infinitely more than the temperature 3 days ago.
**What happens:** We apply the Distance Decay (`alpha`). Let's recalculate the Order signal for Position 2, but this time we heavily weight the immediate past (Pos 1) and ignore the distant past (Pos 0).

* **For Position 2 (25.0°C) with Distance Decay:**
  * Diff to Pos 1 (1 hr ago, Weight 1.0): (25.0 - 28.0) * 1.0 = -3.0
  * Diff to Pos 0 (2 hrs ago, Weight 0.1): (25.0 - 25.0) * 0.1 = 0.0
  * *Weighted Order Signal = **-3.0*** (Instead of -1.5)

**ML Perspective:** Now, Position 2's positional encoding heavily reflects the sudden *drop* from 28.0 to 25.0. It is no longer identical to Position 0. The context is entirely localized to its immediate neighborhood.

### Putting it all together (The Full Phase-Space)
If you combine all three (L + O + D), what is the model actually learning when it looks at the 12:00 PM token?

Instead of just seeing a static value of `25.0°C` at `Index 2`, the self-attention mechanism receives a rich, highly informative embedding that essentially says:
> *"I am uniquely Token 2 (Label). I am in a downward trajectory relative to my surroundings (Order), and I am paying 90% of my attention to the steep 3-degree drop that happened exactly 1 hour ago (Distance)."*

From a Machine Learning perspective, providing the model with this pre-calculated "dynamic trajectory" makes forecasting the ETTh1 Oil Temperature much easier than forcing the neural network to deduce the physical trend from static sine waves!