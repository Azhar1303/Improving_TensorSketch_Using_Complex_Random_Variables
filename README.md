# Improving TensorSketch Using Complex Random Variables

**Authors:** Amit Sharma, Mohammad Azhar Khan, Rameshwar Pratap


This repository contains the source code for the experiments conducted in the paper **Improving TensorSketch Using Complex Random Variables** accpeted at Uncertainty in Artificial Intelligence (UAI) 2026 conference. 

---

## Requirements

The implementation relies on the following Python toolboxes:

* **NumPy**: For numerical linear algebra and FFT operations.
* **SciPy**: For sparse matrix handling when loading datasets.
* **scikit-learn**: For dataset preprocessing, specifically `StandardScaler` and `fetch_openml`.
* **matplotlib**: For the visualization of distortion and timing results.
* **urllib**: For direct dataset downloads of UCI datasets.
* **time**: For runtime measurements.

---

## Implementation Overview

### Proposed Method: Complex-to-Real TensorSketch

This implementation introduces an improvement to TensorSketch by utilizing fourth roots-of-unity distributions. The proposed approach is implemented in the `ctr_tensorsketch_features` function.

The core ideas of this approach are:
* Replace real-valued sign functions with a random function whose values are drawn independently and uniformly from the four fourth roots of unity.
* Convert complex embeddings into real representations by concatenating the real and imaginary parts using `ctr_stack`.

This technique is designed to achieve the following:
* Reduce the variance of polynomial kernel approximations.
* Improve distortion behavior for high-degree kernels.
* Maintain a computational efficiency that is comparable to classical TensorSketch.

---

## Baselines for Comparison

To evaluate the proposed approach, the repository includes the following baselines:

| Baseline Method | Implementation Functions | Description |
| :--- | :--- | :--- |
| **Classical Real TensorSketch**(1,2) | `real_tensorsketch_features` | Standard polynomial kernel approximation using CountSketch hashing, FFT-based convolution, and inverse FFT reconstruction. |
| **Real Johnson–Lindenstrauss (JL) Sketches**(4) | `real_gaussian_features`, `real_rademacher_features` | Random projection methods used to approximate $K(x, y) = (x^\top y)^p$. |
| **Complex-to-Real (CtR) JL Sketches**(3) | `ctr_gaussian_features`, `ctr_rademacher_features` | Complex-to-Real extensions of standard JL projections. |

---

## Datasets and Experimental Setup

The experiments are conducted on both synthetic and real-world datasets, where all input points are treated as real-valued vectors and ℓ₂-normalized prior to sketching.

### Synthetic Data

* Generated $n = 3000$ random vectors in $\mathbb{R}^{d}$ with $d = 2$.
* Each coordinate is drawn independently from a standard Gaussian distribution and normalized to lie on the unit sphere.
* Evaluates homogeneous polynomial kernels of degrees $p \in \{10, 15, 20\}$.
* Sketch dimensions are varied as $D \in \{100, 300, 500\}$ to stress high-degree behavior in a low-dimensional regime.

### Real-World Data

* **MAGIC Gamma Telescope**(6): Contains $d = 10$ real-valued features.
* **COD-RNA**(7): Contains $d = 8$ numerical attributes.
* Up to $n = 3000$ points are subsampled for evaluation.
* Results are averaged over 20 independent random trials.
* The real-world test logs demonstrate evaluation across parameters such as $p \in \{3, 5, 7\}$ and $D \in \{10, 30, 50\}$.

---

## Comparison Metrics

The methods are evaluated for sketch quality and computational efficiency using two primary metrics:

### 1. KL Divergence
Assesses how well the structure of the degree-$p$ polynomial kernel matrix is preserved. The exact kernel $K$ and approximate kernel $\widehat{K}$ are normalized into discrete distributions:

$$
P_{ij} = \frac{K_{ij}}{\sum_{a,b} K_{ab}}, \qquad Q_{ij} = \frac{\widehat{K}_{ij}}{\sum_{a,b} \widehat{K}_{ab}}
$$

The metric reported is the KL divergence, where a lower value indicates better structure preservation:

$$
\mathrm{KL}(K \,\|\, \widehat{K}) = \sum_{i,j} P_{ij} \log\!\left( \frac{P_{ij}}{Q_{ij} + \varepsilon} \right)
$$
*(where $\varepsilon > 0$ ensures numerical stability)*

### 2. Wall-Clock Time
Measures the computational efficiency required to construct sketch features for a given dimension $D$, with a lower runtime indicating better efficiency.
* Timing is measured using `time.perf_counter()`.
* Only sketch construction time is measured; kernel reconstruction is excluded.
* Reported values are averaged over multiple independent trials.

---

## Rebuttal Additions
The repository also contains additional experiments and their corresponding code added during the rebuttal period.

---

## References

1. Ninh Pham and Rasmus Pagh. Fast and scalable polynomial kernels via explicit feature maps. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2013.
2. Ninh Pham and Rasmus Pagh. Tensor sketch: Fast and scalable polynomial kernel approximation, 2025.
3. Jonas Wacker, Ruben Ohana, and Maurizio Filippone. Complex-to-real sketches for tensor products with applications to the polynomial kernel. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS), 2023.
4. Purushottam Kar and Harish Karnick. Random feature maps for dot product kernels. In Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 2012.
5. Charikar, Moses, Kevin C. Chen, and Martin Farach-Colton. "Finding frequent items in data streams." Theoretical Computer Science, 2004.
6. R. Bock. MAGIC Gamma Telescope. UCI Machine Learning Repository, 2004.
7. Andrew Uzilov, Joshua Keegan, and David Mathews. Detection of non-coding rnas on the basis of predicted secondary structure formation free energy change. BMC bioinformatics, 2006.
