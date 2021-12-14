---
title: >-
  SIMILAR: Submodular Information Measures Based Active Learning In Realistic
  Scenarios
description: Summary of the NeurIPS’21 publication SIMILAR
date: '2021-12-07T05:03:11.852Z'
tags:
  - Efficient Machine Learning
  - Subset Selection
  - Semi-Supervised learning
  - Submodular Optimization
author: Suraj Kothawade
member: suraj-kothawade
---

\[Accepted to NeurIPS 2021- [Paper](https://proceedings.neurips.cc/paper/2021/file/9af08cda54faea9adf40a201794183cf-Paper.pdf), [Code](https://github.com/decile-team/distil)\]

Over the past several years, active learning (AL) strategies have proven to be useful in reducing labeling costs. However, current methods do not work well when it comes to real-world datasets, which have imperfections and a number of characteristics that make learning from them challenging:

![](https://www.caraml-lab.com/images/blog/1__vcaxeFANwuAVWtpLRWC__Ow.png)

Firstly, real-world datasets are _imbalanced_ and some classes are _very rare._ Some examples of this imbalance come from medical imaging domains; for instance, images of cancer cells are often rarer than their benign counterparts in cancer imaging datasets. Another example is in the domain of autonomous vehicles, where we want to detect all objects accurately. However, because some objects in certain situations are rare, like pedestrians in the dark, it is often the case that these models fail in detecting and classifying rare classes.

![](https://www.caraml-lab.com/images/blog/1__gpwEQ2xPPqbHVSokUtiDDw.png)

Secondly, real-world data has a lot of redundancy. This redundancy is more prominent in datasets that are created by sampling frames from videos (e.g., footage from a car driving on a freeway or surveillance camera footage).

![](https://www.caraml-lab.com/images/blog/1__AMm9m8DWVFVBpZQwh8GHUQ.png)

Thirdly, it is common to have out-of-distribution (OOD) data, where some part of the unlabeled data is not of concern to the task at hand. For example, in the medical imaging domain, some X-ray images in the dataset may be incorrectly acquired, thereby making them out-of-distribution.

![](https://www.caraml-lab.com/images/blog/1__Bg8__K03FtysxhEqF8VRL6g.png)

In our work, we address the following question:

> Can a machine learning model be trained using a single unified active learning framework that works for a broad spectrum of realistic scenarios?

**The SIMILAR framework  
**We propose [_SIMILAR_](https://arxiv.org/pdf/2107.00717.pdf), a unified active learning framework that acts as a _one-stop solution_ for many realistic scenarios discussed above. The main idea behind our framework is to exploit the relationship between the submodular information measures (SIM) by appropriately choosing a query set _Q_ and a private set _P._ The unification comes from the rich modeling capacity of submodular conditional mutual information (SCMI). We obtain the submodular mutual information (SMI) and submodular conditional gain (SCG) formulations from SCMI and apply them to different realistic scenarios.

![](https://www.caraml-lab.com/images/blog/1__T__fvXmfS9PWSTl3gwA__rsw.png)

We use the last linear layer gradients using hypothesized labels to represent each data point. The hypothesized label for each data point is assigned as the class with the maximum probability. To instantiate the SIM based functions, we compute a similarity kernel using the gradients of the model obtained from the current active learning round. Finally, we optimize the submodular function using a greedy strategy to acquire a subset of the unlabeled set for human labeling. Once labeled, we add it to the labeled training dataset and proceed to the next iteration.

In the above example for real-world dataset scenarios for digit classification, we can apply the SIMILAR framework as follows.

![](https://www.caraml-lab.com/images/blog/1__KsNG8NZSmxIz5vLF9YJk8w.png)

**Results**  
Empirically, we show that _SIMILAR_ significantly outperforms existing active learning algorithms by as much as ≈ 5% − 18% in the case of rare classes and ≈ 5% − 10% in the case of out-of-distribution data on several image classification tasks like CIFAR-10, MNIST, and ImageNet.

![](https://www.caraml-lab.com/images/blog/1__Rq45wnVJDGs__zKjONYLsPA.png)
![](https://www.caraml-lab.com/images/blog/1__BZuTGPXrmIKBmOiERLcIPQ.png)
![](https://www.caraml-lab.com/images/blog/1__jRXrYEnqkvk6WpQbm0X__iw.png)

SIMILAR is available as a part of the DISTIL toolkit: [https://github.com/decile-team/distil](https://github.com/decile-team/distil)

To make using SIMILAR easy, we provide tutorial notebooks for each of the realistic scenarios discussed above:

1.  [Rare Classes Tutorial on CIFAR-10](https://github.com/decile-team/distil/blob/main/tutorials/image_classification/realistic_scenarios/DISTIL_Example_Rare_Classes_CIFAR10.ipynb)
2.  [Rare Classes Tutorial on Medical Data](https://github.com/decile-team/distil/blob/main/tutorials/image_classification/realistic_scenarios/DISTIL_Example_Rare_Classes_PneumoniaMNIST.ipynb)
3.  [Redundancy Tutorial](https://github.com/decile-team/distil/blob/main/tutorials/image_classification/realistic_scenarios/DISTIL_Example_Redundancy_CIFAR10.ipynb)
4.  [Out-of-distribution data Tutorial](https://github.com/decile-team/distil/blob/main/tutorials/image_classification/realistic_scenarios/DISTIL_Example_OOD_CIFAR10.ipynb)

**Future Thoughts  
**We believe that SIMILAR is a promising step in the direction of active learning for realistic scenarios. The main limitation of our work is the dependence on good representations to compute similarity. In the future, we also look forward to approaches that can be used in cases where the characteristics of the dataset are completely unknown.

**Author  
**[**Suraj Kothawade**](https://personal.utdallas.edu/~snk170001/)