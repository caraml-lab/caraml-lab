---
title: AAAI-23 Lab Tutorial
nav:
  order: 2
  tooltip: AAAI-23 Lab Tutorial
---
<!-- # <i class="fa-solid fa-chalkboard-user"></i>  -->
# Subset Selection in Machine Learning: Hands-On Application with CORDS, DISTIL, SUBMODLIB, and TRUST
## Nathan Beck, Suraj Kothawade, Krishnateja Killamsetty, and Rishabh Iyer

## The 37th AAAI Conference on Artificial Intelligence, Washington DC, USA

## February 23rd, 2023 08:30 AM EST - 12:30 PM EST

<!-- # <i class="fa-duotone fa-location-dot"></i> The 37th AAAI Conference on Artificial Intelligence, Washington DC, USA -->
<!-- # <i class="fa-duotone fa-calendar-clock"></i> February 23rd, 2023 08:30 AM EST - 12:30 PM EST -->
{% include section.html %}
# Schedule

<ol>
  <li>**[CARAML Lab Overview and Projects (8:30 AM - 8:50 AM)]()**</li>
  <li>**[Submodlib and TRUST (8:50 AM - 10:00 AM)]()**
    <ol>
      <li>[Overview of Submodular functions and Submodular Information Measures]()</li>
      <li>[Modeling Capabilities of Submodular functions using Submodlib]()</li>
      <li>[Data Summarization using Submodlib]()</li>
      <li>[Overview and Motivation of TRUST]()</li>
      <li>[TRUST Demonstrations]()</li>
    </ol>
  </li>
  <li>**[DISTIL (10:00 AM - 10:30 AM)]**
    <ol>
      <li>[Introduction]()</li>
      <li>[Overview of Selection Strategies]()</li>
    </ol>
  </li>
  <li>**Break (10:30 AM - 11:00 AM)**</li>
  <li>**[DISTIL (11:00 AM - 11:30 AM)]**
    <ol>
      <li>[DISTIL Demonstrations]()</li>
    </ol>
  </li>
  <li>**[CORDS (11:30 AM - 12:30 AM)]()**
  <ol>
      <li>[Introduction]()</li>
      <li>[Use Cases of CORDS]()</li>
      <li>[Overview of Selection Strategies]()</li>
      <li>[CORDS Demonstrations]()</li>
    </ol>
  </li>
</ol>


# CARAML Lab Overview and Projects (8:30 AM - 8:50 AM)

To contextualize the various subset selection problems and toolkits that will be presented in this lab, a brief overview of CARAML Lab is given, including an overview of the pertinent projects that have been and are being conducted.

# Submodlib and TRUST (8:50 AM - 10:00 AM)

## Introduction

Overview of Data subset selection, Submodlib and Trust._

**SubModLib** is an easy-to-use, efficient and scalable Python library for submodular optimization with a C++ optimization engine. Submodlib finds its application in summarization, data subset selection, hyper parameter tuning, efficient training etc. Through a rich API, it offers a great deal of flexibility in the way it can be used. The SubModLib toolkit ia available at: [https://github.com/decile-team/submodlib](https://github.com/decile-team/submodlib)

**TRUST** is a toolkit which provides support for various targeted selection algorithms. Most real-world datasets have one or more charateristics that make its use on the state-of-the-art subset selection algorithms very difficult. Quite often, these characteristics are either known or can be easily found out. For example, real-world data is imbalanced, redudant and has samples that are of not of concern to the task at hand. Hence, there is a need to favor some samples while ignore the others. This is possible via different Submodular Information Measures based algorithms implemented in TRUST. The TRUST toolkit ia available at: [https://github.com/decile-team/trust](https://github.com/decile-team/trust)

## Overview of Submodular functions and Submodular Information Measures

We briefly discuss submodularity and information theory, and show the formulations of some submodular functions and their information measures.

## Modeling Capabilities of Submodular functions using Submodlib

We discuss different submodular functions implemented in Submodlib and the ease of use of obtaining a subset with just a few lines of code. We also show the modeling capabilities like representation, diversity and coverage of different[submodular functions](https://colab.research.google.com/drive/1f6T4jAdkMQAwmrnmOichs9sRPEHavk_1?usp=sharing), [Submodular Mutual Information](https://colab.research.google.com/drive/1NTWF8g83IC_JgEd9qjaxnB6GyhpD53rM?usp=sharing), [Conditional Gain](https://colab.research.google.com/drive/1BwBlTf7gOmUhIsI0ZpImiaI6fPylbyYn?usp=sharing), and [Conditional Mutual Information](https://colab.research.google.com/drive/1pUkZMmhJvfqZhje1tIKlM8IDFRElifYs?usp=sharing) functions.

## Data Summarization using Submodlib

We illustrate the usage of Submodlib for [visual data summarization](https://colab.research.google.com/drive/1pHhyfxMlueYgwX0DuKMI-eb98biN8ALA?usp=sharing). Particularly use submodular information measures for Generic, Query-focused and Privacy preserving summarization.

## Overview and Motivation of TRUST

We discuss different applications of Targeted Subset Selection using TRUST and the utility of using Submodular Information Measures for class imbalance, OOD and Redundancy. We illustrate real-world Medical imaging and Autonomous driving use cases that can be tackled using TRUST.

## TRUST Demonstrations

We show the utility of TRUST via an Interactive Application. The TRUST Interactive Application is available at: [https://github.com/decile-team/trust/tree/demo-app/demo](https://github.com/decile-team/trust/tree/demo-app/demo)

# DISTIL (10:00 AM - 10:30 AM)

## Introduction

DISTIL is a toolkit designed to ease the complexity of performing active learning in a number of settings. Here, a brief introduction of active learning is given along with an overview of DISTIL's salient features.

## Overview of Selection Strategies

To better understand the selection strategies implemented within DISTIL, brief overviews of each strategy are given, including how these strategies can be used via DISTIL.

# Break (10:30 AM - 11:00 AM)

# DISTIL (11:00 AM - 11:30 AM)

## DISTIL Demonstrations

To see how DISTIL is used in a full implementation, a number of working demonstrations are presented, spanning from medical imaging tasks to sentiment analysis tasks.

# CORDS (11:30 AM - 12:30 AM)

## Introduction

CORDS is a toolkit designed for efficient learning of machine learning models using subset selection. Here, we give a brief into various features implemented to CORDS library for ease of usage and various subset selection strategies incorporated for efficient learning. The CORDS toolkit is available at: [https://github.com/decile-team/cords](https://github.com/decile-team/cords)

## Use Cases of CORDS

We elaborate on use-cases of CORDS for different applications like supervised learning, semi-supervised learning, and hyper-parameter tuning. We further give a peek into future usecases that we plan to incorporate into CORDS shortly.

## Overview of Selection Strategies

To better understand the selection strategies implemented within CORDS, brief overviews of each strategy are given, including how these strategies can be used via CORDS.

## CORDS Demonstrations

To see how CORDS is used in a full implementation, a number of working demonstrations are presented for supervised learning, semi-supervised learning, and hyper-parameter tuning across different datasets from vision, text, and tabular domains. Demonstration notebooks shown for CORDS toolkit during the tutorial are available at: [https://github.com/decile-team/cords/tree/main/tutorial](https://github.com/decile-team/cords/tree/main/tutorial)

<!-- {% include search-info.html %} -->

<!-- {% include list.html data="works" component="work-excerpt" %} -->



