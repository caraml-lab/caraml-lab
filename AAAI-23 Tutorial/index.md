---
title: AAAI-23 Lab Tutorial
nav:
  order: 2
  tooltip: AAAI-23 Lab Tutorial
---
# Subset Selection in Machine Learning: Hands-On Application with CORDS, DISTIL, SUBMODLIB, and TRUST
### The 37th AAAI Conference on Artificial Intelligence, Washington DC, USA
### February 23rd, 2023 08:30 AM EST - 12:30 PM EST

{% include section.html dark=true%}
*   [About Tutorial](#about-tutorial)
*   [Schedule](#schedule)
*   [Organizers](#organizers)
*   [Contact Us](#contact-us)

{% include section.html%}
# About Tutorial {#about-tutorial}
Machine learning -- specifically, deep learning -- has transformed numerous application domains like computer vision and video analytics, speech recognition, natural language processing, and so on. As a result, significant focus of researchers in the last decade has been on obtaining the most accurate models, often matching and sometimes surpassing human level performance in these areas. However, deep learning is also unlike human learning in many ways. To achieve the human level performance, deep models require large amounts of labeled training data, several GPU instances to train, and massive size models (ranging from hundreds of millions to billions of parameters). In addition, they are often not robust to noise, imbalance, and out of distribution data and can also easily inherit the biases in the training data. Motivated by these desiderata and many more, we will present a rich framework of PyTorch toolkits for subset selection and coreset-based approaches that satisfy them. We will begin by providing a brief introduction of these desiderata and how they are handled by the methods implemented in our toolkits. Next, we will then introduce each toolkit -- CORDS, DISTIL, SUBMODLIB, and TRUST -- by highlighting their field of application and by walking through enriching, real-scenario tutorials showcasing their ease of use and capability for satisfying the above desiderata. In particular, we will provide hands-on experiences for compute-efficient training through CORDS; label-efficient training through DISTIL; powerful submodular optimization through SUBMODLIB; and robust, fair, and personalized learning via TRUST. We will present these toolkits under the larger cooperative effort of [DECILE](www.decile.org), highlighting the rich community of researchers and practitioners supporting these toolkits. Our toolkits are available in the following [git link](https://github.com/decile-team).

## Lab Tutorial Goals:
The goal of this lab is to provide and highlight a toolkit framework for solving many real-world complications within deep learning using subset selection and coreset-based approaches. Specifically, we believe that providing these toolkits and enriching hands-on experience regarding their use will enable researchers and practitioners to think beyond just improving the model accuracy and in broader yet important aspects like Green AI, fairness, robustness, personalization, data efficiency, and so on. Furthermore, the hands-on demonstrations will also be useful to students and researchers from industry to get oriented in and practically started with the subject matter of each toolkit and its related aspects. By introducing these toolkits, we also hope to build a larger community around their usage, which will help strengthen their applicability across deep learning and help connect the interests of like-minded researchers and practitioners.

***Key Takeaways:***

1. How to learn machine learning models in real world settings achieving near optimal performanc while achieving other desiderata like compute efficiency, data efficiency, robustness, fairness, personalization, etc.
2. How is the industry/academia tackling this challenge and doing Data-Efficient Learning? 
3.  Hands-on session using PyTorch for Data-Efficient Learning with the following State-of-the-Art toolkits:
    1.  CORDS for Compute Efficient Learning
    2.  DISTIL for Active Learning
    3.  SUBMODLIB for Submodular Optimization
    4.  TRUST for Targeted Learning
       
## Who should attend this Lab Tutorial Session?

 The target audience of this lab is practitioners in deep learning and machine learning as well as researchers working on more theoretical areas in optimization in machine learning. Further, we encourage participation from individuals working in the industry, academia, and students with experience and/or interest in Machine Learning, Deep Learning, and its efficient application for solving problems quickly. Finally, this session is highly relevant for academics who wish to learn how to design efficient, resilient, and computationally efficient models with limited GPU resources.

{% include section.html dark=true%}
# Schedule {#schedule}

1. **[An Introduction into Subset Selection for Data-Efficient ML (8:30 AM - 8:50 AM)](#schedule-1)**
2. **[Submodlib and TRUST (8:50 AM - 10:00 AM)](#schedule-2)**
    1. [Introduction of Submodlib and TRUST](#schedule-2.1) 
    2. [Overview of Submodular functions and Submodular Information Measures](#schedule-2.2)
    3. [Modeling Capabilities of Submodular functions using Submodlib](#schedule-2.3)
    4. [Data Summarization using Submodlib](#schedule-2.4)
    5. [Overview and Motivation of TRUST](#schedule-2.5)
    6. [TRUST Demonstrations](#schedule-2.6)    
3. **[DISTIL (10:00 AM - 10:30 AM)](#schedule-3)**
    1. [Introduction of DISTIL](#schedule-3.1)
    2. [Overview of Selection Strategies](#schedule-3.2)
4. **[Break (10:30 AM - 11:00 AM)](#break)**
5. **[DISTIL (11:00 AM - 11:30 AM)](#schedule-3(2nd))**
    1. [DISTIL Demonstrations](#schedule-3.3)
6. **[CORDS (11:30 AM - 12:30 AM)](#schedule-4)**
    1. [Introduction of CORDS](#schedule-4.1)
    2. [Use Cases of CORDS](#schedule-4.2)
    3. [Overview of Selection Strategies](#schedule-4.3)
    4. [CORDS Demonstrations](#schedule-4.4)


# An Introduction into Subset Selection for Data-Efficient ML (8:30 AM - 8:50 AM) {#schedule-1}

To contextualize the various subset selection problems and toolkits that will be presented in this lab, a brief overview of CARAML Lab is given, including an overview of the pertinent projects that have been and are being conducted.

# SubModlib and TRUST (8:50 AM - 10:00 AM) {#schedule-2}

## Introduction {#schedule-2.1}

Overview of Data subset selection, Submodlib and Trust.

**SubmodLib** is an easy-to-use, efficient and scalable Python library for submodular optimization with a C++ optimization engine. Submodlib finds its application in summarization, data subset selection, hyper parameter tuning, efficient training etc. Through a rich API, it offers a great deal of flexibility in the way it can be used. The SubModLib toolkit ia available at: [https://github.com/decile-team/submodlib](https://github.com/decile-team/submodlib)

**TRUST** is a toolkit which provides support for various targeted selection algorithms. Most real-world datasets have one or more charateristics that make its use on the state-of-the-art subset selection algorithms very difficult. Quite often, these characteristics are either known or can be easily found out. For example, real-world data is imbalanced, redudant and has samples that are of not of concern to the task at hand. Hence, there is a need to favor some samples while ignore the others. This is possible via different Submodular Information Measures based algorithms implemented in TRUST. The TRUST toolkit ia available at: [https://github.com/decile-team/trust](https://github.com/decile-team/trust)

## Overview of Submodular functions and Submodular Information Measures {#schedule-2.2}

We briefly discuss submodularity and information theory, and show the formulations of some submodular functions and their information measures.

## Modeling Capabilities of Submodular functions using Submodlib {#schedule-2.3}

We discuss different submodular functions implemented in Submodlib and the ease of use of obtaining a subset with just a few lines of code. We also show the modeling capabilities like representation, diversity and coverage of different[submodular functions](https://colab.research.google.com/drive/1f6T4jAdkMQAwmrnmOichs9sRPEHavk_1?usp=sharing), [Submodular Mutual Information](https://colab.research.google.com/drive/1NTWF8g83IC_JgEd9qjaxnB6GyhpD53rM?usp=sharing), [Conditional Gain](https://colab.research.google.com/drive/1BwBlTf7gOmUhIsI0ZpImiaI6fPylbyYn?usp=sharing), and [Conditional Mutual Information](https://colab.research.google.com/drive/1pUkZMmhJvfqZhje1tIKlM8IDFRElifYs?usp=sharing) functions.

## Data Summarization using Submodlib {#schedule-2.4}

We illustrate the usage of Submodlib for [visual data summarization](https://colab.research.google.com/drive/1pHhyfxMlueYgwX0DuKMI-eb98biN8ALA?usp=sharing). Particularly use submodular information measures for Generic, Query-focused and Privacy preserving summarization.

## Overview and Motivation of TRUST {#schedule-2.5}

We discuss different applications of Targeted Subset Selection using TRUST and the utility of using Submodular Information Measures for class imbalance, OOD and Redundancy. We illustrate real-world Medical imaging and Autonomous driving use cases that can be tackled using TRUST.

## TRUST Demonstrations {#schedule-2.6}

We show the utility of TRUST via an Interactive Application. The TRUST Interactive Application is available at: [https://github.com/decile-team/trust/tree/demo-app/demo](https://github.com/decile-team/trust/tree/demo-app/demo)

# DISTIL (10:00 AM - 10:30 AM) {#schedule-3}

## Introduction {#schedule-3.1}

DISTIL is a toolkit designed to ease the complexity of performing active learning in a number of settings. Here, a brief introduction of active learning is given along with an overview of DISTIL's salient features.

## Overview of Selection Strategies {#schedule-3.2}

To better understand the selection strategies implemented within DISTIL, brief overviews of each strategy are given, including how these strategies can be used via DISTIL.

# Break (10:30 AM - 11:00 AM) {#break}

# DISTIL (11:00 AM - 11:30 AM) {#schedule-3(2nd)}

## DISTIL Demonstrations {#schedule-3.3}

To see how DISTIL is used in a full implementation, a number of working demonstrations are presented, spanning from medical imaging tasks to sentiment analysis tasks.

# CORDS (11:30 AM - 12:30 AM) {#schedule-4}

## Introduction {#schedule-4.1}

CORDS is a toolkit designed for efficient learning of machine learning models using subset selection. Here, we give a brief into various features implemented to CORDS library for ease of usage and various subset selection strategies incorporated for efficient learning. The CORDS toolkit is available at: [https://github.com/decile-team/cords](https://github.com/decile-team/cords)

## Use Cases of CORDS {#schedule-4.2}

We elaborate on use-cases of CORDS for different applications like supervised learning, semi-supervised learning, and hyper-parameter tuning. We further give a peek into future usecases that we plan to incorporate into CORDS shortly.

## Overview of Selection Strategies {#schedule-4.3}

To better understand the selection strategies implemented within CORDS, brief overviews of each strategy are given, including how these strategies can be used via CORDS.

## CORDS Demonstrations {#schedule-4.4}

To see how CORDS is used in a full implementation, a number of working demonstrations are presented for supervised learning, semi-supervised learning, and hyper-parameter tuning across different datasets from vision, text, and tabular domains. Demonstration notebooks shown for CORDS toolkit during the tutorial are available at: [https://github.com/decile-team/cords/tree/main/tutorial](https://github.com/decile-team/cords/tree/main/tutorial)

{% include section.html %}
# Organizers {#organizers}
<div id="team" class='container container-team mx-auto mt-5 col-md-10 mt-100'>
<div class="row justify-content-center pb-5">
<div class="card col-md-3 mt-100">
    <div class="card-content">
        <div class="card-body p-0">
            <div class="profile"> <img src="../images/nathan-beck.jpg"  width="120" 
     height="120"> </div>
            <a href="">
                <div class="card-title mt-4"> Nathan Beck<br /> <small>PhD Student, University of Texas at Dallas
                    </small>
                </div>
            </a>
          </div>
      </div>
</div>


<div class="card col-md-3 mt-100">
    <div class="card-content">
        <div class="card-body p-0">
            <div class="profile"> <img src="../images/suraj-kothawade.jpg"  width="120" 
     height="120"> </div>
            <a href="">
                <div class="card-title mt-4"> Suraj Kothawade<br /> <small>PhD Student, University of Texas at Dallas
                    </small>
                </div>
            </a>
          </div>
      </div>
</div>

<div class="card col-md-3 mt-100">
    <div class="card-content">
        <div class="card-body p-0">
            <div class="profile"> <img src="../images/krishnateja-killamsetty.jpg"  width="120" 
     height="120"> </div>
            <a href="">
                <div class="card-title mt-4"> Krishnateja Killamsetty<br /> <small>PhD Student, University of Texas at Dallas
                    </small>
                </div>
            </a>
          </div>
      </div>
</div>

<div class="card col-md-3 mt-100">
    <div class="card-content">
        <div class="card-body p-0">
            <div class="profile"> <img src="../images/rishabh-iyer.jpg"  width="120" 
     height="120"> </div>
            <a href="">
                <div class="card-title mt-4"> Rishabh Iyer<br /> <small>Assistant Professor, University of Texas at Dallas
                    </small>
                </div>
            </a>
          </div>
      </div>
</div>
</div>
</div>

{% include section.html dark=true%}
# Contact Us {#contact-us}

{%
  include link.html
  type="email"
  icon=""
  text="rishabh.iyer@utdallas.edu"
  tooltip=""
  link="rishabh.iyer@utdallas.edu"
  style="button"
%}
{%
  include link.html
  type="address"
  icon=""
  text="Google Maps"
  tooltip="Our location on Google Maps for easy navigation"
  link="https://goo.gl/maps/6gUTngJmW5aHtxFq9"
  style="button"
%}
{:.center}

<!-- {% include search-info.html %} -->

<!-- {% include list.html data="works" component="work-excerpt" %} -->



