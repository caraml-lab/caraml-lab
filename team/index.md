---
title: Team
nav:
  order: 5
  tooltip: About our team
---

# <i class="fas fa-users"></i>Team

We are a diverse team of highly motivated and collaborative researchers working on various aspects of machine learning, focusing on going beyond accuracy and achieving other desiderata such as compute and memory efficiency, human interaction, label efficiency, robustness, fairness, etc. As a team, we value all our members' diverse views and experiences, and they strengthen us. Thanks to a caring team, we have a friendly, inclusive, and healthy environment in our lab. For each team member, success means something different, and we encourage each other to develop and pursue individual passions. Our current team includes graduate students, undergraduate students.

{% include section.html %}

{%
  include list.html
  data="members"
  component="portrait"
  filters="role: pi"
%}
{%
  include list.html
  data="members"
  component="portrait"
  filters="role: phd"
%}
{%
  include list.html
  data="members"
  component="portrait"
  filters="role: programmer"
%}
{:.center}

{% include section.html background="images/banner2.jpg" dark=true%}

Currently, we are seeking highly motivated students with a broad interest in machine learning and optimization in general. In addition, if you are interested in our work and want to discuss it further, please do not hesitate to contact us. In addition, we welcome collaborations with researchers on the topics of Efficient ML, Active Learning, Semi-Supervised Learning, Robust and Fair ML.

{%
  include link.html
  icon="fas fa-hands-helping"
  text="Join the Team"
  link="join"
  style="button"
%}
{:.center}

{% include section.html %}

## Funding

Our work is made possible by funding from several organizations.
{:.center}

{%
  include gallery.html
  style="square"

  image1="images/nsf_logo.png"
  link1="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2106937&HistoricalAwards=false"
  tooltip1="National Science Foundation"

  image2="images/adobe_logo.png"
  link2="https://www.adobe.com/"
  tooltip2="Adobe"

  image3="images/google_logo.png"
  link3="https://about.google/intl/ALL_us/"
  tooltip3="Google"

  image4="images/utd_logo.jpg"
  link4="https://www.utdallas.edu/"
  tooltip4="The University of Texas at Dallas"
%}
