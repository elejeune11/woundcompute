# Wound Compute Repository

[![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![tests](https://github.com/elejeune11/woundcompute/workflows/tests/badge.svg)](https://github.com/elejeune11/woundcompute/actions) [![codecov](https://codecov.io/gh/elejeune11/woundcompute/branch/main/graph/badge.svg?token=EVCCPWCUE7)](https://codecov.io/gh/elejeune11/woundcompute)


### Table of Contents
* [Project Summary](#summary)
* [Project Roadmap](#roadmap)
* [Installation Instructions](#install)
* [Current Functionality](#functionality)
* [Tutorial](#tutorial)
* [References to Related Work](#references)
* [Contact Information](#contact)


### Project Summary <a name="summary"></a>

This software is designed to analyze experimental data from micro-tissue wound experiments (see: [references](#references)). The goal of our software is to extract quantitative information from these images and movies. For example, we can automatically identify the wound region which allows us extract properties such as wound area, major axis length, and minor axis length with respect to time.

<p align = "center">
<img alt="schematic of experimental system" src="tutorials/figs/schematic.png" width="21%" />
<img alt="brightfield with contour" src="tutorials/figs/brightfield_contour.gif" width="35%" />
<img alt="fluorescent with contour" src="tutorials/figs/fluorescent_contour.gif" width="35%" />
</p>

We are also adding additional funcitonality for tracking tissue deformation between frames -- stay tuned!

### Project Roadmap <a name="roadmap"></a>

We plan to develop and disseminate an "all-purpose" software for data curation and analysis from micro-tissue wound experiments. The roadmap for this collaborative endeavor is as follows:

`Preliminary Dataset + Software` $\mapsto$ `Larger Dataset + Software Testing and Validation` $\mapsto$ `Published Software Package` $\mapsto$ `Published Validation Examples and Tutorial` $\mapsto$ `Automated Analysis of High-Throughput Experiments`

At present (**September 2022**) we are in the process of testing and validating our software on larger datasets. Specifically, we are (1) identifying scenarios where our approach fails and creating functions to accomodate them, and (2) comparing software results to previous manual approaches for extracting quantitative information. We will continue to update this repository as the project progresses.

### Installation Instructions <a name="install"></a>


### Current Functionality <a name="functionality"></a>


### Tutorial <a name="tutorial"></a>


### References to Related Work <a name="references"></a>
* Sakar, M. S., Eyckmans, J., Pieters, R., Eberli, D., Nelson, B. J., & Chen, C. S. (2016). Cellular forces and matrix assembly coordinate fibrous tissue repair. Nature communications, 7(1), 1-8. https://doi.org/10.1038/ncomms11036 -- source of schematic diagram in the [Project Summary](#summary).


### Contanct Information <a name="contact"></a>

At present, this repository is public to facilitate collaboration between the Lejeune + Eyckmans research groups. However, we welcome feedback and interest from others.

For information about this software, please get in touch with [Emma Lejeune](https://www.bu.edu/eng/profile/emma-lejeune/). For information about this experimental system, please get in touch with [Jeroen Eyckmans](https://www.bu.edu/eng/profile/jeroen-eyckmans-phd/).