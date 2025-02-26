---
title: 'TensorFlow MRI: An Open-Source TensorFlow Library for MRI Applications'
tags:
  - Magnetic Resonance Imaging
  - MRI
  - Medical Imaging
  - Reconstruction
  - Post-processing
  - Segmentation
  - Machine Learning
  - Python
authors:
  - name: Javier Montalt-Tordera
    orcid: 0000-0003-1956-1409
    affiliation: 1
  - name: Vivek Muthurangu
    orcid: 0000-0002-4292-6456
    affiliation: 1
  - name: Olivier Jaubert
    orcid: 0000-0002-7854-4150
    affiliation: 1
  - name: Rebecca Baker
    orcid: 0000-0002-5016-7912
    affiliation: "1, 2"
  - name: Tina Yao
    orcid: 0000-0003-0470-311X
    affiliation: "1, 3"
  - name: Michele Pascale
    orcid: 0000-0001-6853-2685
    affiliation: 1
  - name: Christopher Watson
    affiliation: 1
  - name: Anirudh Raman
    orcid: 0009-0005-8301-0742
    affiliation: 1
  - name: Ruaraidh Campbell
    orcid: 0000-0002-7457-9065
    affiliation: "1, 3"
  - name: Jennifer Steeden
    orcid: 0000-0002-9792-2022
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: UCL Centre for Translational Cardiovascular Imaging, University College London, Zayed Centre for Research, London, UK.
   index: 1
   ror: 02jx3x895   
 - name: UCL Centre for Medical Imaging, University College London, London, UK.
   index: 2
 - name: UCL Institute of Health Informatics, University College London, London, UK.
   index: 3
date: 26 February 2025
bibliography: paper.bib
---

# Summary

Open-source reconstruction software is vital for the ongoing development of accelerated Magnetic Resonance Imaging (MRI). Current state-of-the-art techniques are complex and often time-consuming, which hinders uptake of accelerated techniques in the clinical environment. Recently, Deep Learning (DL) has revolutionized MRI reconstruction, reducing reconstruction time while maintaining high image quality. However, deployment in the research and clinical environment requires a of combination of modern DL frameworks, such as TensorFlow, with MRI reconstruction operations. Thus, we present *‘TensorFlow MRI’*, an open-source library of TensorFlow operators for rapid computational MRI, particularly focussed on reconstruction (DL and non-DL) and post-processing for MRI data. 

# Statement of need

*‘TensorFlow MRI’* is a library of TensorFlow operators for computational magnetic resonance imaging. Its purpose is to simplify the creation of end-to-end MRI reconstruction and processing pipelines, within a unified platform.
*‘TensorFlow’* is a highly efficient scientific computing framework, designed for machine learning. It has two key advantages: (i) It can be run easily on heterogenous hardware, including CPUs and GPUs, with no or minimal code changes, and, (ii) It enables automatic differentiation, which is essential for deep learning, but also convenient for other optimization problems, such as iterative types of MRI reconstruction. 
However, MRI reconstructions often rely on MRI-specific software, which does not typically interface easily with TensorFlow’s computing model. We developed *‘TensorFlow MRI’* to address this, by extending the TensorFlow ecosystem to seamlessly incorporate MRI-specific functionality.
*‘TensorFlow MRI’* features include: MRI-adapted linear algebra and convex optimization frameworks, differentiable operators and utilities for k-space sampling, parallel imaging and compressed sensing, and other signal processing operators that are common in MRI. It also provides a collection of frequently used ML models, layers, metrics and loss functions for image reconstruction and post-processing.
TensorFlow MRI also contributes a native non-uniform fast Fourier transform (NUFFT), a basic operation in non-Cartesian MRI reconstruction. We use and adapt the FINUFFT library [@BarnettNUFFT][@ShihNUFFT] and contribute interfacing code and a custom gradient implementation for seamless use within TensorFlow.
Importantly, ML and non-ML components integrate seamlessly in *‘TensorFlow MRI’*. For example, using *‘TensorFlow MRI’* it is possible to include a parallel imaging operator within a neural network, or to use a trained prior as part of an iterative reconstruction without any additional interfacing.
*‘TensorFlow MRI’* is primarily written in Python. It is easy to use, understand and extend. The framework is well-documented with extensive docs, detailed guides and tutorials to enable users to get started easily. 
In addition to its user-friendly interface, *‘TensorFlow MRI’* has competitive performance and can run on heterogenous and distributed systems.
Furthermore, *‘TensorFlow MRI’* can be deployed in streaming reconstructions at clinical MR scanners, e.g. using Gadgetron’s external language [@HansenGadgetron] interface, for seamless online use in the clinical environment.

# Research projects

*‘TensorFlow MRI’* has been used in several peer-reviewed publications, which are relevant for clinical MRI. Importantly, in all these applications, *‘TensorFlow MRI’* reconstruction has been seamlessly integrated with Gadgetron [@HansenGadgetron] for online low-latency reconstruction. The following three papers use *‘TensorFlow MRI’* for image reconstruction, with open-source code available:

[@JaubertHyperSlice] *‘HyperSLICE: HyperBand optimized spiral for low-latency interactive cardiac examination’*
Open source code available at: 
[https://github.com/mrphys/HyperSLICE.git](https://github.com/mrphys/HyperSLICE.git)
In this paper we used *‘TensorFlow MRI’* to train DL models to reconstruct rapid, interactive cardiac MRI data for interventional imaging. The code provides instructions on how to use hyperband to enable DL optimisation of the spiral acquisition trajectory, as well as model training for DL reconstruction using a FastDVNet.  

[@JaubertNaturalVideos] *‘Training deep learning based dynamic MR image reconstruction using open-source natural videos’*
Open source code available at: [https://github.com/mrphys/Image_Reconstruction_Inter4k.git](https://github.com/mrphys/Image_Reconstruction_Inter4k.git)
In this paper we used *‘TensorFlow MRI’* to train DL models using natural video data, for reconstruction of real-time MRI data. The repository uses Cartesian, Radial and Spiral real-time acquisition sampling trajectories, with reconstructions from VarNet, 3D-UNet and FastDVDNet models.
 
[@BakerDenoising] *‘Rapid 2D 23Na MRI of the calf using a denoising convolutional neural network’*
Open source code available at: [https://github.com/mrphys/sodium_MRI_DnCNN.git](https://github.com/mrphys/sodium_MRI_DnCNN.git) 
In this paper we used *‘TensorFlow MRI’* to train a denoising network to improve the image quality for Sodium MRI data.

# Financial support
This work was funded by UKRI (grant MR/S032290/1).

# References