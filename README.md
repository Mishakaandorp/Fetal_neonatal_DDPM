# Fetal&Neonatal-DDPM
[[Preprint on ArXiv](https://arxiv.org/abs/2501.19338)]

This repository contains the code for our Fetal&Neonatal-DDPM, a diffusion model for 3D pathological fetal and neonatal brain MRI synthesis from 3D segmentation maps. It also includes pretrained model weights and a label manipulation script that generates pathological labels from healthy labels.

The work focuses on improving segmentation of pathological regions in fetal and neonatal brain MRI by synthesizing pathological training data from healthy scans. This approach addresses the limited availability of annotated pathological data by leveraging generative methods for data augmentation.

The GIF below shows modified pathological label maps derived from healthy labels, wherefrom the pathological synthetic MRI are synthesized using our Fetal&Neonatal-DDPM.

![image34](https://github.com/user-attachments/assets/d2542b7e-e849-4f06-a2c4-c1e908f5c7ac)

This framework builds upon the Med-DDPM framework described by [Dorjsembe et al. (2024)](https://arxiv.org/abs/2305.18453), with code available on [Med-DDPM GitHub](https://github.com/mobaidoctor/med-ddpm).

This project was supported by the Swiss National Science Foundation (SNSF), grant Nr. IZKSZ3_218590.

## 🛠️ Setup 

Ensure you have the following libraries installed for training and generating images using python 3.8:

- **Convert3D Version 1.1.0**: [Convert3D](https://sourceforge.net/projects/c3d/files/c3d/Experimental/)

```
pip install -r requirements.txt
```
### 🚀 Run example

We have provided example images of the [FeTA2021](https://feta.grand-challenge.org/feta-2021/#:~:text=The%20Fetal%20Brain%20Tissue%20Annotation,of%20developing%20human%20brain%20tissues.) MICCAI Challenge with available code to label augment and generate synthetic MRIs.
Running ``` /scripts/synthetic_label_map_generation_pipeline_example.sh ``` does the following:
- Generate pathological label augmentations on the provided (healthy) label image.
- Relabel and resize to image dimensions 160x160x160 for Fetal/Neonatal MRI synthesis.
- Generate MRI images utilizing Fetal&Neonatal-DDPM, the trained model (See model weights below).
- Upsample and upscale to original dimensions (Note: it needs to have the original MRI to know the scaling factor).

### 🚀 Generate your own synthetic fetal and neonatal MRIs

You can add your own label images and MRIs to ``` /Dataset/... ```.
Change the dataset in ``` /scripts/synthetic_label_map_generation_pipeline_example.sh ``` and run the pipeline. 

### 🧠 Model Weights

Get our Fetal&Neonatal-DDPM model weights for fetal and neonatal MRI synthesis from the link below:

[Download Model Weights](UPDATE...)

After downloading, place the files under the ``` /Fetal_Neonatal_DDPM/Trained_models/ ``` directory.

## 📜 Citation

Our manuscript is currently under review:
"Pathological MRI Segmentation by Synthetic Pathological Data Generation in Fetuses and Neonates".

You can refer to the preprint version of the manuscript on arXiv:

📄 https://arxiv.org/abs/2501.19338

## 💡 Acknowledgements

Gratitude to these repositories:

1. [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
2. [guided-diffusion](https://github.com/openai/guided-diffusion)
3. [med-ddpm](https://github.com/mobaidoctor/med-ddpm)




