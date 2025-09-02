# Fetal&Neonatal-DDPM
This repository contains the code for our Fetal&Neonatal-DDPM, a diffusion model for 3D pathological fetal and neonatal brain MRI synthesis from 3D segmentation maps. It also includes pretrained model weights and a label manipulation script that generates pathological labels from healthy labels.

The work focuses on improving segmentation of pathological regions in fetal and neonatal brain MRI by synthesizing pathological training data from healthy scans. This approach addresses the limited availability of annotated pathological data by leveraging generative methods for data augmentation.

The Giff below shows modified pathological label maps derived from healthy labels, wherefrom the pathological synthetic MRI are synthesized using our Fetal&Neonatal-DDPM.

![image34](https://github.com/user-attachments/assets/d2542b7e-e849-4f06-a2c4-c1e908f5c7ac)


This project was supported by the Swiss National Science Foundation (SNSF), grant Nr. IZKSZ3_218590.

## 🛠️ Setup 

Ensure you have the following libraries installed for training and generating images using python 3.8:

c3d Version 1.1.0

pip install -r requirements.txt

## 🚀 Run on Your Own Dataset

The code is currently under development and will be publicly released upon publication of our manuscript:
"Pathological MRI Segmentation by Synthetic Pathological Data Generation in Fetuses and Neonates" (under review).

In the meantime, you can refer to the preprint version of the manuscript on arXiv:

📄 https://arxiv.org/abs/2501.19338

# Contact
For questions, please contact:
📧 Misha.Kaandorp@kispi.uzh.ch


