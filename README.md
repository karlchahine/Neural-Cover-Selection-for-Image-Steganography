# Neural Cover Selection for Image Steganography
This repository contains the code for our paper [Neural Cover Selection for image Steganography](https://arxiv.org/abs/2410.18216) by Karl Chahine and Hyeji Kim (NeurIPS 2024). This repository is being continuously updated.

# Framework summary
Image steganography embeds secret bit strings within typical cover images, making them imperceptible to the naked eye yet retrievable through specific decoding techniques. The encoder takes as input a cover image ***x*** and a secret message ***m***, outputting a steganographic image ***s*** that appears visually similar to the original ***x***. The decoder then estimates the message ***m̂*** from ***s***. The setup is shown below, where _H_ and _W_ denote the image dimensions and the payload _B_ denotes the number of encoded bits per pixel (bpp).

<p style="margin-top: 30px;">
    <img src="steg_setup.png" alt="Model performance" width="600"/>
</p>

The effectiveness of steganography is significantly influenced by the choice of the cover image x, a process known as cover selection. Different images have varying capacities to conceal data without detectable alterations, making cover selection a critical factor in maintaining the reliability of the steganographic process.

Traditional methods for selecting cover images have three key limitations: (i) They rely on heuristic image metrics that lack a clear connection to steganographic effectiveness, often leading to suboptimal message hiding. (ii) These methods ignore the influence of the encoder-decoder pair on the cover image choice, focusing solely on image quality metrics. (iii) They are restricted to selecting from a fixed set of images, rather than generating one tailored to the steganographic task, limiting their ability to find the most suitable cover.

In this work, we introduce a novel, optimization-driven framework that combines pretrained generative models with steganographic encoder-decoder pairs. Our method guides the image generation process by incorporating a message recovery loss, thereby producing cover images that are optimally tailored for specific secret messages. We investigate the workings of the neural encoder and find it hides messages within low variance pixels, akin to the water-filling algorithm in parallel Gaussian channels. Interestingly, we observe that our cover selection framework increases these low variance spots, thus improving message concealment.

The DDIM cover-selection framework is illustrated below: 

<p style="margin-top: 30px;">
    <img src="DDIM_setup.png" alt="Model performance" width="600"/>
</p>

The initial cover image $\textbf{x}_0$ (where the subscript denotes the diffusion step) goes through the forward diffusion process to get the latent $\textbf{x}_T$ after _T_ steps. We optimize $\textbf{x}_T$ to minimize the loss ||***m*** - ***m̂***||. Specifically, $\textbf{x}_T$ goes through the backward diffusion process generating cover images that minimize the loss. We evaluate the gradients of the loss with respect to $\textbf{x}_T$ using backpropagation and use standard gradient based optimizers to get the optimal $\textbf{x}^*_T$ after some optimization steps. We use a pretrained DDIM, and a pretrained LISO, the state-of-the-art steganographic encoder and decoder from Chen et al. [2022]. The weights of the DDIM and the steganographic encoder-decoder are fixed throughout $\textbf{x}_T$'s optimization process.


# What is in this repository?
We provide a PyTorch implementation of our DDIM cover selection framework for AFHQ and CelebA-HQ. We also provide a script for computing performance metrics such as Error Rate, BRISQUE, PSNR and SSIM.

# Example
This code was run with the following dependencies.
```bash
Python 3.8.10
PyTorch 2.2.2
NumPy 1.24.4
CUDA 12.1

```
1. We include pretrained LISO steganographic encoder-decoder model weights for payloads $B=1,2,3,4$ bpp in the ```./logs``` directory.
2. In contrast, you need to download the weights of the Diffusion models pretrained on [AFHQ-Dog](https://arxiv.org/abs/1912.01865) or [CelebA-HQ](https://arxiv.org/abs/1710.10196) and put them in the ```./pretrained``` directory. Detailed instructions on how to download the models can be found [here](https://github.com/gwang-kim/DiffusionCLIP).
3. To download the datasets, you can use the following code. The data will be stored in ```./data/afhq``` and ```./data/celeba_hq```. More details can be found [here](https://github.com/gwang-kim/DiffusionCLIP).
```bash
# CelebA-HQ 
bash data_download.sh celeba_hq 

# AFHQ-Dog 
bash data_download.sh afhq 
```

4. To run the script for $10$ images of CelebA-HQ and a payload $B=3$ bpp, use the following command:

```bash
python3 main.py --config celeba.yml --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --bpp 3 --dataset-class CelebAHQ --num-images 10
```
The code will perform 50 iterations on each of the 10 images. This will generate a .csv file named ```CelebAHQ_3bpp.csv```.

5. To calculate the metrics such as Error Rate, BRISQUE, PSNR and SSIM, use the following command:
   
```bash
python3 calc_metrics.py --bpp 3 --dataset-class CelebAHQ
```
This script will find the iteration with the lowest error rate for every image, and calculate all image quality metrics for that corresponding iteration. It will also compute the metrics for iteration 0 (pre-optimization). To run experiments on AFHQ, simply replace ```--config celeba.yml``` by ```--config afhq.yml``` and ```--dataset-class CelebAHQ``` by ```--dataset-class AFHQ```.

# Citation
If you found our work helpful, please consider citing it.
```bibtex
@article{chahine2024neural,
  title={Neural Cover Selection for Image Steganography},
  author={Chahine, Karl and Kim, Hyeji},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

# Acknowledgments
Part of the code in this repository are based on the following public repositories:

- [https://github.com/gwang-kim/DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP)
- [https://github.com/cxy1997/LISO](https://github.com/cxy1997/LISO)

 


