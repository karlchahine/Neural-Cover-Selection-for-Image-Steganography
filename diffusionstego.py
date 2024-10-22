import os
import numpy as np
import torch
from torch import nn
from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from configs.paths_config import MODEL_PATHS
from torch.nn.functional import binary_cross_entropy_with_logits
from liso.utils import calc_psnr, calc_ssim, to_np_img
import csv
import os
from imageio import imwrite

def val(cover, payload, model_steg, itr, csv_file_path, fieldnames, index, classvec, num_bits):

        with torch.no_grad():
            generated, payload, decoded, grads, ptbs = model_steg._encode_decode(cover, quantize=True, payload = payload)

        cover = to_np_img(cover[0])

        _psnrs = [calc_psnr(cover,to_np_img(x[0], dtype=np.float32)) for x in generated]
        with torch.no_grad():
            _errors = [float(1 - (x >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()) * 100 for x in decoded]
        costs = np.array([-y if x == 0 else x for x, y in zip(_errors, _psnrs)])
        best_idx = np.argmin(costs)

        generated = to_np_img(generated[best_idx][0])

        error = _errors[best_idx]

        ssim = calc_ssim(cover, generated.astype(np.float32))
        psnr = calc_psnr(cover, generated.astype(np.float32))

        imwrite("./Cover_images/" + classvec + "/" + str(num_bits) + "bpp/image" + str(index) + "_" + str(itr) + ".png", cover)
        imwrite("./Stego_images/" + classvec + "/" + str(num_bits) + "bpp/image" + str(index) + "_" + str(itr) + ".png", generated)

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row = {'index': index, 'iteration': itr, 'error': error, 'psnr': psnr, 'ssim': ssim}
            writer.writerow(row)

def seq_loss(loss_func, generated, target, gamma=0.8, normalize=False):
    weights = [gamma ** x for x in range(len(generated)-1, -1, -1)]
    loss = 0
    for w, x in zip(weights, generated):
        loss += loss_func(x, target) * w
    if normalize:
        loss /= sum(weights)
    return loss

class DiffusionSteg(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def finetune_steg_latent(self, model_steg, random_image, index, class_vec, num_bits):

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ","IMAGENET"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")

        # ----------- Model -----------#
        print("Improved diffusion Model loaded.")
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train

        fieldnames = ['index', 'iteration', 'error', 'psnr', 'ssim']

        csv_file_path = class_vec + '_' + str(num_bits) + 'bpp.csv'
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # Move the image to the specified device
        x0 = random_image.to(self.config.device)

        p = 0.5
        prob_tensor = torch.full((1, num_bits, 256, 256), p)
        payload = torch.bernoulli(prob_tensor).cuda()

        ##initial cover error
        model_steg.encoder.iters = 150
        model_steg.encoder.step_size = 0.1
        val(x0, payload, model_steg, 0, csv_file_path, fieldnames, index, class_vec, num_bits)

        model_steg.encoder.iters = 15 
        
        ##Compute Latent
        x = x0.clone()
        model.eval()
        with torch.no_grad():
            for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):

                t = (torch.ones(n) * i).to(self.device)
                t_prev = (torch.ones(n) * j).to(self.device)

                x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                    logvars=self.logvar,
                                    sampling_type='ddim',
                                    b=self.betas,
                                    eta=0.0,
                                    learn_sigma=learn_sigma)

            x_lat0 = x.clone()

        seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
        seq_train = [int(s) for s in list(seq_train)]
        seq_train_next = [-1] + list(seq_train[:-1])

        x_lat = nn.Parameter(x_lat0.clone())
        
        optimizer = torch.optim.Adam([x_lat], weight_decay=0, lr=self.args.lr_lat_opt)
        
        num_iters = 50

        with torch.set_grad_enabled(True):
            for itr in range(1, num_iters+1):
                x = x_lat
                optimizer.zero_grad()

                for i, j in zip(reversed(seq_train), reversed(seq_train_next)):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=0.5,
                                        learn_sigma=learn_sigma)
                    
                generated, payload, decoded, grads, ptbs = model_steg._encode_decode(x, quantize=False, payload = payload)

                model_steg.encoder.iters = 150
                model_steg.encoder.step_size = 0.1
                val(x, payload, model_steg, itr, csv_file_path, fieldnames, index, class_vec, num_bits)

                model_steg.encoder.iters = 15
                
                decoder_loss = seq_loss(binary_cross_entropy_with_logits, decoded, payload, gamma=0.8)
                        
                decoder_loss.backward()
                optimizer.step()

            
