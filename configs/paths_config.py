DATASET_PATHS = {
	'CelebA_HQ': 'data/celeba_hq/',
	'AFHQ': 'data/afhq',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhq_dog_4m.pt",
    'CelebA_HQ': 'pretrained/celeba_hq.ckpt',
}

HYBRID_CONFIG = \
	{ 300: [0.4, 0.6, 0],
	    0: [0.15, 0.15, 0.7]}