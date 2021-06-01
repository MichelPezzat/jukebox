**Status:** Archive (code is provided as-is, no updates expected)

# Jukebox
Code modified of "Jukebox: A Generative Model for Music" using NVAE architecture instead of VQ-VAE

[Jukebox-Paper](https://arxiv.org/abs/2005.00341) 
[NVAE-Paper](https://arxiv.org/abs/2007.03898)
[Blog](https://openai.com/blog/jukebox) 
[Explorer](http://jukebox.openai.com/) 
[Colab](https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb) 

# Install
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html    
    
``` 
#Requirments are the as in the original Jukebox
# Required: Sampling
conda create --name jukebox python=3.7.5
conda activate jukebox
conda install mpi4py=3.0.3 # if this fails, try: pip install mpi4py==3.0.3
conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch
git clone https://github.com/openai/jukebox.git
cd jukebox
pip install -r requirements.txt
pip install -e .

# Required: Training
conda install av=7.0.01 -c conda-forge 
pip install ./tensorboardX
 
# Optional: Apex for faster training with fused_adam
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

# Sampling
Yet to be done


# Training
## NVAE
To train a small nvae, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_nvae,all_fp16 --name=small_nvae --sample_length=262144 
--bs=2 --audio_files_dir={audio_files_dir} --labels=False --train --aug_shift --aug_blend  
--save_iters=1000 --sr=44100 --epochs=200

```
Here, `{audio_files_dir}` is the directory in which you can put the audio files for your dataset, and `{ngpus}` is number of GPU's you want to use to train. 
Checkpoints are stored in the `logs` folder. You can monitor the training by running Tensorboard
```
tensorboard --logdir logs
```
# Changes made to original NVAE
NVAE architecture is practically the same from the original, but some changes were made for the audio version:
* All convs were changed from 2D to 1D.

* The sampling is made from mixture of logistics distribution as in [Parallel-Wavenet](https://arxiv.org/pdf/1711.10433.pdf)

* Autoregressive Convs were changed to causal as in [Clarinet](https://github.com/ksw0306/ClariNet/tree/df31b4c4ea78d3b52274632791d0a2c6e8ed6b64)

More changes are expected to be made, as this project is still on experimental stage.

# Citation

Please cite using the following bibtex entry:

```
@article{dhariwal2020jukebox,
  title={Jukebox: A Generative Model for Music},
  author={Dhariwal, Prafulla and Jun, Heewoo and Payne, Christine and Kim, Jong Wook and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2005.00341},
  year={2020}
}
```

# License 
[Noncommercial Use License](./LICENSE) 

It covers both released code and weights. 

