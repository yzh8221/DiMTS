# DiM-TS: Bridge the Gap between Selective State Space Models and Time Series for Generative Modeling

The repo is the official implementation for the paper: https://arxiv.org/pdf/2511.18312

## DiM-TS Architecture

![image](https://github.com/yzh8221/DiMTS/blob/master/Figure/DiMTS_Main.png)

## Running the Code

The following instructions explain how to run the code. 

#### Environment & Libraries

The full libraries list is provided as `requirements.txt`. Please create a virtual environment and run

~~~bash
pip install -r requirements.txt
~~~

Install dependent packages

~~~bash
pip install --upgrade pip
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
cd kernels/dwconv2d && python3 setup.py install --user
~~~

#### Training

For training, you can reproduce the experimental results by runing

~~~bash
python main.py --name {name} --config_file {config.yaml} --gpu 0 --train
~~~

#### Unconstrained Sampling

Please use the saved model for sampling by running

```bash
python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 0 --milestone {checkpoint_number}
```
## Channel Permutation Scanning

[channel_permutation.ipynb](https://github.com/yzh8221/DiMTS/blob/master/channel_permutation.ipynb) provides an example code of reproducing Permutation Scanning Algorithm. You can modify the content according to your dataset requirements.

## Citation

If you find this repo useful, please cite our paper!

~~~bash
@article{yao2025dim,
  title={DiM-TS: Bridge the Gap between Selective State Space Models and Time Series for Generative Modeling},
  author={Yao, Zihao and Zuo, Jiankai and Zhang, Yaying},
  journal={arXiv preprint arXiv:2511.18312},
  year={2025}
}
~~~

## Code

Thanks for the open sources papers listed below which DiM-TS is build on.

https://github.com/wmd3i/PaD-TS

https://github.com/EdwardChasel/Spatial-Mamba

https://github.com/Y-debug-sys/Diffusion-TS
