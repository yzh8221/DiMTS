# DiM-TS: Bridge the Gap between Selective State Space Models and Time Series for Generative Modeling

## Running the Code

The following instructions explain how to run the code. 

### Environment & Libraries

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

### Training

For training, you can reproduce the experimental results by runing

~~~bash
python main.py --name {name} --config_file {config.yaml} --gpu 0 --train
~~~

#### Unconstrained Sampling

Please use the saved model for sampling by running

```bash
python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 0 --milestone {checkpoint_number}
```
