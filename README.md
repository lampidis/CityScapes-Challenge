# CityScapes-Challenge

This is a project for the CityScapes-Challenge

Name: Loukas Lampidis

Student number: 2212277

CodaLab name: lampidis

## Benchmarks

The competition comprises four benchmarks, each targeting a specific aspect of model performance. This repository focuses on the Out-of-distibution detection.

4. **Out-of-distribution detection**  
   Models often encounter data that differs from the training distribution, leading to unreliable predictions. This benchmark evaluates your model's ability to detect and handle such out-of-distribution samples.

## Installation

### Clone the repository:

```
git clone https://github.com/lampidis/CityScapes-Challenge.git`
cd CityScapes-Challenge
```

### Install dependencies:

```
pip install -r requirements.txt
```

### Dataset Preparation

Download the Cityscapes and the Wilddash2 datasets
Extract the contents into `data` dir

<pre>
CityScapes-Challenge
      └── data
            ├── cityscapes
            └── wilddash2
</pre>

move the script inside the wilddash dir and run to transform the labels of Wilddash2 from panoptic to semantic to match the CityScapes dataset.

```
python panoptic2segm.py
```

## Running the Code

The code can be run in slurm format by building the docker and sending a slurm job

```
sbatch download_docker_and_data.sh
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

or by calling the train_model.py

```
python train_model.py
```
