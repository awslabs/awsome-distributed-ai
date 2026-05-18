# How to finetune ESM2 with SageMaker Hyperpod using Amazon G5 instances

## What is SageMaker Hyperpod?
[Amazon SageMaker Hyperpod](https://aws.amazon.com/sagemaker/hyperpod/) offers advanced training tools to help you accelerate scalable, reliable, and secure generative AI application development. It removes the undifferentiated heavy lifting involved in building and optimizing machine learning (ML) infrastructure for training foundation models (FMs) significantly reducing training time. SageMaker Hyperpod ensure customers can continue FM training uninterrupted by periodically saving checkpoints. When a hardware failure occurs during training, SageMaker Hyperpod automatically detects the failure, repairs, or replaces the faulty instance, and resumes the training from the last saved checkpoint, removing the need for customers to manually manage this process and helping them train for week or months in a distributed setting without disruption.


## What is ESM-2?
[ESM-2](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1) is a pLM trained using unsupervised masked language modelling on 250 Million protein sequences by researchers at [Facebook AI Research (FAIR)](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1). It is available in several sizes, ranging from 8 Million to 15 Billion parameters. The smaller models are suitable for various sequence and token classification tasks. The FAIR team also adapted the 3 Billion parameter version into the ESMFold protein structure prediction algorithm. They have since used ESMFold to predict the structure of [more than 700 Million metagenomic proteins](https://esmatlas.com/about).

ESM-2 is a powerful pLM. We will demonstrate how to use QLoRA to fine-tune ESM-2 on g5.24xlarge instances. We will use ESM-2 to predict [subcellular localization](https://academic.oup.com/nar/article/50/W1/W228/6576357?login=false). Understanding where proteins appear in cells can help us understand their role in disease and find new drug targets.

## 0. Prerequisites
You will need to set up a SageMaker Hyperpod cluster using 2 g5.24xlarge instances with a shared parallel filesystem such as [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/getting-started.html).  See the sagemaker-hyperpod section in the [Sagemaker Hyperpod](https://github.com/awslabs/awsome-distributed-training/tree/main/1.architectures/5.sagemaker-hyperpod) folder for setup instructions.  

## 1. Install conda

You can install MiniConda as follows:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ./miniconda3

source ./miniconda3/bin/activate
```
## 2. Create conda environment

You can create conda environment as follows:

```bash
 conda create --name esm python=3.10
 conda activate esm
 conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
 pip3 install -r requirements.txt
```

## 3. Prepare dataset

Next we need to download the Uniref50 training data. You can do so by running:

```bash
python3 0.download_data.py
```
It would download the data and partitions the data in 50 .csv files in `/fsx/ubuntu/csv` folder. The whole process should take less than 30 mins.

```bash
(esm) (CONTROLLER) ubuntu@ip-10-1-71-160:~$ python3 download_data.py
07/03/2024 21:07:01 - INFO - Parsing arguments
07/03/2024 21:07:01 - INFO - Downloading FASTA
07/03/2024 21:07:01 - INFO - Downloading https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz to /fsx/ubuntu/tmp9kq51ybi/fasta
https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz: 100%|████████████████████████████████████████████████████████████████████████████████| 12.8G/12.8G [06:11<00:00, 36.8MB/s]
07/03/2024 21:13:13 - INFO - Generating csv files
Reading FASTA file
498383it [00:12, 59276.95it/s]07/03/2024 21:13:26 - INFO - Writing 500000 records to /fsx/ubuntu/csv/x000.csv
994642it [00:47, 77930.58it/s]07/03/2024 21:14:00 - INFO - Writing 500000 records to /fsx/ubuntu/csv/x001.csv
1495773it [01:08, 88755.06it/s]07/03/2024 21:14:22 - INFO - Writing 500000 records to /fsx/ubuntu/csv/x002.csv
1993826it [01:26, 98115.08it/s]07/03/2024 21:14:40 - INFO - Writing 500000 records to /fsx/ubuntu/csv/x003.csv
...
...
65446537it [11:32, 608611.75it/s]07/03/2024 21:24:46 - INFO - Writing 500000 records to /fsx/ubuntu/csv/x130.csv
65672468it [11:33, 94696.65it/s]
07/03/2024 21:24:47 - INFO - Writing 172468 records to /fsx/ubuntu/csv/x131.csv
07/03/2024 21:24:49 - INFO - Save complete
```

## 4. Convert CSVs to HuggingFace Dataset and Tokenize

Next we need to tokenize the dataset. This will split the data in training, test and validation folders, tokenize them and save the arrow files in `processed` folder.

```bash
(esm) (CONTROLLER) ubuntu@ip-10-1-71-160:~$ python3 1.tokenize_uniref_csv.py
07/03/2024 23:07:49 - INFO - Parsing arguments
07/03/2024 23:07:49 - INFO - Loading csv files from /fsx/ubuntu/csv
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 356272.93it/s]
07/03/2024 23:07:51 - INFO - DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 65672468
    })
})
07/03/2024 23:07:51 - INFO - Splitting dataset
Flattening the indices: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000000/10000000 [06:02<00:00, 27582.63 examples/s]
Flattening the indices: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:00<00:00, 59268.14 examples/s]
Flattening the indices: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:00<00:00, 62442.35 examples/s]
07/03/2024 23:14:01 - INFO - Saving splits to csv
Creating CSV from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [01:51<00:00, 89.70ba/s]
Creating CSV from Arrow format: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 89.99ba/s]
Creating CSV from Arrow format: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 89.29ba/s]
/fsx/ubuntu/miniconda3/envs/esm/lib/python3.10/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 95.0/95.0 [00:00<00:00, 949kB/s]
vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 93.0/93.0 [00:00<00:00, 1.09MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [00:00<00:00, 1.55MB/s]
07/03/2024 23:15:56 - INFO - Processing line by line
Running tokenizer on dataset line_by_line (num_proc=8): 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10000000/10000000 [23:46<00:00, 7008.67 examples/s]
Running tokenizer on dataset line_by_line (num_proc=8): 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:57<00:00, 870.72 examples/s]
Running tokenizer on dataset line_by_line (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:08<00:00, 5695.93 examples/s]
Saving the dataset (62/62 shards): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000000/10000000 [00:55<00:00, 180076.96 examples/s]
Saving the dataset (1/1 shards): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:00<00:00, 177160.38 examples/s]
Saving the dataset (1/1 shards): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [00:00<00:00, 182452.27 examples/s]
```

## 5. Submit training job

Once data is processed, we are ready to train the ESM2 model. To run distributed data parallel (DDP) training, we provide the `train_ddp.sh` script which you can submt as below and training should start:

```
sbatch 2.train_ddp.sh
```

```
1: [INFO|trainer.py:2128] 2024-07-17 00:18:20,620 >> ***** Running training *****
1: [INFO|trainer.py:2129] 2024-07-17 00:18:20,620 >>   Num examples = 100,000
1: [INFO|trainer.py:2130] 2024-07-17 00:18:20,620 >>   Num Epochs = 1
1: [INFO|trainer.py:2131] 2024-07-17 00:18:20,620 >>   Instantaneous batch size per device = 8
1: [INFO|trainer.py:2134] 2024-07-17 00:18:20,620 >>   Total train batch size (w. parallel, distributed & accumulation) = 1,024
1: [INFO|trainer.py:2135] 2024-07-17 00:18:20,620 >>   Gradient Accumulation steps = 16
1: [INFO|trainer.py:2136] 2024-07-17 00:18:20,620 >>   Total optimization steps = 97
1: [INFO|trainer.py:2137] 2024-07-17 00:18:20,622 >>   Number of trainable parameters = 148,796,794
0: [INFO|trainer.py:2128] 2024-07-17 00:18:20,685 >> ***** Running training *****
0: [INFO|trainer.py:2129] 2024-07-17 00:18:20,685 >>   Num examples = 100,000
0: [INFO|trainer.py:2130] 2024-07-17 00:18:20,685 >>   Num Epochs = 1
0: [INFO|trainer.py:2131] 2024-07-17 00:18:20,685 >>   Instantaneous batch size per device = 8
0: [INFO|trainer.py:2134] 2024-07-17 00:18:20,685 >>   Total train batch size (w. parallel, distributed & accumulation) = 1,024
0: [INFO|trainer.py:2135] 2024-07-17 00:18:20,685 >>   Gradient Accumulation steps = 16
0: [INFO|trainer.py:2136] 2024-07-17 00:18:20,685 >>   Total optimization steps = 97
0: [INFO|trainer.py:2137] 2024-07-17 00:18:20,687 >>   Number of trainable parameters = 148,796,794
0: {'loss': 2.9859, 'grad_norm': 0.9704080820083618, 'learning_rate': 4.175257731958763e-05, 'epoch': 0.16}
 19%|█▊        | 18/97 [01:50<08:31,  6.39s/it]
0: {'loss': 2.8209, 'grad_norm': 2.9741921424865723, 'learning_rate': 3.3505154639175256e-05, 'epoch': 0.33}
 36%|███▌      | 35/97 [03:39<06:42,  6.39s/it]
0: {'loss': 2.716, 'grad_norm': 2.2170701026916504, 'learning_rate': 2.5257731958762887e-05, 'epoch': 0.49}
 53%|█████▎    | 50/97 [05:21<05:00,  6.39s/it]
0: {'loss': 2.6697, 'grad_norm': 0.8555800318717957, 'learning_rate': 1.7010309278350517e-05, 'epoch': 0.66}
 68%|██████▊   | 65/97 [06:57<03:24,  6.38s/it]
0: {'loss': 2.6591, 'grad_norm': 0.5596509575843811, 'learning_rate': 8.762886597938144e-06, 'epoch': 0.82}
 82%|████████▏ | 80/97 [08:32<

```

### 5.1 Accelerate training with torch.compile

HuggingFace provides an easy to use [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) class that also provides an option to compile the model graph. For more details on torch.compile, follow this [blog](https://pytorch.org/blog/maximizing-training-throughput/) We notice a speedup of 43% when pre-training ESM2 with torch.compile:

|  Model | device_batch_size | num_nodes | torch.compile |     Instance   |   Throughput   |
|:------:|:-----------------:|:---------:|:-------------:| :------------: | :------------: |
|  ESM2  |         8         |     2     |       No      |  g5.12xlarge   |  160 samples/s | 
|  ESM2  |         8         |     2     |      Yes      |  g5.12xlarge   |  229 samples/s |


### 5.2 Train larger models with Fully Sharded Data Parallel (FSDP)

A disadvantage of DDP training is that it requires the entire model to fit in the GPU memory. The ESM2 150M parameter model easily fits in the A10G GPU of g5.12xlarge instances. However, as models get bigger that may not be possible. For such situations, [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) api is an effective way to shard model parameters which includes optimizer states, gradients and model parameters. However, as model is sharded more, the communication burden between the GPUs also increases. So the best practice is to use FSDP only when DDP is not sufficient. 

We use the [HuggingFace Accelerate](https://github.com/huggingface/accelerate) repo to setup FSDP with ESM2 on a slurm multinode cluster. To this end, we provide the `train_fsd.sh` script that you can submit as below:

```bash
sbatch 3.train_fsdp.sh
```

|  Model | device_batch_size | num_nodes | Strategy |     Instance   |   Throughput   |
|:------:|:-----------------:|:---------:|:--------:| :------------: | :------------: |
|  ESM2  |        14         |     2     |    DDP   |  g5.12xlarge   |  253 samples/s | 
|  ESM2  |        14         |     2     |    FSDP  |  g5.12xlarge   |  162 samples/s |

### 5.3 More results

|  Model | device_batch_size | num_nodes |  Grad Acc  | Strategy |     Instance   |       Throughput       |      GPU Mem       |
|:------:|:-----------------:|:---------:|:-----------|:--------:| :------------: | :--------------------: |:-----------------: |
|  ESM2  |        8          |     2     |     16     |   DDP    |  p5.48xlarge   |    4.36 M samples/s    |     | 





