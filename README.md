# Hybrid Lizard Motion Tracking with DeepLabCut and TAPNext
Hello and Welcome, this repo is for Hybrid motion tracking tool by utilizing DeepLabCut (DLC) and TAPNext. 
The repo consists of two main structures which is divided based on the tracking video size and GPU (Cluster) Usage.

## Local Workflow (Labelling, Quick Runs & CPU based)
First, we introduce the local workflow which highlights the usage of hybird motion tracking in a local server. 
1. Manual labelling using DLC interface
   : Manulal labelling job is necessary before traning DLC, utilizing DLC interface and details of the program can be found at https://deeplabcut.github.io/DeepLabCut/README.html
2. DLC Training
   
3. Multiprocessing (Parallel Computing)
   For an efficient job running, we use multiprocessing (specifically queue) by separating jobs into 4. The methods and procedures are documented here: 

## Cluster Workflow (GPU Training & Large Scale Inference)
1. DLC Training
Usually, Running the job for training DLC takes time, so DLC training job is also recommended to run on GPU (we use external data server, ComputeCanada, for compensating such GPU usage).
The main files (scripts, videos and datas) to run DLC training on cluster is listed below
  - Training script: train_dlc.py and submit_dlc_train.sbatch
  - DLC config: config_of_the_video.yaml
  - Labeled dataset folder: which consists of every manual labeled frame and csv file for those coordinates.
  - video file

  Side Note: to reduce trasnferring time for manual labeled frames and config file, we can use command "tar" which compresses the transferring data and unpack it on the cluster.

  After transferring, we can submit the slurm job to complete training job.

2. Multiprocessing (Parallel Computing)
  The procedure and methods are same as local workflow, instead we use 2 GPU's for prediction of DLC (related command: "deeplabcut.analyze_videos(...)) and TAPNext tracking.


## F
  
