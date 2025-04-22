# DRIFT-DAgger: Dynamic Rank Adjustment in Diffusion Policies for Efficient and Flexible Training

## Setup
- Create conda environment for different scenarios
    ```
    conda env create -f environment.yml
    conda activate drift-dagger
    ```

- Download and train a behavioral cloning policy as the expert using [this repo](https://github.com/Apollo-Lab-Yale/spaces_comparative_study.git)
    - The dataset can be found at [here](https://yaleedu-my.sharepoint.com/my?id=%2Fpersonal%2Fxiatao%5Fsun%5Fyale%5Fedu%2FDocuments%2FProjects%2FArchived%5FResources%2Fspaces%5Fcomparative%5Fstudy%2Ftraining%5Fdatasets&ga=1)
  
- Train a policy with DRIFT-DAgger
  ```
  python training/mavis/mavis_drift_dagger.py 
  ```