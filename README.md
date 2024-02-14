# ki2024
This the official repository for Biopsy adequacy assessment tool using MedSAM model.


## Installation 
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`




## Fine-tune SAM on customized dataset

We provide a step-by-step tutorial with a small dataset to help you quickly start the training process.

### Data preparation and preprocessing

Run pre-processing


```bash
python pre_CT.py -i path_to_image_folder -gt path_to_gt_folder -o path_to_output
```

- split dataset: 80% for training and 20% for testing
- image normalization
- pre-compute image embedding
- save the normalized images, ground truth masks, and image embedding as a `npz` file


### Model Training

Please check the step-by-step tutorial: `finetune_and_inference_tutorial.py`.

Train the model

```bash
python train -i ./data/Tr_emb --task_name SAM-ViT-B --num_epochs 1000 --batch_size 8 --lr 1e-5
```



## Inference

Download the MedSAM checkpoint ([GoogleDrive](https://drive.google.com/drive/folders/1bWv_Zs5oYLpGMAvbotnlNXJPq7ltRUvF?usp=share_link)) and put it to `work_dir/MedSAM`. 

Run

```bash
python MedSAM_Inference.py -i ./data/Test -o ./ -chk work_dir/MedSAM/medsam_20230423_vit_b_0.0.1.pth
```

## Run the model
After preprocessing and creating checkpoints, run the file run.py. The annotation images will be created on "/Annotated_images" folder and a csv file
including columns: image ID, actual cortex percentage, predicted cortex percentage, and error of prediction is created in data.csv in the current folder.
