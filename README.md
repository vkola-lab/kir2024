# Pilot study of a web-based tool for real-time adequacy assessment of kidney biopsies

This work is published in _Kidney International Reports_ (https://www.kireports.org/article/S2468-0249(24)01795-9/fulltext).

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


## Web development framework
To ensure our segmentation model is widely accessible for onsite kidney biopsy adequacy estimation, we have developed it into a user-friendly web-based application on http://www.biopsyadequacy.org. The tool features an intuitive interface compatible with modern web browsers such as Google Chrome and Firefox. To ensure security and manageability, the application requires user registration. New users are guided through an account creation process with verifiable credentials, while returning users can easily log in with their existing credentials.
Once registered and logged in, users can upload digitized square images of biopsy cores in common image formats such as JPG, PNG, and TIFF. The application provides a feature for delineating the region of interest, allowing users to crop the biopsy area in a square image format and exclude extraneous background elements. This functionality focuses the analysis on the biopsy itself, improving the quality of the output and enhancing user engagement.
Upon confirming and submitting an image for analysis, the platform generates two key outputs: (1) an annotated image highlighting the cortex area within the core biopsy, and (2) a calculated percentage representing the cortex area in relation to the entire core biopsy image. The software is capable of handling various scenarios, including digitized biopsy images that may entirely lack cortex or contain the full kidney cortex. The overal structure of the web-based application is demonstrated below:

<img src = "https://github.com/vkola-lab/kir2024/blob/main/Figure%201.svg">

```

