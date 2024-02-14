#%% Run model on test case images and generates annotated images and the cortex percentage values
# load the original SAM model
from skimage import io, transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import sys
from rembg import remove
import pandas as pd

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()

npz_tr_path = 'data/demo2D_vit_b'
demo_dataset = NpzDataset(npz_tr_path)
demo_dataloader = DataLoader(demo_dataset, batch_size=1, shuffle=True)
for img_embed, gt2D, bboxes in demo_dataloader:
    print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}")
    break

npz_tr_path = 'data/demo2D_vit_b'
work_dir = './work_dir'
task_name = 'demo2D'

# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir/demo2D/sam_model_best.pth'
device = 'cuda:0'
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)    
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.train()
# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

ori_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
ori_sam_predictor = SamPredictor(ori_sam_model)



#create csv file
if not os.path.exists('data.csv'):
    with open('data.csv', 'w') as file:
        pass

# Initialize headers in a csv file if it is empty
headers = ["ID", "Predicted Percentage", "Actual Percentage", "Error"]
empty_df = pd.DataFrame(columns=headers)
csv_file_path = "data.csv"
empty_df.to_csv(csv_file_path, index=False)



data = {}
data['ID'] = []
data['Predicted Percentage'] = []
data['Actual Percentage'] = []
data['Error'] = []

excel = pd.read_excel('cortex_perc.xlsx')
id_list = excel.iloc[:, 0].tolist()
labeled_perc_list = excel.iloc[:,1].tolist()


labeled_data = {}
for i in range(len(id_list)):
    labeled_data[id_list[i]] = labeled_perc_list[id_list.index(id_list[i])] 

ts_img_path = 'data/MedSAMDemo_2D/test/images/'
test_names = sorted(os.listdir(ts_img_path))
ctr = 1
# for img_idx in range(len(test_names)):
for img_idx in range(len(test_names)):
    image_data = io.imread(join(ts_img_path, test_names[img_idx]))
    print(test_names[img_idx])


    image_data = remove(image_data)
    # Resize the image_data to 256x256 pixels
    new_size = (256, 256)
    resized_image_data = transform.resize(image_data, new_size, anti_aliasing=True)

    # If the image_data is of type float, convert it back to uint8 in the range [0, 255]
    if resized_image_data.dtype == float:
        resized_image_data = (resized_image_data * 255).astype('uint8')

    image_data = resized_image_data


    if image_data.shape[-1]>3 and len(image_data.shape)==3:
        image_data = image_data[:,:,:3]
    if len(image_data.shape)==2:
        image_data = np.repeat(image_data[:,:,None], 3, axis=-1)

    # read ground truth (gt should have the same name as the image) and simulate a bounding box
    def get_bbox_from_mask(mask):
    
        '''Returns a bounding box from a mask'''
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return np.array([x_min, y_min, x_max, y_max])

  


    # preprocess: cut-off and max-min normalization
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre[image_data==0] = 0
    image_data_pre = np.uint8(image_data_pre)
    H, W, _ = image_data_pre.shape

    # predict the segmentation mask using the original SAM model
    ori_sam_predictor.set_image(image_data_pre)
    ori_sam_seg, _, _ = ori_sam_predictor.predict(point_coords=None, box=None, multimask_output=False)
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_img = sam_transform.apply_image(image_data_pre)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

    def calculate_bounding_box(image):
        # Convert image to grayscale
        gray_image = np.sum(image, axis=-1) // 3

        # Find non-white (non-light) pixel coordinates
        rows, cols = np.where(gray_image < 255)  # Assuming white is 255,255,255 in grayscale
        if rows.size > 0 and cols.size > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
        else:
            # Default to full image if no non-white pixels found
            y_min, x_min = 0, 0
            y_max, x_max = gray_image.shape

        return np.array([x_min, y_min, x_max, y_max])

    threshold = 50  # Adjust this value based on your preference

    # Remove black or very dark gray pixels
    image_without_dark = image_data.copy()
    gray_intensity = np.sum(image_data, axis=-1) // 3  # Convert RGB to grayscale
    image_without_dark[gray_intensity <= threshold] = [255, 255, 255]

    with torch.no_grad():
        # pre-compute the image embedding
        ts_img_embedding = sam_model.image_encoder(input_image)
        box_np = calculate_bounding_box(image_without_dark)

        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)

        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4) #My code for not showing the bbox
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        print(medsam_seg.shape)

    grayscale_image = np.sum(image_without_dark, axis=-1) // 3

    non_white_pixels = np.sum(grayscale_image < 255)

    overlay = image_data.copy()
    overlay[(medsam_seg > 0) & (grayscale_image < 255)] = [0, 255, 0]

    cv2.rectangle(overlay, (box_np[0], box_np[1]), (box_np[2], box_np[3]), (255, 0, 0), 2)  # Color: Blue, Thickness: 2

    # Save the overlaid image with the bounding box as a JPEG file
    cv2.imwrite("image_with_mask.png", overlay)
    image = cv2.imread("image_with_mask.png")

    # Save the overlaid image as a JPEG file
    output_folder = 'Annotated_images/'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_file_name = test_names[img_idx]
    cv2.imwrite(output_folder + output_file_name, image)

    segmented = np.sum((medsam_seg > 0) & (grayscale_image < 255))

    cortex_percentage = (segmented / non_white_pixels) * 100
    formatted_percentage = "{:.2f}\n".format(cortex_percentage)    
    id = test_names[img_idx][:-4]
    error = (cortex_percentage - labeled_data[int(id)])/100
    error = abs(error)
    error = round(error, 4)

    data['ID'].append(str(id))
    data['Predicted Percentage'].append(str(cortex_percentage))
    data['Actual Percentage'].append(str(labeled_data[int(id)]))
    data['Error'].append(str(error))
    

csv = pd.read_csv('data.csv')
df = pd.DataFrame(data)
updated_data = pd.concat([csv, df], ignore_index=True)
updated_data.to_csv('data.csv', index=False)