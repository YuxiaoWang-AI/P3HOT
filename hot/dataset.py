import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        
        self.padding_constant = opt.padding_constant

        
        self.parse_input_list(odgt, **kwargs)

        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0

    def img_transform(self, img):
        
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        
        
        segm = torch.from_numpy(np.array(segm)).long()
        return segm

    
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        self.num_class = opt.num_class

        
        self.batch_record_list = [[], []]

        
        self.cur_idx = 0
        self.if_shuffled = False
        self.a_transform = A.ReplayCompose([
                
                A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255), p=.75),
                
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10,val_shift_limit=20,p=0.2),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10,b_shift_limit=10,p=0.2),
                    A.RandomGamma(gamma_limit=(70,150),p=0.2),
                    ],p=0.25),
                  
                A.OneOf([
                    A.GaussianBlur(sigma_limit=(0.6, 1.4),p=0.2),
                    A.MotionBlur(blur_limit=5,p=0.2),
                    A.MedianBlur(blur_limit=5,p=0.2),         
                    ], p=0.25),
                
                A.OneOf([
                    A.ISONoise(p=0.3),
                    A.GaussNoise(p=0.3),
                    A.MultiplicativeNoise(p=0.3)
                    ], p=0.25),
                
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5),lightness=(0.9, 1.1)),
                    A.Emboss(),
                    ], p=0.15),
                
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(num_steps = 5, distort_limit = 0.1,p=0.3),
                    A.ElasticTransform(alpha=1, sigma=5,alpha_affine=5,p=0.3)
                    ], p=0.15),
                
                A.OneOf([
                    A.Downscale(scale_min=0.5,scale_max=0.9, p=0.5),
                    A.ImageCompression(quality_lower=30,quality_upper=99, p=0.5)
                    ], p=0.15),
                
                A.Resize(300, 300),
                A.RandomCrop(224, 224),
                
            ])
        self.b_transform = A.Compose([
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
            
        ])

        self.c_transform = A.Compose([
            ToTensorV2(),
        ])

    def _get_sub_batch(self):
        while True:
            
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) 
            else:
                self.batch_record_list[1].append(this_sample) 

            
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j, crop_h, crop_w

    def __getitem__(self, index):
        
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        
        batch_records = self._get_sub_batch() 

        batch_height, batch_width = 224, 224
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros( 
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()
        
        batch_segm_onehot = torch.zeros(
            self.batch_per_gpu,
            self.num_class).long()
        
        batch_depth= torch.zeros( 
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate)

        batch_person_mask = torch.zeros(
            self.batch_per_gpu,
            10, 
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate)

        batch_total_person = torch.zeros(
            self.batch_per_gpu,
            1).long()
        
       


        for i in range(self.batch_per_gpu):
            
            this_record = batch_records[i]
            
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            
            depth_path = segm_path.replace(this_record['fpath_segm'].split("/")[1] + '/annotations', 'HOT-Annotated-DepthMap-ZoeDepth').replace(".png", ".npy")
            
            person_mask = image_path.replace("/images/", "/segments_lang_sam/").replace(".jpg", ".npy")
            person_mask_array = np.load(person_mask)
            
            
            try:
                sum_hw = person_mask_array.sum(axis=(1, 2))
                
                sorted_indices = np.argsort(sum_hw)[::-1]

                
                person_mask_array = person_mask_array[sorted_indices]
            except:
                segm = cv2.imread(segm_path)
                
                person_mask_array = np.zeros((1, segm.shape[0], segm.shape[1]))

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segm = cv2.imread(segm_path)
            
            person_mask_array = person_mask_array.astype(np.uint8)
            assert (segm[:, :, 0] == segm[:, :, 1]).all() and (segm[:, :, 1] == segm[:, :, 2]).all()
            segm = segm[:, :, 0]

            
            depth = np.load(depth_path)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
  

            masks_list = [segm]
            for item in range(person_mask_array.shape[0]):
                masks_list.append(person_mask_array[item])
                
            masks_list.append(depth_norm)
                
            
            img_data = self.a_transform(image=img, masks=masks_list)

            img = img_data["image"]
            masks = img_data["masks"]
            
            segm = masks[0]
            depth = masks[-1]
            person_mask = np.zeros((len(masks) - 2, segm.shape[0], segm.shape[1]))
            
            for item in range(1, len(masks)-1):
                person_mask[item-1] = masks[item]
            
            img = self.b_transform(image=img)["image"]
            
            depth = torch.from_numpy(depth)
            segm = torch.from_numpy(segm)
            person_mask = torch.from_numpy(person_mask)
            
            
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(batch_height // self.segm_downsampling_rate, batch_width // self.segm_downsampling_rate), mode="nearest").squeeze()
            segm = torch.nn.functional.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(batch_height // self.segm_downsampling_rate, batch_width // self.segm_downsampling_rate), mode="nearest").squeeze()
            
            person_mask = torch.nn.functional.interpolate(person_mask.unsqueeze(1), size=(batch_height // self.segm_downsampling_rate, batch_width // self.segm_downsampling_rate), mode="nearest").squeeze(1)

            segm_onehot = np.zeros(self.num_class) 
            for uid in torch.unique(segm):
                segm_onehot[uid] = 1
            segm_onehot = self.segm_transform(segm_onehot) 
            
            
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm
            
            batch_segm_onehot[i][:self.num_class] = segm_onehot
            
            batch_depth[i][:depth.shape[0], :depth.shape[1]] = depth
            if person_mask.shape[0] > 10:
                batch_person_mask[i][:, :person_mask.shape[1], :person_mask.shape[2]] = person_mask[:10]
                
                batch_total_person[i][0] = 10
            else:
                
                batch_person_mask[i][:person_mask.shape[0], :person_mask.shape[1], :person_mask.shape[2]] = person_mask
                batch_total_person[i][0] = person_mask.shape[0]   
        
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        
        output['seg_onehot'] = batch_segm_onehot
        output['depth_label'] = batch_depth
        
        output['person_mask'] = batch_person_mask
        output["total_person"] = batch_total_person
        
        return output

    def __len__(self):
        return int(1e10) 
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.a_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ])

        self.c_transform = A.Compose([
            A.Resize(224, 224),
            
            ToTensorV2(),
        ])
  

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        
        depth_path = segm_path.replace(this_record['fpath_segm'].split("/")[1] + '/annotations', 'HOT-Annotated-DepthMap-ZoeDepth').replace(".png", ".npy")
        
        
        
        person_mask = image_path.replace("/images/", "/segments_lang_sam/").replace(".jpg", ".npy")
        person_mask_array = np.load(person_mask)
        

        try:
            sum_hw = person_mask_array.sum(axis=(1, 2))
            
            sorted_indices = np.argsort(sum_hw)[::-1]

            
            person_mask_array = person_mask_array[sorted_indices]
        except:
            segm = cv2.imread(segm_path)
            
            person_mask_array = np.zeros((1, segm.shape[0], segm.shape[1]))
            
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm = cv2.imread(segm_path)

        person_mask_array = person_mask_array.astype(np.uint8)
        assert (segm[:, :, 0] == segm[:, :, 1]).all() and (segm[:, :, 1] == segm[:, :, 2]).all()
        segm = segm[:, :, 0]
   

        depth = np.load(depth_path)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
            

        masks_list = [segm]
        for item in range(person_mask_array.shape[0]):
            masks_list.append(person_mask_array[item])
            
        masks_list.append(depth_norm)
        
        img_data = self.a_transform(image=img, masks=masks_list)
            

        img = img_data["image"]
        masks = img_data["masks"]
        

        
        segm = masks[0]
        depth = masks[-1]
        person_mask = torch.zeros((len(masks) - 2, segm.shape[0], segm.shape[1]))
        
        for item in range(1, len(masks)-1):
            person_mask[item-1] = masks[item]

        depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(224 // 4, 224 // 4), mode="nearest").squeeze()
        segm = torch.nn.functional.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(224 // 4, 224 // 4), mode="nearest").squeeze()
            
        person_mask = torch.nn.functional.interpolate(person_mask.unsqueeze(1), size=(224 // 4, 224 // 4), mode="nearest").squeeze(1)


        output = dict()
        
        
        img_ori = Image.open(image_path).convert('RGB')
        img_ori = img_ori.resize((224, 224))
        output['img_ori'] = np.array(img_ori)
        output['img_data'] = img
        output['depth_label'] = depth
        
        output['seg_label'] = segm
        output["person_mask"] = person_mask
        output["total_person"] = person_mask.shape[0]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.b_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ])

        self.c_transform = A.Compose([
            A.Resize(224, 224),
            
            ToTensorV2(),
        ])
        

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        
        depth_path = segm_path.replace(this_record['fpath_segm'].split("/")[1] + '/annotations', 'HOT-Annotated-DepthMap-ZoeDepth').replace(".png", ".npy")
        
        
        
        person_mask = image_path.replace("/images/", "/segments_lang_sam/").replace(".jpg", ".npy")
        person_mask_array = np.load(person_mask)

        try:
            sum_hw = person_mask_array.sum(axis=(1, 2))
            
            sorted_indices = np.argsort(sum_hw)[::-1]

            
            person_mask_array = person_mask_array[sorted_indices]
        except:
            segm = cv2.imread(segm_path)
            
            person_mask_array = np.zeros((1, segm.shape[0], segm.shape[1]))
            
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm = cv2.imread(segm_path)

        person_mask_array = person_mask_array.astype(np.uint8)
        assert (segm[:, :, 0] == segm[:, :, 1]).all() and (segm[:, :, 1] == segm[:, :, 2]).all()
        segm = segm[:, :, 0]

        depth = np.load(depth_path)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
            

        masks_list = [segm]
        for item in range(person_mask_array.shape[0]):
            masks_list.append(person_mask_array[item])
            
        masks_list.append(depth_norm)
        
        img_data = self.a_transform(image=img, masks=masks_list)
            

        img = img_data["image"]
        masks = img_data["masks"]
        

        
        segm = masks[0]
        depth = masks[-1]
        person_mask = torch.zeros((len(masks) - 2, segm.shape[0], segm.shape[1]))
        
        for item in range(1, len(masks)-1):
            person_mask[item-1] = masks[item]


        depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(224 // 4, 224 // 4), mode="nearest").squeeze()
        segm = torch.nn.functional.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(224 // 4, 224 // 4), mode="nearest").squeeze()
            
        person_mask = torch.nn.functional.interpolate(person_mask.unsqueeze(1), size=(224 // 4, 224 // 4), mode="nearest").squeeze(1)
         
       
        output = dict()
        
        
        img_ori = Image.open(image_path).convert('RGB')
        img_ori = img_ori.resize((224, 224))
        output['img_ori'] = np.array(img_ori)
        output['img_data'] = img
        output['depth_label'] = depth
        
        output['seg_label'] = segm
        output["person_mask"] = person_mask
        output["total_person"] = person_mask.shape[0]
        output['info'] = this_record['fpath_img']
        return output


    def __len__(self):
        return self.num_sample
