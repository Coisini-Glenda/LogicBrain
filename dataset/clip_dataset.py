import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import json
import monai.transforms as mtf
import SimpleITK as sitk
from monai.data import set_track_meta
import pandas as pd


class CSVCLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        df = pd.read_csv(args.csv_data_path)
        if mode == "train":
            self.df = df[df['mode'].isin(['train', 'validation'])]
        elif mode == "test":
            self.df = df[df['mode'] == 'test']
        
        self.data_list = self.df.to_dict('records')



         # 打印数据分布信息
        mode_counts = self.df['mode'].value_counts()
        print(f"Total VQA data distribution: {dict(mode_counts)}")
        
        # Define transforms
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)
        self.transform = train_transform
    
    def _construct_image_path(self, patient, study):
        image_filename = f"A_{patient}/Study_{study}.nii.gz"
        return os.path.join(self.data_root, image_filename)
    
    def truncate_text(self, input_text, max_tokens):
        if pd.isna(input_text):
            return "No description available."
            
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')
        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        
        for attempt in range(max_attempts):
            try:
                data = self.data_list[idx]
                patient_id = data["Patient"]
                study_id = data["Study"]
                processed_text = data["Processed_Text"]
                
                # Construct and load image
                image_path = self._construct_image_path(patient_id, study_id)
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image).to(torch.float16)
                # image.unsqueeze(0)  #hlip的时候需要加入
                image = image.unsqueeze(0) 
              
                # Process text
                text = self.truncate_text(processed_text, self.args.max_length)
                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': text_tensor["input_ids"][0],
                    'attention_mask': text_tensor["attention_mask"][0],
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}, attempt {attempt+1}: {e}")
                print(f"!!!!!! Failed at index: {idx}, path: {image_path}")
                idx = random.randint(0, len(self.data_list) - 1)
                continue
        
        # 如果所有尝试都失败，返回一个默认的样本
        print(f"Failed to load any sample after {max_attempts} attempts, returning default sample")
        
        # 创建默认返回值，避免返回None
        default_image = torch.zeros((1, 64, 64, 64), dtype=torch.float)  # 根据你的图像尺寸调整
        default_text = "No description available."
        default_text_tensor = self.tokenizer(
            default_text, max_length=self.args.max_length, truncation=True, 
            padding="max_length", return_tensors="pt"
        )
        
        return {
            'image': default_image,
            'text': default_text,
            'input_id': default_text_tensor["input_ids"][0],
            'attention_mask': default_text_tensor["attention_mask"][0],
            'question_type': "Image_text_retrieval",
        }
    
    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.df),  # 正确: 使用 self.df
            "mode": self.mode,
            "patient_count": self.df['Patient'].nunique(), # 正确
            "study_count": self.df['Study'].nunique(), # 正确
        }
        return info


class CLIPDataset2(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                # image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
    
    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
        }
        return info

