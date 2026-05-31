import os
import json
import torch
import random
import numpy as np
import monai.transforms as mtf
import SimpleITK as sitk  # 移除 SimpleITK，改用 numpy
import pandas as pd
from monai.data import set_track_meta
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import BrainCT_Report_Prompts, BrainCT_TherapyPrognosis_Prompts


import os
import pandas as pd
import random
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
import monai.transforms as mtf
from monai.data import set_track_meta


class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=None):  # 修改默认值为 None
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # 读取CSV文件
        csv_path = os.path.join(args.data_root, args.cap_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'Processed_Text', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
            # 删除限制数据条数的代码，测试全部数据
            # if test_size and len(self.data_df_filtered) > test_size:
            #     self.data_df_filtered = self.data_df_filtered.head(test_size)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        self.caption_prompts = BrainCT_Report_Prompts
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total data distribution: {dict(mode_counts)}")
        
        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)
        
        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                caption = str(data_row['Processed_Text']).strip()
                
                # 构建图像路径：brain_origin_data_processed/A_{patient_id}/Study_{study_id}.nii.gz
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")
                
                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                
                image = np.expand_dims(image, axis=0)

                # 应用变换
                image = self.transform(image).to(torch.float16)
                # image = image.unsqueeze(0)   #HLIP需要加入
                
                # 处理文本
                answer = caption
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question
                
                # tokenize文本
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                valid_len = torch.sum(attention_mask)
                
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                # 处理问题部分
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                question_len = torch.sum(question_tensor["attention_mask"][0])
                
                # 创建标签
                label = input_id.clone()
                label[:question_len] = -100
                
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }
                
                return ret
                
            except Exception as e:
                print(f"Error in __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load data after {max_attempts} attempts. Original index: {original_idx}")
    
    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
        }
        return info

class CapDataset_HLIP(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=None):  # 修改默认值为 None
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # 读取CSV文件
        csv_path = os.path.join(args.data_root, args.cap_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'Processed_Text', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
            # 删除限制数据条数的代码，测试全部数据
            # if test_size and len(self.data_df_filtered) > test_size:
            #     self.data_df_filtered = self.data_df_filtered.head(test_size)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        self.caption_prompts = BrainCT_Report_Prompts
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total data distribution: {dict(mode_counts)}")
        
        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)
        
        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                caption = str(data_row['Processed_Text']).strip()
                
                # 构建图像路径：brain_origin_data_processed/A_{patient_id}/Study_{study_id}.nii.gz
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")
                
                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                
                image = np.expand_dims(image, axis=0)

                # 应用变换
                image = self.transform(image).to(torch.float16)

                image = image.unsqueeze(0)  
                # 处理文本
                answer = caption
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question
                
                # tokenize文本
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                valid_len = torch.sum(attention_mask)
                
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                # 处理问题部分
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                question_len = torch.sum(question_tensor["attention_mask"][0])
                
                # 创建标签
                label = input_id.clone()
                label[:question_len] = -100
                
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }
                
                return ret
                
            except Exception as e:
                print(f"Error in __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load data after {max_attempts} attempts. Original index: {original_idx}")
    
    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
        }
        return info

class VQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        # 读取CSV文件 - 使用统一的VQA CSV文件
        csv_path = os.path.join(args.data_root, args.vqa_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded VQA CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'Question', 'Answer', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} VQA samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total VQA data distribution: {dict(mode_counts)}")

        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])

        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                question = str(data_row['Question']).strip()
                answer = str(data_row['Answer']).strip()
                
                # 构建图像路径：和CapDataset保持一致
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")

                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                image = np.expand_dims(image, axis=0)

                image = self.transform(image).to(torch.float16)

                # image = image.unsqueeze(0)   #HLIP需要加入

                # 处理问题和答案 - 只处理开放式问题
                question_with_tokens = self.image_tokens + " " + question
                
                # tokenize文本
                text_tensor = self.tokenizer(
                    question_with_tokens + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question_with_tokens,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question_with_tokens,
                    "answer": answer,
                    "question_type": "VQA",
                }

                return ret

            except Exception as e:
                print(f"Error in VQA __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load VQA data after {max_attempts} attempts. Original index: {original_idx}")

    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
        }
        return info

class VQADataset_HLIP(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        # 读取CSV文件 - 使用统一的VQA CSV文件
        csv_path = os.path.join(args.data_root, args.vqa_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded HLIP VQA CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'Question', 'Answer', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} HLIP VQA samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total HLIP VQA data distribution: {dict(mode_counts)}")

        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])

        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                question = str(data_row['Question']).strip()
                answer = str(data_row['Answer']).strip()
                
                # 构建图像路径
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")

                # ==================== HLIP 图像处理逻辑 ====================
                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                
                # 1. 准备4D数据给MONAI: (D, H, W) -> (1, D, H, W)
                image = np.expand_dims(image, axis=0)
                
                # 2. 应用变换 (输入和输出都是4D Tensor)
                image = self.transform(image).to(torch.float16)
                
                # 3. 在变换后增加第5个维度以满足HLIP模型要求: (1, D, H, W) -> (1, 1, D, H, W)
                image = image.unsqueeze(0)
                # =========================================================

                # 处理问题和答案
                question_with_tokens = self.image_tokens + " " + question
                
                # tokenize文本
                text_tensor = self.tokenizer(
                    question_with_tokens + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question_with_tokens,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question_with_tokens,
                    "answer": answer,
                    "question_type": "VQA",
                }

                return ret

            except Exception as e:
                print(f"Error in VQA_HLIP __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        raise RuntimeError(f"Failed to load VQA_HLIP data after {max_attempts} attempts. Original index: {original_idx}")


class ClosedVQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        # 读取CSV文件 - 使用封闭式VQA CSV文件
        csv_path = os.path.join(args.data_root, args.closed_vqa_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded Closed VQA CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'Question', 'Choices', 'Answer', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} Closed VQA samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total Closed VQA data distribution: {dict(mode_counts)}")

        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])

        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def format_question_with_choices(self, question, choices):
        """
        将问题和选项格式化为完整的输入文本
        """
        formatted_choices = "Choices: " + choices
        return question + " " + formatted_choices

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                question = str(data_row['Question']).strip()
                choices = str(data_row['Choices']).strip()
                answer = str(data_row['Answer']).strip()
                
                # 构建图像路径：和VQADataset保持一致
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")

                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image).to(torch.float16)

                # image = image.unsqueeze(0)

                # 处理问题和选项 - 将问题和选项组合
                formatted_question = self.format_question_with_choices(question, choices)
                question_with_tokens = self.image_tokens + " " + formatted_question
                
                # tokenize文本 - 包含问题、选项和答案
                text_tensor = self.tokenizer(
                    question_with_tokens + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # tokenize问题部分（包含选项）
                question_tensor = self.tokenizer(
                    question_with_tokens,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                # 创建标签 - 只有答案部分参与损失计算
                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question_with_tokens,
                    "answer": answer,
                    "question_type": "Closed_VQA",
                }

                return ret

            except Exception as e:
                print(f"Error in Closed VQA __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load Closed VQA data after {max_attempts} attempts. Original index: {original_idx}")

    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
            "question_count": len(self.data_df_filtered),
        }
        return info

    def get_sample_data(self, idx=0):
        """获取样本数据用于调试"""
        if idx >= len(self.data_df_filtered):
            idx = 0
        
        data_row = self.data_df_filtered.iloc[idx]
        return {
            "patient_id": data_row['Patient'],
            "study_id": data_row['Study'],
            "question": data_row['Question'],
            "choices": data_row['Choices'],
            "answer": data_row['Answer'],
            "mode": data_row['mode']
        }
    

class ClosedVQADataset_HLIP(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        # 读取CSV文件 - 使用封闭式VQA CSV文件
        csv_path = os.path.join(args.data_root, args.closed_vqa_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded Closed VQA CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'Question', 'Choices', 'Answer', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} Closed VQA samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total Closed VQA data distribution: {dict(mode_counts)}")

        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])

        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def format_question_with_choices(self, question, choices):
        """
        将问题和选项格式化为完整的输入文本
        """
        formatted_choices = "Choices: " + choices
        return question + " " + formatted_choices

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                question = str(data_row['Question']).strip()
                choices = str(data_row['Choices']).strip()
                answer = str(data_row['Answer']).strip()
                
                # 构建图像路径：和VQADataset保持一致
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")

                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image).to(torch.float16)

                image = image.unsqueeze(0)

                # 处理问题和选项 - 将问题和选项组合
                formatted_question = self.format_question_with_choices(question, choices)
                question_with_tokens = self.image_tokens + " " + formatted_question
                
                # tokenize文本 - 包含问题、选项和答案
                text_tensor = self.tokenizer(
                    question_with_tokens + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # tokenize问题部分（包含选项）
                question_tensor = self.tokenizer(
                    question_with_tokens,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                # 创建标签 - 只有答案部分参与损失计算
                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question_with_tokens,
                    "answer": answer,
                    "question_type": "Closed_VQA",
                    "patient_id": patient_id,
                    "study_id":study_id
                }

                return ret

            except Exception as e:
                print(f"Error in Closed VQA __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load Closed VQA data after {max_attempts} attempts. Original index: {original_idx}")

    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
            "question_count": len(self.data_df_filtered),
        }
        return info

    def get_sample_data(self, idx=0):
        """获取样本数据用于调试"""
        if idx >= len(self.data_df_filtered):
            idx = 0
        
        data_row = self.data_df_filtered.iloc[idx]
        return {
            "patient_id": data_row['Patient'],
            "study_id": data_row['Study'],
            "question": data_row['Question'],
            "choices": data_row['Choices'],
            "answer": data_row['Answer'],
            "mode": data_row['mode']
        }



class TherapyPrognosisDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # 读取CSV文件
        csv_path = os.path.join(args.data_root, args.therapy_prognosis_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded Therapy Prognosis CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'treatment_prognosis_opinion', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        # 使用治疗预后的prompts
        self.caption_prompts = BrainCT_TherapyPrognosis_Prompts
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} Therapy Prognosis samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total Therapy Prognosis data distribution: {dict(mode_counts)}")
        
        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)
        
        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                therapy_prognosis = str(data_row['treatment_prognosis_opinion']).strip()
                
                # 构建图像路径：和其他Dataset保持一致的路径格式
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")
                
                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                
                image = np.expand_dims(image, axis=0)

                # 应用变换
                image = self.transform(image).to(torch.float16)

                # image = image.unsqueeze(0)  
                
                # 处理文本 - 使用治疗预后意见作为答案
                answer = therapy_prognosis
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question
                
                # tokenize文本
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                valid_len = torch.sum(attention_mask)
                
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                # 处理问题部分
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                question_len = torch.sum(question_tensor["attention_mask"][0])
                
                # 创建标签
                label = input_id.clone()
                label[:question_len] = -100
                
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "TherapyPrognosis",
                }
                
                return ret
                
            except Exception as e:
                print(f"Error in Therapy Prognosis __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load Therapy Prognosis data after {max_attempts} attempts. Original index: {original_idx}")
    
    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
        }
        return info
    
    def get_sample_data(self, idx=0):
        """获取样本数据用于调试"""
        if idx >= len(self.data_df_filtered):
            idx = 0
        
        data_row = self.data_df_filtered.iloc[idx]
        return {
            "patient_id": data_row['Patient'],
            "study_id": data_row['Study'],
            "therapy_prognosis": data_row['treatment_prognosis_opinion'],
            "mode": data_row['mode']
        }



class TherapyPrognosisDataset_HLIP(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # 读取CSV文件
        csv_path = os.path.join(args.data_root, args.therapy_prognosis_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded Therapy Prognosis CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        # 检查必要的列是否存在
        required_columns = ['Patient', 'Study', 'treatment_prognosis_opinion', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 根据mode筛选数据
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        # 重置索引
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        # 使用治疗预后的prompts
        self.caption_prompts = BrainCT_TherapyPrognosis_Prompts
        
        # 检查数据是否为空
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} Therapy Prognosis samples for {mode} mode")
        
        # 打印数据分布信息
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total Therapy Prognosis data distribution: {dict(mode_counts)}")
        
        # 数据增强变换
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)
        
        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # 获取当前数据项
                data_row = self.data_df_filtered.iloc[idx]
                patient_id = data_row['Patient']
                study_id = data_row['Study']
                therapy_prognosis = str(data_row['treatment_prognosis_opinion']).strip()
                
                # 构建图像路径：和其他Dataset保持一致的路径格式
                image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                # 检查文件是否存在
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")
                
                # 使用 SimpleITK 读取 .nii.gz 文件
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                
                image = np.expand_dims(image, axis=0)

                # 应用变换
                image = self.transform(image).to(torch.float16)

                image = image.unsqueeze(0)  
                
                # 处理文本 - 使用治疗预后意见作为答案
                answer = therapy_prognosis
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question
                
                # tokenize文本
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                valid_len = torch.sum(attention_mask)
                
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                # 处理问题部分
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                question_len = torch.sum(question_tensor["attention_mask"][0])
                
                # 创建标签
                label = input_id.clone()
                label[:question_len] = -100
                
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "TherapyPrognosis",
                }
                
                return ret
                
            except Exception as e:
                print(f"Error in Therapy Prognosis __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Patient={data_row.get('Patient', 'N/A')}, Study={data_row.get('Study', 'N/A')}")
                
                # 随机选择另一个索引
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError(f"Failed to load Therapy Prognosis data after {max_attempts} attempts. Original index: {original_idx}")
    
    def get_data_info(self):
        """返回数据集的基本信息"""
        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "patient_count": self.data_df_filtered['Patient'].nunique(),
            "study_count": self.data_df_filtered['Study'].nunique(),
        }
        return info
    
    def get_sample_data(self, idx=0):
        """获取样本数据用于调试"""
        if idx >= len(self.data_df_filtered):
            idx = 0
        
        data_row = self.data_df_filtered.iloc[idx]
        return {
            "patient_id": data_row['Patient'],
            "study_id": data_row['Study'],
            "therapy_prognosis": data_row['treatment_prognosis_opinion'],
            "mode": data_row['mode']
        }

class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, mode=mode),
            ClosedVQADataset(args, tokenizer, mode=mode),
            TherapyPrognosisDataset(args, tokenizer, mode=mode)
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]




class TextDatasets_HLIP(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets_HLIP, self).__init__()
        self.ds_list = [
            CapDataset_HLIP(args, tokenizer, mode=mode),
            VQADataset_HLIP(args, tokenizer, mode=mode),
            ClosedVQADataset_HLIP(args, tokenizer, mode=mode),
            TherapyPrognosisDataset_HLIP(args, tokenizer, mode=mode)
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
        
        


class VQADataset_HLIP_CQ500(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        csv_path = os.path.join(args.data_root, args.vqa_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded HLIP VQA CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        required_columns = ['Path', 'Question', 'Answer', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")
        
        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} HLIP VQA samples for {mode} mode")
        
        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total HLIP VQA data distribution: {dict(mode_counts)}")

        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])

        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:

                data_row = self.data_df_filtered.iloc[idx]
                path = data_row['Path']  
                question = str(data_row['Question']).strip()
                answer = str(data_row['Answer']).strip()
                

                image_filename = f"{path}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")

                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                

                image = np.expand_dims(image, axis=0)
                

                image = self.transform(image).to(torch.float16)
                
                image = image.unsqueeze(0)

                question_with_tokens = self.image_tokens + " " + question
                

                text_tensor = self.tokenizer(
                    question_with_tokens + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question_with_tokens,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question_with_tokens,
                    "answer": answer,
                    "question_type": "VQA",
                }

                return ret

            except Exception as e:
                print(f"Error in VQA_HLIP __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Path={data_row.get('Path', 'N/A')}")
                
                idx = random.randint(0, len(self.data_df_filtered) - 1)
        
        raise RuntimeError(f"Failed to load VQA_HLIP data after {max_attempts} attempts. Original index: {original_idx}")
        
        

class TherapyPrognosisDataset_HLIP_CQ500(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        

        csv_path = os.path.join(args.data_root, args.therapy_prognosis_csv_filename)
        try:
            self.data_df = pd.read_csv(csv_path)
            print(f"Successfully loaded Therapy Prognosis CSV file: {csv_path}")
            print(f"CSV columns: {list(self.data_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Could not read CSV file {csv_path}: {e}")
        
        required_columns = ['Path', 'mode']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        if mode == "train":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'train'].copy()
        elif mode == "validation":
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'validation'].copy()
        elif "test" in mode:
            self.data_df_filtered = self.data_df[self.data_df['mode'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, validation, test")

        self.data_df_filtered = self.data_df_filtered.reset_index(drop=True)
        

        self.caption_prompts = BrainCT_TherapyPrognosis_Prompts
        
        if len(self.data_df_filtered) == 0:
            raise ValueError(f"No data found for mode '{mode}'. Please check your CSV file.")
        
        print(f"Loaded {len(self.data_df_filtered)} Therapy Prognosis samples for {mode} mode")
        

        mode_counts = self.data_df['mode'].value_counts()
        print(f"Total Therapy Prognosis data distribution: {dict(mode_counts)}")
        
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        val_transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)
        
        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df_filtered)

    def __getitem__(self, idx):
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
     
                data_row = self.data_df_filtered.iloc[idx]
                path = data_row['Path']  
                
         
                image_filename = f"{path}.nii.gz"
                full_image_path = os.path.join(self.args.image_folder, image_filename)
                
            
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file not found: {full_image_path}")
                
               
                sitk_image = sitk.ReadImage(full_image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                
                image = np.expand_dims(image, axis=0)
                image = self.transform(image).to(torch.float16)

                image = image.unsqueeze(0)  
                

                answer = ""  
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question
                
        
                text_tensor = self.tokenizer(
                    question, 
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                valid_len = torch.sum(attention_mask)
                
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                question_len = valid_len  
                
    
                label = input_id.clone()
                label[:] = -100  
                
                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,  
                    "question_type": "TherapyPrognosis",
                    "path": path, 
                }
                
                return ret
                
            except Exception as e:
                print(f"Error in Therapy Prognosis __getitem__ at index {idx} (attempt {attempt + 1}): {e}")
                print(f"Failed data: Path={data_row.get('Path', 'N/A')}")
                

                idx = random.randint(0, len(self.data_df_filtered) - 1)
        

        raise RuntimeError(f"Failed to load Therapy Prognosis data after {max_attempts} attempts. Original index: {original_idx}")
    
    def get_data_info(self):

        info = {
            "total_samples": len(self.data_df_filtered),
            "mode": self.mode,
            "path_count": self.data_df_filtered['Path'].nunique(),
        }
        return info
    
    def get_sample_data(self, idx=0):

        if idx >= len(self.data_df_filtered):
            idx = 0
        
        data_row = self.data_df_filtered.iloc[idx]
        return {
            "path": data_row['Path'],
            "mode": data_row['mode']
        }  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        