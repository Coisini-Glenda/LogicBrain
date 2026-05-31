import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import monai.transforms as mtf
from monai.data import set_track_meta


sys.path.append(os.getcwd())

from src.model.llm.hlip_qwen import VLMQwenForCausalLM

class FeatureExtractionDataset(Dataset):
    def __init__(self, args):
        self.args = args
        

        csv_path = os.path.join(args.data_root, args.csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data_df = pd.read_csv(csv_path)
        

        if 'Patient' not in self.data_df.columns or 'Study' not in self.data_df.columns:
            raise ValueError("CSV must contain 'Patient' and 'Study' columns.")
        

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if 'mode' in self.data_df.columns and args.mode != 'all':
            self.data_df = self.data_df[self.data_df['mode'] == args.mode].reset_index(drop=True)
            if local_rank == 0:
                print(f"Filtered to {len(self.data_df)} samples for mode '{args.mode}'")
        else:
            if local_rank == 0:
                print(f"Loaded {len(self.data_df)} samples.")


        self.transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float), 
        ])
        set_track_meta(False)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        try:
            row = self.data_df.iloc[idx]
            patient_id = str(row['Patient']).strip()
            study_id = str(row['Study']).strip()
            

            image_filename = f"A_{patient_id}/Study_{study_id}.nii.gz"
            full_image_path = os.path.join(self.args.image_folder, image_filename)

            if not os.path.exists(full_image_path):
                return None


            sitk_image = sitk.ReadImage(full_image_path)
            image = sitk.GetArrayFromImage(sitk_image) # (D, H, W)
            image = np.expand_dims(image, axis=0)      # (1, D, H, W)
            

            image = self.transform(image) 
            
            image = image.unsqueeze(0)                 # HLIP: (1, 1, D, H, W)

            return {
                "image": image,
                "patient_id": patient_id,
                "study_id": study_id
            }

        except Exception as e:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    images = torch.stack([b['image'] for b in batch])
    metadata = [{'patient_id': b['patient_id'], 'study_id': b['study_id']} for b in batch]
    return images, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--csv_filename", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1) 
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus == 0:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Warning: No GPU detected, using CPU.")
    else:
        device_id = local_rank % num_visible_gpus
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        
    if local_rank == 0:
        print(f"Distributed Init: World Size={world_size}, Visible GPUs per process={num_visible_gpus}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    if local_rank == 0:
        print(f"Loading model from {args.model_name_or_path}...")
        
    model = VLMQwenForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": device}, 
        trust_remote_code=True
    )
    model.eval()

    dataset = FeatureExtractionDataset(args)
    total_len = len(dataset)
    indices = list(range(rank, total_len, world_size))
    subset_dataset = Subset(dataset, indices)
    
    if local_rank == 0:
        print(f"Total samples: {total_len}. Each GPU processing approx: {len(subset_dataset)}")

    dataloader = DataLoader(
        subset_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    desc = f"Rank {rank}"
    with torch.no_grad():
        for images, metadata_list in tqdm(dataloader, desc=desc, disable=(world_size > 1 and local_rank != 0)):
            if images is None: continue
            
            images = images.to(device, dtype=model.dtype)
            
            try:
                features_seq = model.encode_images(images) 
                
                features_np = features_seq.float().cpu().numpy()
                
                for i, meta in enumerate(metadata_list):
                    filename = f"{meta['patient_id']}_{meta['study_id']}.npy"
                    save_path = os.path.join(args.output_dir, filename)
                    

                    np.save(save_path, features_np[i])
                    
            except Exception as e:
                print(f"[Rank {rank}] Error extracting {metadata_list}: {e}")
                continue

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank == 0:
        print("Done.")

if __name__ == "__main__":
    main()