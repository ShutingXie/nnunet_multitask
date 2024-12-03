import re
import os
from collections import Counter
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

def extract_subtype_from_filename(filename):
    """
    Extract subtypes from the filename.
    """
    match = re.search(r'quiz_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Unable to extract subtype from filename: {filename}")

class nnUNetDatasetSubtype(nnUNetDataset):
    def __getitem__(self, idx):
        if isinstance(idx, int):
            case_id = list(self.dataset.keys())[idx]
        elif isinstance(idx, str):
            case_id = idx
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Expected int or str.")
        
        data_info = super().__getitem__(case_id) 
        
        image_path = data_info['data_file']
        seg_path = data_info.get('seg_from_prev_stage_file', None)
        properties = data_info['properties']

        class_target = extract_subtype_from_filename(image_path)
        
        # The following return info will not be store in json file. They will be directly used during the training process.
        return {
            'data': image_path,
            'seg_target': seg_path,
            'class_target': class_target
        }


if __name__ == '__main__':
    folder = 'nnUnet_preprocessed/Dataset001_PancreasTumor/nnUNetPlans_3d_fullres'
    folder = os.path.abspath(folder)
    print(f"Using folder: {folder}")
    
    case_ids = [f.split(".")[0] for f in os.listdir(folder) if f.endswith(".npz")]
    print(f"Case IDs: {case_ids}")

    dataset = nnUNetDatasetSubtype(folder=folder, case_identifiers=case_ids)

    class_counts = Counter()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        class_target = sample['class_target']  
        class_counts[class_target] += 1       

        print(f"Sample {idx}:")
        print(f"  Data Path: {sample['data']}")
        print(f"  Segmentation Target: {sample['seg_target']}")
        print(f"  Class Target: {sample['class_target']}")

        # 打印类别统计结果
        print("\nClass Distribution:")
        for class_id, count in class_counts.items():
            print(f"  Class {class_id}: {count} samples")