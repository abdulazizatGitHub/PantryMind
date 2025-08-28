#!/usr/bin/env python3
"""
GroZi-120 Dataset Preparation Script
Converts GroZi-120 dataset from original format to YOLO format for training.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import random

class GroZiDatasetConverter:
    def __init__(self, dataset_path: str, output_path: str, train_split: float = 0.8, val_split: float = 0.1):
        """
        Initialize the dataset converter.
        
        Args:
            dataset_path: Path to the original GroZi-120 dataset
            output_path: Path where YOLO format dataset will be saved
            train_split: Fraction of data for training (default: 0.8)
            val_split: Fraction of data for validation (default: 0.1)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_split = train_split
        self.val_split = val_split
        
        # Create output directories
        self.create_output_dirs()
        
    def create_output_dirs(self):
        """Create the necessary output directories."""
        dirs = [
            self.output_path / "images" / "train",
            self.output_path / "images" / "val", 
            self.output_path / "images" / "test",
            self.output_path / "labels" / "train",
            self.output_path / "labels" / "val",
            self.output_path / "labels" / "test"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def parse_coordinates(self, coord_file: Path) -> List[Tuple[int, int, int, int]]:
        """
        Parse coordinates.txt file and convert to bounding boxes.
        
        Args:
            coord_file: Path to coordinates.txt file
            
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        boxes = []
        
        try:
            with open(coord_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Format: frame_id x y width height confidence
                        parts = line.split('\t')
                        if len(parts) >= 5:
                            frame_id = int(parts[0])
                            x = int(parts[1])
                            y = int(parts[2])
                            width = int(parts[3])
                            height = int(parts[4])
                            
                            boxes.append((x, y, width, height))
        except Exception as e:
            print(f"Error parsing {coord_file}: {e}")
            
        return boxes
    
    def convert_to_yolo_format(self, img_width: int, img_height: int, 
                              boxes: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Convert bounding boxes to YOLO format (normalized coordinates).
        
        Args:
            img_width: Image width
            img_height: Image height
            boxes: List of bounding boxes as (x, y, width, height)
            
        Returns:
            List of YOLO format annotations as strings
        """
        yolo_annotations = []
        
        for x, y, width, height in boxes:
            # Convert to center coordinates
            center_x = x + width / 2
            center_y = y + height / 2
            
            # Normalize coordinates
            norm_center_x = center_x / img_width
            norm_center_y = center_y / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            # Ensure coordinates are within [0, 1]
            norm_center_x = max(0, min(1, norm_center_x))
            norm_center_y = max(0, min(1, norm_center_y))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            # YOLO format: class_id center_x center_y width height
            yolo_line = f"0 {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_annotations.append(yolo_line)
        
        return yolo_annotations
    
    def process_item_directory(self, item_dir: Path) -> List[Tuple[str, str]]:
        """
        Process a single item directory and convert all images/annotations.
        
        Args:
            item_dir: Path to item directory (e.g., "1", "2", etc.)
            
        Returns:
            List of (image_path, label_path) tuples
        """
        results = []
        
        # Check if directory has video folder
        video_dir = item_dir / "video"
        coord_file = item_dir / "coordinates.txt"
        
        if not video_dir.exists() or not coord_file.exists():
            print(f"Skipping {item_dir}: missing video folder or coordinates.txt")
            return results
        
        # Parse coordinates
        boxes = self.parse_coordinates(coord_file)
        if not boxes:
            print(f"No valid annotations found in {coord_file}")
            return results
        
        # Process each image in the video folder
        for img_file in video_dir.glob("*.png"):
            try:
                # Read image to get dimensions
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Could not read image: {img_file}")
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Convert boxes to YOLO format
                yolo_annotations = self.convert_to_yolo_format(img_width, img_height, boxes)
                
                if yolo_annotations:
                    # Create unique filename
                    unique_name = f"{item_dir.name}_{img_file.stem}"
                    
                    # Image path
                    img_dest = f"{unique_name}.png"
                    
                    # Label path
                    label_dest = f"{unique_name}.txt"
                    
                    results.append((img_dest, label_dest, yolo_annotations, str(img_file)))
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        return results
    
    def split_dataset(self, all_items: List[Tuple[str, str, List[str], str]]) -> Dict[str, List]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            all_items: List of all processed items
            
        Returns:
            Dictionary with train, val, test splits
        """
        random.shuffle(all_items)
        
        total = len(all_items)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)
        
        return {
            'train': all_items[:train_end],
            'val': all_items[train_end:val_end],
            'test': all_items[val_end:]
        }
    
    def copy_files(self, items: List[Tuple[str, str, List[str], str]], split: str):
        """
        Copy image and label files to the appropriate split directory.
        
        Args:
            items: List of items to copy
            split: Split name ('train', 'val', 'test')
        """
        images_dir = self.output_path / "images" / split
        labels_dir = self.output_path / "labels" / split
        
        for img_dest, label_dest, yolo_annotations, img_src in items:
            try:
                # Copy image
                shutil.copy2(img_src, images_dir / img_dest)
                
                # Write label file
                label_path = labels_dir / label_dest
                with open(label_path, 'w') as f:
                    for annotation in yolo_annotations:
                        f.write(annotation + '\n')
                        
            except Exception as e:
                print(f"Error copying {img_src}: {e}")
    
    def convert_dataset(self):
        """Main method to convert the entire dataset."""
        print("Starting GroZi-120 dataset conversion...")
        
        # Get all item directories
        item_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        item_dirs.sort(key=lambda x: int(x.name))
        
        print(f"Found {len(item_dirs)} item directories")
        
        # Process all items
        all_items = []
        for item_dir in item_dirs:
            print(f"Processing {item_dir.name}...")
            items = self.process_item_directory(item_dir)
            all_items.extend(items)
        
        print(f"Total processed items: {len(all_items)}")
        
        if not all_items:
            print("No items processed. Check dataset structure.")
            return
        
        # Split dataset
        splits = self.split_dataset(all_items)
        
        # Copy files to appropriate directories
        for split_name, split_items in splits.items():
            print(f"Copying {len(split_items)} items to {split_name} split...")
            self.copy_files(split_items, split_name)
        
        # Print statistics
        print("\nDataset conversion completed!")
        print(f"Train: {len(splits['train'])} images")
        print(f"Val: {len(splits['val'])} images") 
        print(f"Test: {len(splits['test'])} images")
        print(f"Total: {len(all_items)} images")
        print(f"Output directory: {self.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert GroZi-120 dataset to YOLO format")
    parser.add_argument("--dataset-path", type=str, default="../data/GroZi-120",
                       help="Path to original GroZi-120 dataset")
    parser.add_argument("--output-path", type=str, default="../data/GroZi-120-yolo",
                       help="Path where YOLO format dataset will be saved")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Fraction of data for validation")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Convert dataset
    converter = GroZiDatasetConverter(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    converter.convert_dataset()

if __name__ == "__main__":
    main()
