#!/usr/bin/env python3
"""
YOLOv8 Inference Script for Food Detection
Performs real-time food detection using trained YOLOv8 model.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
from typing import List, Dict, Tuple, Optional
import json

class FoodDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize the food detector.
        
        Args:
            model_path: Path to trained YOLOv8 model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use (auto, cpu, 0, 1, etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        try:
            self.model = YOLO(model_path)
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect_food(self, image: np.ndarray) -> List[Dict]:
        """
        Detect food items in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class
        """
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        detection = {
                            'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': 'food_item'  # Update based on your classes
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with detection boxes drawn
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Convert bbox to integer coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process a single image and detect food items.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            
        Returns:
            Dictionary with detection results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect food items
        detections = self.detect_food(image)
        
        # Draw detections
        result_image = self.draw_detections(image, detections)
        
        # Save output image if specified
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"📸 Output image saved to: {output_path}")
        
        # Prepare results
        results = {
            'image_path': image_path,
            'num_detections': len(detections),
            'detections': detections,
            'output_path': output_path
        }
        
        return results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     show_preview: bool = False) -> Dict:
        """
        Process a video and detect food items in each frame.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_preview: Whether to show real-time preview
            
        Returns:
            Dictionary with video processing results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎬 Processing video: {video_path}")
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Process frames
        frame_count = 0
        total_detections = 0
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            start_time = time.time()
            detections = self.detect_food(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            total_detections += len(detections)
            
            # Draw detections
            result_frame = self.draw_detections(frame, detections)
            
            # Add processing info to frame
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | FPS: {1/processing_time:.1f}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if writer:
                writer.write(result_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('Food Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"📹 Processed {frame_count}/{total_frames} frames...")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Calculate statistics
        avg_processing_time = np.mean(processing_times)
        avg_fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
        
        results = {
            'video_path': video_path,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            'avg_processing_time': avg_processing_time,
            'avg_fps': avg_fps,
            'output_path': output_path
        }
        
        print(f"✅ Video processing completed!")
        print(f"📊 Results: {frame_count} frames, {total_detections} detections, {avg_fps:.1f} FPS")
        
        return results
    
    def webcam_detection(self, camera_id: int = 0, output_path: Optional[str] = None):
        """
        Perform real-time food detection using webcam.
        
        Args:
            camera_id: Camera device ID
            output_path: Path to save video recording (optional)
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
        
        print(f"📹 Starting webcam detection... Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Process frame
            detections = self.detect_food(frame)
            
            # Draw detections
            result_frame = self.draw_detections(frame, detections)
            
            # Add info to frame
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_text = f"Detections: {len(detections)} | FPS: {fps:.1f} | Press 'q' to quit"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if writer:
                writer.write(result_frame)
            
            # Show frame
            cv2.imshow('Food Detection - Webcam', result_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Webcam detection stopped. Processed {frame_count} frames.")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Food Detection Inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained YOLOv8 model")
    parser.add_argument("--source", type=str, required=True,
                       help="Source: image path, video path, or 'webcam'")
    parser.add_argument("--output", type=str,
                       help="Output path for processed image/video")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--show", action="store_true",
                       help="Show real-time preview for video/webcam")
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = FoodDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        sys.exit(1)
    
    # Process based on source type
    try:
        if args.source.lower() == 'webcam':
            # Webcam detection
            detector.webcam_detection(output_path=args.output)
            
        elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video processing
            results = detector.process_video(
                video_path=args.source,
                output_path=args.output,
                show_preview=args.show
            )
            
            # Save results
            if args.output:
                results_file = args.output.replace('.mp4', '_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"📄 Results saved to: {results_file}")
                
        else:
            # Image processing
            results = detector.process_image(
                image_path=args.source,
                output_path=args.output
            )
            
            # Print results
            print(f"\n📊 Detection Results:")
            print(f"   Image: {results['image_path']}")
            print(f"   Detections: {results['num_detections']}")
            
            for i, detection in enumerate(results['detections']):
                print(f"   Detection {i+1}: {detection['class_name']} "
                      f"(confidence: {detection['confidence']:.3f})")
            
            # Save results
            if args.output:
                results_file = args.output.replace('.jpg', '_results.json').replace('.png', '_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"📄 Results saved to: {results_file}")
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)
    
    print(f"✅ Processing completed successfully!")

if __name__ == "__main__":
    main()
