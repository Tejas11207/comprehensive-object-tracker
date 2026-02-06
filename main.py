import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import cv2
from scipy import ndimage
from collections import defaultdict

class ComprehensiveObjectTracker:
    """
    Comprehensive real-time object detection and tracking system
    Detects and marks: buildings, houses, roads, vehicles, mountains, water bodies, and more
    """
    
    def __init__(self):
        """Initialize segmentation models for comprehensive detection"""
        print("🚀 Loading models for object detection...")
        
        # Primary high-accuracy segmenter
        self.segmenter = pipeline(
            "image-segmentation",
            model="nvidia/segformer-b5-finetuned-ade-640-640",
            device=-1  # CPU
        )
        
        # Backup segmenter for ensemble (better accuracy)
        self.segmenter_backup = pipeline(
            "image-segmentation",
            model="nvidia/segformer-b3-finetuned-ade-512-512",
            device=-1
        )
        
        # Define comprehensive object categories
        self.object_categories = {
            "buildings": ["building", "house", "skyscraper", "edifice", "tower", "apartment building"],
            "roads": ["road", "path", "street", "highway", "sidewalk", "crosswalk"],
            "vehicles": ["car", "truck", "bus", "van", "bicycle", "motorcycle", "vehicle"],
            "nature": ["mountain", "hill", "rock", "stone", "tree", "grass", "plant", "flower"],
            "water": ["water", "sea", "river", "lake", "pool", "waterfall", "ocean", "pond"],
            "sky": ["sky", "cloud"],
            "people": ["person", "people"],
            "infrastructure": ["bridge", "fence", "wall", "pole", "traffic light", "sign"],
            "terrain": ["field","sand, "dirt", "soil", "ground"]
        }
        
        # Define color scheme for each category
        self.category_colors = {
            "buildings": (255, 0, 0),        # Red
            "roads": (128, 128, 128),        # Gray
            "vehicles": (255, 165, 0),       # Orange
            "nature": (0, 255, 0),           # Green
            "water": (0, 100, 255),          # Blue
            "sky": (135, 206, 235),          # Sky Blue
            "people": (255, 255, 0),         # Yellow
            "infrastructure": (128, 0, 128), # Purple
            "terrain": (139, 69, 19)         # Brown
        }
        
        print("✅ Models loaded successfully!")
        print(f"📦 Tracking {sum(len(v) for v in self.object_categories.values())} object types across {len(self.object_categories)} categories")
    
    def get_category_for_label(self, label):
        """Determine which category a label belongs to"""
        for category, labels in self.object_categories.items():
            if label.lower() in labels:
                return category
        return "other"
    
    def preprocess_image(self, image_path, target_size=(640, 640)):
        """Enhanced preprocessing for better detection"""
        image = Image.open(image_path).convert("RGB")
        self.original_size = image.size
        
        # Convert to numpy for processing
        image_np = np.array(image)
        
        # Enhance image quality
        # 1. Increase contrast
        image_np = cv2.convertScaleAbs(image_np, alpha=1.3, beta=15)
        
        # 2. Denoise while preserving edges
        image_np = cv2.bilateralFilter(image_np, 9, 75, 75)
        
        # 3. Sharpen image
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        image_np = cv2.filter2D(image_np, -1, kernel)
        
        image_enhanced = Image.fromarray(image_np)
        image_resized = image_enhanced.resize(target_size, Image.Resampling.LANCZOS)
        
        return image_resized, image
    
    def get_segments_ensemble(self, image):
        """Get segmentation results using ensemble of models"""
        print("🔍 Running AI segmentation (ensemble mode)...")
        
        # Get results from both models
        segments_primary = self.segmenter(image)
        segments_backup = self.segmenter_backup(image)
        
        # Merge for better accuracy
        merged_segments = self._merge_segments(segments_primary, segments_backup)
        
        return merged_segments
    
    def _merge_segments(self, seg1, seg2):
        """Merge segments from two models using consensus"""
        merged = defaultdict(list)
        
        # Collect all masks for each label
        for seg in seg1:
            merged[seg["label"]].append(seg["mask"])
        
        for seg in seg2:
            merged[seg["label"]].append(seg["mask"])
        
        # Create consensus masks
        result = []
        for label, masks in merged.items():
            if len(masks) > 0:
                # Average masks and threshold
                mask_arrays = [np.array(m).astype(float) for m in masks]
                consensus_mask = np.mean(mask_arrays, axis=0)
                consensus_mask = (consensus_mask > 0.4).astype(np.uint8) * 255
                
                result.append({
                    "label": label,
                    "mask": Image.fromarray(consensus_mask)
                })
        
        return result
    
    def extract_all_objects(self, segments, min_area_ratio=0.0005):
        """
        Extract ALL detected objects with bounding boxes
        
        Args:
            segments: Segmentation results
            min_area_ratio: Minimum object size (relative to image)
        
        Returns:
            List of detected objects with bounding boxes
        """
        detected_objects = []
        
        for seg in segments:
            label = seg["label"]
            category = self.get_category_for_label(label)
            
            # Skip if not in our tracking categories
            if category == "other":
                continue
            
            mask = np.array(seg["mask"])
            
            # Find contours (object boundaries)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Calculate minimum area
            min_area = (mask.shape[0] * mask.shape[1]) * min_area_ratio
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detected_objects.append({
                        "label": label,
                        "category": category,
                        "bbox": (x, y, x + w, y + h),
                        "area": area,
                        "confidence": "high" if area > min_area * 5 else "medium"
                    })
        
        return detected_objects
    
    def draw_detections(self, image, objects):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: PIL Image
            objects: List of detected objects
        
        Returns:
            Annotated image
        """
        image_annotated = image.copy()
        draw = ImageDraw.Draw(image_annotated)
        
        # Load font
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw each detected object
        for obj in objects:
            category = obj["category"]
            label = obj["label"]
            x1, y1, x2, y2 = obj["bbox"]
            
            # Get color for this category
            color = self.category_colors.get(category, (0, 255, 0))
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Prepare label text
            text = f"{label.upper()}"
            
            # Calculate text dimensions
            try:
                bbox = draw.textbbox((x1, y1), text, font=font_small)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width, text_height = draw.textsize(text, font=font_small)
            
            # Draw background for text
            padding = 6
            draw.rectangle(
                [x1, y1 - text_height - padding*2, x1 + text_width + padding*2, y1],
                fill=color
            )
            
            # Draw text
            draw.text(
                (x1 + padding, y1 - text_height - padding),
                text,
                fill=(255, 255, 255),
                font=font_small
            )
        
        return image_annotated
    
    def scale_objects_to_original(self, objects, original_size, processed_size):
        """Scale bounding boxes to original image dimensions"""
        scale_x = original_size[0] / processed_size[0]
        scale_y = original_size[1] / processed_size[1]
        
        scaled_objects = []
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            scaled_objects.append({
                "label": obj["label"],
                "category": obj["category"],
                "bbox": (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ),
                "area": obj["area"] * scale_x * scale_y,
                "confidence": obj["confidence"]
            })
        
        return scaled_objects
    
    def detect_and_track_objects(self, image_path, output_prefix="tracked"):
        """
        Main detection pipeline - detects and tracks ALL objects
        
        Args:
            image_path: Path to input image
            output_prefix: Prefix for output files
        
        Returns:
            Detection results
        """
        print("\n" + "="*70)
        print(f"🎯 COMPREHENSIVE OBJECT DETECTION & TRACKING")
        print("="*70)
        print(f"📷 Processing: {image_path}\n")
        
        # Step 1: Preprocess
        image_processed, image_original = self.preprocess_image(image_path)
        
        # Step 2: Segment
        segments = self.get_segments_ensemble(image_processed)
        
        # Step 3: Extract all objects
        print("🔎 Extracting detected objects...")
        objects = self.extract_all_objects(segments)
        
        if not objects:
            print("⚠️  No objects detected in image")
            return None
        
        # Step 4: Scale to original size
        objects_scaled = self.scale_objects_to_original(
            objects,
            self.original_size,
            image_processed.size
        )
        
        # Step 5: Draw detections
        print(f"✏️  Drawing {len(objects_scaled)} detected objects...")
        result_image = self.draw_detections(image_original, objects_scaled)
        
        # Step 6: Generate statistics
        stats = self._generate_statistics(objects_scaled, segments)
        
        # Step 7: Visualize
        self._create_visualization(image_original, result_image, stats, output_prefix)
        
        # Step 8: Save results
        self._save_comprehensive_results(
            result_image,
            objects_scaled,
            stats,
            segments,
            output_prefix
        )
        
        # Print summary
        self._print_detection_summary(stats)
        
        return {
            "original": image_original,
            "annotated": result_image,
            "objects": objects_scaled,
            "statistics": stats,
            "segments": segments
        }
    
    def _generate_statistics(self, objects, segments):
        """Generate comprehensive statistics"""
        stats = {
            "total_objects": len(objects),
            "by_category": defaultdict(int),
            "by_label": defaultdict(int),
            "all_labels_found": set()
        }
        
        # Count by category and label
        for obj in objects:
            stats["by_category"][obj["category"]] += 1
            stats["by_label"][obj["label"]] += 1
        
        # All detected segments
        for seg in segments:
            stats["all_labels_found"].add(seg["label"])
        
        return stats
    
    def _print_detection_summary(self, stats):
        """Print detection summary to console"""
        print("\n" + "="*70)
        print("📊 DETECTION SUMMARY")
        print("="*70)
        print(f"✅ Total Objects Detected: {stats['total_objects']}\n")
        
        print("📦 By Category:")
        for category, count in sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True):
            emoji_map = {
                "buildings": "🏢",
                "roads": "🛣️",
                "vehicles": "🚗",
                "nature": "🌳",
                "water": "💧",
                "sky": "☁️",
                "people": "👤",
                "infrastructure": "🌉",
                "terrain": "🏔️"
            }
            emoji = emoji_map.get(category, "📌")
            print(f"  {emoji} {category.capitalize()}: {count}")
        
        print(f"\n🏷️  Unique Object Types: {len(stats['by_label'])}")
        print("="*70 + "\n")
    
    def _create_visualization(self, original, annotated, stats, prefix):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(24, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(original)
        ax1.set_title("Original Image", fontsize=18, fontweight='bold')
        ax1.axis('off')
        
        # Annotated image
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.imshow(annotated)
        ax2.set_title(f"Detected Objects ({stats['total_objects']} found)", 
                     fontsize=18, fontweight='bold')
        ax2.axis('off')
        
        # Statistics - Category breakdown
        ax3 = fig.add_subplot(gs[0, 2])
        categories = list(stats["by_category"].keys())
        counts = list(stats["by_category"].values())
        colors = [self.category_colors.get(cat, (0, 255, 0)) for cat in categories]
        colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors]
        
        ax3.barh(categories, counts, color=colors_normalized)
        ax3.set_xlabel('Count', fontsize=12)
        ax3.set_title('Objects by Category', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Statistics - Top object types
        ax4 = fig.add_subplot(gs[1, 2])
        top_labels = sorted(stats["by_label"].items(), key=lambda x: x[1], reverse=True)[:10]
        if top_labels:
            labels, label_counts = zip(*top_labels)
            ax4.barh(labels, label_counts, color='steelblue')
            ax4.set_xlabel('Count', fontsize=12)
            ax4.set_title('Top 10 Object Types', fontsize=14, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Comprehensive Object Detection & Tracking Results', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(f"{prefix}_visualization.png", dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved: {prefix}_visualization.png")
        plt.show()
    
    def _save_comprehensive_results(self, annotated_image, objects, stats, segments, prefix):
        """Save all results to files"""
        # Save annotated image
        annotated_image.save(f"{prefix}_tracked.png")
        print(f"💾 Annotated image saved: {prefix}_tracked.png")
        
        # Save detailed detection report
        with open(f"{prefix}_detection_report.txt", 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE OBJECT DETECTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Objects Detected: {stats['total_objects']}\n\n")
            
            # By category
            f.write("DETECTION BY CATEGORY:\n")
            f.write("-"*70 + "\n")
            for category, count in sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {category.upper()}: {count} objects\n")
            f.write("\n")
            
            # Detailed object list
            f.write("DETAILED OBJECT LIST:\n")
            f.write("-"*70 + "\n\n")
            
            # Group by category
            objects_by_cat = defaultdict(list)
            for obj in objects:
                objects_by_cat[obj["category"]].append(obj)
            
            for category, cat_objects in sorted(objects_by_cat.items()):
                f.write(f"[{category.upper()}] - {len(cat_objects)} objects:\n")
                for i, obj in enumerate(cat_objects, 1):
                    x1, y1, x2, y2 = obj["bbox"]
                    f.write(f"  {i}. {obj['label'].capitalize()}\n")
                    f.write(f"      Position: ({x1}, {y1}) to ({x2}, {y2})\n")
                    f.write(f"      Size: {x2-x1}×{y2-y1} pixels\n")
                    f.write(f"      Confidence: {obj['confidence']}\n")
                f.write("\n")
        
        print(f"💾 Detection report saved: {prefix}_detection_report.txt")
        
        # Save all detected segments
        with open(f"{prefix}_all_segments.txt", 'w') as f:
            f.write("ALL DETECTED SEGMENTS (AI Model Output):\n")
            f.write("="*70 + "\n\n")
            for label in sorted(stats["all_labels_found"]):
                f.write(f"  • {label}\n")
        
        print(f"💾 Segment list saved: {prefix}_all_segments.txt")
        
        print("\n✅ All results saved successfully!\n")


def main():
    """Main execution - comprehensive object detection"""
    
    # Initialize tracker
    tracker = ComprehensiveObjectTracker()
    
    # Run comprehensive detection
    results = tracker.detect_and_track_objects(
        image_path="village2.PNG",
        output_prefix="comprehensive_tracking"
    )
    
    results2 = tracker.detect_and_track_objects(
        image_path="village2.PNG",
        Image_path="village2.PNG",
        output_prefix="comprehensive_tracking2"
    )
    
    if results:
        print("="*70)
        print("🎉 PROCESSING COMPLETE!")
        print("="*70)
        print("\n📁 Output Files:")
        print("  1. comprehensive_tracking_tracked.png - Annotated image with all detections")
        print("  2. comprehensive_tracking_visualization.png - Full visualization with statistics")
        print("  3. comprehensive_tracking_detection_report.txt - Detailed detection report")
        print("  4. comprehensive_tracking_all_segments.txt - Complete segment list")
        print("\n")


if __name__ == "__main__":
    main()
