import os
import numpy as np
from PIL import Image
import shutil
from pathlib import Path
from datetime import datetime

class ImageRatioAnalyzer:
    def __init__(self, input_folder, output_folder=None):
        """
        Initialize the analyzer
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to save filtered images (default: input_folder + '_filtered')
        """
        self.input_folder = Path(input_folder)
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = Path(str(input_folder) + '_filtered_below30')
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Region size (width × height)
        self.region_width = 750
        self.region_height = 1250
        
        # Results storage
        self.results = []
        
    def get_image_files(self):
        """Get all image files from the input folder"""
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_folder.glob(f'*{ext}'))
            image_files.extend(self.input_folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def extract_bottom_left_region(self, image_array):
        """
        Extract bottom-left 750×1250 region from image
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            Extracted region or None if image is too small
        """
        height, width = image_array.shape[:2]
        
        # Check if image is large enough
        if height < self.region_height or width < self.region_width:
            return None
        
        # Extract bottom-left corner
        # Bottom-left means: from (height-1250) to height, from 0 to 750
        region = image_array[height-self.region_height:height, 0:self.region_width]
        
        return region
    
    def calculate_255_ratio(self, region):
        """
        Calculate the ratio of pixels with value 255 in the region
        
        Args:
            region: numpy array of the image region
            
        Returns:
            Ratio of 255 pixels (0 to 1)
        """
        # Handle different image types
        if len(region.shape) == 2:
            # Grayscale image
            count_255 = np.sum(region == 255)
        else:
            # Color image - check if all channels are 255 (white pixel)
            # For RGB, a white pixel is (255, 255, 255)
            if region.shape[2] == 3:
                white_pixels = np.all(region == 255, axis=2)
            elif region.shape[2] == 4:
                # RGBA - check RGB channels only
                white_pixels = np.all(region[:, :, :3] == 255, axis=2)
            else:
                # Other formats - use first channel
                white_pixels = region[:, :, 0] == 255
            
            count_255 = np.sum(white_pixels)
        
        total_pixels = self.region_width * self.region_height
        ratio = count_255 / total_pixels
        
        return ratio
    
    def process_all_images(self):
        """Process all images in the input folder"""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images in {self.input_folder}")
        print(f"Processing images (extracting {self.region_width}×{self.region_height} bottom-left region)...\n")
        print("-" * 80)
        print(f"{'Filename':<40} {'255 Ratio':<15} {'Percentage':<15} {'Status':<10}")
        print("-" * 80)
        
        saved_count = 0
        
        for img_path in image_files:
            try:
                # Load image
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Extract region
                region = self.extract_bottom_left_region(img_array)
                
                if region is None:
                    print(f"{img_path.name:<40} {'Image too small':<15} {'-':<15} {'SKIPPED':<10}")
                    continue
                
                # Calculate ratio
                ratio = self.calculate_255_ratio(region)
                percentage = ratio * 100
                
                # Store result
                result = {
                    'filename': img_path.name,
                    'path': img_path,
                    'ratio': ratio,
                    'percentage': percentage
                }
                self.results.append(result)
                
                # Determine if we should save this image
                status = ""
                if percentage < 30:
                    # Copy image to output folder
                    output_path = self.output_folder / img_path.name
                    shutil.copy2(img_path, output_path)
                    saved_count += 1
                    status = "SAVED"
                else:
                    status = "SKIPPED"
                
                # Print result
                print(f"{img_path.name:<40} {ratio:<15.4f} {percentage:<14.2f}% {status:<10}")
                
            except Exception as e:
                print(f"{img_path.name:<40} Error: {str(e)}")
                continue
        
        print("-" * 80)
        print(f"\nProcessing complete!")
        print(f"Total images processed: {len(self.results)}")
        print(f"Images with <30% white pixels: {saved_count}")
        print(f"Saved to: {self.output_folder}")
        
        # Generate summary statistics
        self.print_summary()
        
        # Save detailed report
        self.save_report()
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            return
        
        ratios = [r['percentage'] for r in self.results]
        
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Average 255-pixel ratio: {np.mean(ratios):.2f}%")
        print(f"Median 255-pixel ratio: {np.median(ratios):.2f}%")
        print(f"Min 255-pixel ratio: {np.min(ratios):.2f}%")
        print(f"Max 255-pixel ratio: {np.max(ratios):.2f}%")
        print(f"Std deviation: {np.std(ratios):.2f}%")
        
        # Distribution
        print("\nDistribution:")
        ranges = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
        for low, high in ranges:
            count = sum(1 for r in ratios if low <= r < high)
            print(f"  {low:3d}% - {high:3d}%: {count:3d} images {'*' * min(count, 50)}")
    
    def save_report(self):
        """Save detailed report to CSV file"""
        report_path = self.output_folder / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(report_path, 'w') as f:
            f.write("Filename,255_Pixel_Ratio,Percentage,Saved\n")
            for result in self.results:
                saved = "Yes" if result['percentage'] < 30 else "No"
                f.write(f"{result['filename']},{result['ratio']:.6f},{result['percentage']:.2f},{saved}\n")
        
        print(f"\nDetailed report saved to: {report_path}")

def visualize_region(image_path, save_preview=False):
    """
    Visualize the 750×1250 region extraction for a single image
    
    Args:
        image_path: Path to the image
        save_preview: Whether to save the preview
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    img = Image.open(image_path)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    region_width = 750
    region_height = 1250
    
    if height < region_height or width < region_width:
        print(f"Image too small: {width}x{height} (need at least {region_width}×{region_height})")
        return
    
    # Extract region
    region = img_array[height-region_height:height, 0:region_width]
    
    # Calculate ratio
    if len(region.shape) == 2:
        count_255 = np.sum(region == 255)
    else:
        white_pixels = np.all(region[:, :, :3] == 255, axis=2) if region.shape[2] >= 3 else region[:, :, 0] == 255
        count_255 = np.sum(white_pixels)
    
    ratio = count_255 / (region_width * region_height)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with region marked
    axes[0].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    rect = patches.Rectangle((0, height-region_height), region_width, region_height, 
                            linewidth=3, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title(f'Original Image ({width}x{height})')
    axes[0].axis('off')
    
    # Extracted region
    axes[1].imshow(region, cmap='gray' if len(region.shape) == 2 else None)
    axes[1].set_title(f'Extracted Region ({region_width}×{region_height})')
    axes[1].axis('off')
    
    # Binary mask showing 255 pixels
    if len(region.shape) == 2:
        mask = region == 255
    else:
        mask = np.all(region[:, :, :3] == 255, axis=2) if region.shape[2] >= 3 else region[:, :, 0] == 255
    
    axes[2].imshow(mask, cmap='binary')
    axes[2].set_title(f'255-Value Pixels (Ratio: {ratio*100:.2f}%)')
    axes[2].axis('off')
    
    plt.suptitle(f'Image: {os.path.basename(image_path)}')
    plt.tight_layout()
    
    if save_preview:
        preview_path = f"preview_{os.path.basename(image_path)}"
        plt.savefig(preview_path, dpi=100, bbox_inches='tight')
        print(f"Preview saved to: {preview_path}")
    
    plt.show()

def main():
    """Main function"""
    print("=" * 60)
    print("IMAGE 255-VALUE RATIO ANALYZER")
    print("Analyzes bottom-left 750×1250 region for 255-value pixels")
    print("=" * 60)
    
    # Get input folder
    input_folder = "data"
    
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist!")
        return
    

    output_folder = "light_cloud"
    
    # Create analyzer and process images
    analyzer = ImageRatioAnalyzer(input_folder, output_folder)
    analyzer.process_all_images()
    
    # Option to visualize a specific image
    visualize = input("\nVisualize region extraction for a specific image? (y/n): ").strip().lower()
    if visualize == 'y':
        image_name = input("Enter image filename from the folder: ").strip()
        image_path = Path(input_folder) / image_name
        if image_path.exists():
            visualize_region(image_path, save_preview=True)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()