import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from datetime import datetime

class SatelliteImageInpainter:
    """
    Inpainting class for satellite images with cloud/missing data (255 values)
    """
    
    def __init__(self, input_folder, output_folder=None):
        """
        Initialize the inpainter
        
        Args:
            input_folder: Path to folder containing images with clouds/missing data
            output_folder: Path to save inpainted images
        """
        self.input_folder = Path(input_folder)
        
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = Path(str(input_folder) + '_inpainted')
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders for different methods
        self.methods_folders = {}
        for method in ['telea', 'ns', 'hybrid', 'statistical', 'comparison']:
            folder = self.output_folder / method
            folder.mkdir(exist_ok=True)
            self.methods_folders[method] = folder
    
    def create_mask_from_255(self, image):
        """
        Create a binary mask where 255 values are marked as areas to inpaint
        
        Args:
            image: Input image array
            
        Returns:
            Binary mask (255 for areas to inpaint, 0 for valid data)
        """
        if len(image.shape) == 2:
            # Grayscale
            mask = (image == 255).astype(np.uint8) * 255
        else:
            # Color image - check if all channels are 255
            mask = np.all(image == 255, axis=2).astype(np.uint8) * 255
        
        return mask
    
    def dilate_mask(self, mask, kernel_size=3):
        """
        Slightly dilate the mask to ensure complete coverage of cloud edges
        
        Args:
            mask: Binary mask
            kernel_size: Size of dilation kernel
            
        Returns:
            Dilated mask
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated
    
    def inpaint_telea(self, image, mask):
        """
        Inpaint using Telea method (Fast Marching Method)
        Good for smooth, gradual transitions
        
        Args:
            image: Input image
            mask: Binary mask of areas to inpaint
            
        Returns:
            Inpainted image
        """
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    def inpaint_ns(self, image, mask):
        """
        Inpaint using Navier-Stokes method
        Better for larger areas and texture propagation
        
        Args:
            image: Input image
            mask: Binary mask of areas to inpaint
            
        Returns:
            Inpainted image
        """
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    def statistical_inpaint(self, image, mask):
        """
        Statistical inpainting using median filtering and interpolation
        Suitable for satellite imagery with repetitive patterns
        
        Args:
            image: Input image
            mask: Binary mask of areas to inpaint
            
        Returns:
            Inpainted image
        """
        result = image.copy()
        
        # Convert mask to boolean
        mask_bool = mask > 0
        
        if len(image.shape) == 2:
            # Grayscale
            # Use median filter to estimate values
            filtered = ndimage.median_filter(image, size=5)
            result[mask_bool] = filtered[mask_bool]
            
            # Refine with Gaussian smoothing
            for _ in range(3):
                temp = result.copy()
                temp[mask_bool] = ndimage.gaussian_filter(result, sigma=2)[mask_bool]
                result = temp
        else:
            # Color image - process each channel
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                filtered = ndimage.median_filter(channel, size=5)
                result[:, :, i][mask_bool] = filtered[mask_bool]
                
                # Refine with Gaussian smoothing
                for _ in range(3):
                    temp = result[:, :, i].copy()
                    temp[mask_bool] = ndimage.gaussian_filter(result[:, :, i], sigma=2)[mask_bool]
                    result[:, :, i] = temp
        
        return result
    
    def hybrid_inpaint(self, image, mask):
        """
        Hybrid approach combining multiple methods
        Best for satellite imagery with varied cloud patterns
        
        Args:
            image: Input image
            mask: Binary mask of areas to inpaint
            
        Returns:
            Inpainted image
        """
        # Step 1: Statistical filling for initial estimate
        statistical = self.statistical_inpaint(image, mask)
        
        # Step 2: Refine edges with Telea
        refined_mask = self.dilate_mask(mask, kernel_size=2)
        result = cv2.inpaint(statistical, refined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Step 3: Smooth transitions
        mask_bool = mask > 0
        if len(image.shape) == 2:
            smoothed = cv2.bilateralFilter(result, 9, 75, 75)
            result[mask_bool] = smoothed[mask_bool]
        else:
            smoothed = cv2.bilateralFilter(result, 9, 75, 75)
            for i in range(image.shape[2]):
                result[:, :, i][mask_bool] = smoothed[:, :, i][mask_bool]
        
        return result
    
    def smart_inpaint(self, image, mask, method='hybrid'):
        """
        Smart inpainting with preprocessing and postprocessing
        
        Args:
            image: Input image
            mask: Binary mask
            method: Inpainting method to use
            
        Returns:
            Inpainted image
        """
        # Preprocess: Remove small isolated mask regions (noise)
        cleaned_mask = morphology.remove_small_objects(mask.astype(bool), min_size=10)
        cleaned_mask = (cleaned_mask * 255).astype(np.uint8)
        
        # Choose inpainting method
        if method == 'telea':
            result = self.inpaint_telea(image, cleaned_mask)
        elif method == 'ns':
            result = self.inpaint_ns(image, cleaned_mask)
        elif method == 'statistical':
            result = self.statistical_inpaint(image, cleaned_mask)
        else:  # hybrid
            result = self.hybrid_inpaint(image, cleaned_mask)
        
        # Postprocess: Ensure no 255 values remain
        if len(result.shape) == 2:
            remaining_255 = result == 255
            if np.any(remaining_255):
                result[remaining_255] = ndimage.median_filter(result, size=3)[remaining_255]
        else:
            for i in range(result.shape[2]):
                remaining_255 = result[:, :, i] == 255
                if np.any(remaining_255):
                    result[:, :, i][remaining_255] = ndimage.median_filter(
                        result[:, :, i], size=3)[remaining_255]
        
        return result
    
    def process_single_image(self, image_path, save_all_methods=False):
        """
        Process a single image with inpainting
        
        Args:
            image_path: Path to the image
            save_all_methods: If True, save results from all methods
            
        Returns:
            Dictionary with results from different methods
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.array(Image.open(image_path))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Create mask
        mask = self.create_mask_from_255(img)
        
        # Calculate cloud coverage
        cloud_percentage = (np.sum(mask > 0) / mask.size) * 100
        
        results = {
            'original': img,
            'mask': mask,
            'cloud_percentage': cloud_percentage
        }
        
        # Apply different methods
        print(f"  Applying Telea method...")
        results['telea'] = self.smart_inpaint(img, mask, 'telea')
        
        print(f"  Applying Navier-Stokes method...")
        results['ns'] = self.smart_inpaint(img, mask, 'ns')
        
        print(f"  Applying Statistical method...")
        results['statistical'] = self.smart_inpaint(img, mask, 'statistical')
        
        print(f"  Applying Hybrid method...")
        results['hybrid'] = self.smart_inpaint(img, mask, 'hybrid')
        
        # Save results
        filename = Path(image_path).name
        
        if save_all_methods:
            for method in ['telea', 'ns', 'statistical', 'hybrid']:
                output_path = self.methods_folders[method] / filename
                cv2.imwrite(str(output_path), results[method])
        else:
            # Save only the best (hybrid) method
            output_path = self.output_folder / filename
            cv2.imwrite(str(output_path), results['hybrid'])
        
        return results
    
    def create_comparison_image(self, results, filename):
        """
        Create a comparison image showing all methods
        
        Args:
            results: Dictionary with inpainting results
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Original (Cloud: {results['cloud_percentage']:.1f}%)")
        axes[0, 0].axis('off')
        
        # Mask
        axes[0, 1].imshow(results['mask'], cmap='gray')
        axes[0, 1].set_title("Cloud Mask")
        axes[0, 1].axis('off')
        
        # Telea
        axes[0, 2].imshow(cv2.cvtColor(results['telea'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Telea Method")
        axes[0, 2].axis('off')
        
        # NS
        axes[1, 0].imshow(cv2.cvtColor(results['ns'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Navier-Stokes Method")
        axes[1, 0].axis('off')
        
        # Statistical
        axes[1, 1].imshow(cv2.cvtColor(results['statistical'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Statistical Method")
        axes[1, 1].axis('off')
        
        # Hybrid
        axes[1, 2].imshow(cv2.cvtColor(results['hybrid'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Hybrid Method (Best)")
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Inpainting Comparison: {filename}")
        plt.tight_layout()
        
        comparison_path = self.methods_folders['comparison'] / f"comparison_{filename}"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_folder(self, save_all_methods=False, create_comparisons=True):
        """
        Process all images in the input folder
        
        Args:
            save_all_methods: If True, save results from all methods
            create_comparisons: If True, create comparison images
        """
        # Get all image files
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_folder.glob(f'*{ext}'))
            image_files.extend(self.input_folder.glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print("=" * 60)
        
        # Process statistics
        stats = {
            'total': len(image_files),
            'processed': 0,
            'failed': 0,
            'avg_cloud_coverage': []
        }
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"\nProcessing {idx}/{len(image_files)}: {img_path.name}")
            
            try:
                results = self.process_single_image(img_path, save_all_methods)
                stats['processed'] += 1
                stats['avg_cloud_coverage'].append(results['cloud_percentage'])
                
                if create_comparisons:
                    self.create_comparison_image(results, img_path.name)
                
                print(f"  ✓ Successfully processed (Cloud coverage: {results['cloud_percentage']:.1f}%)")
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                stats['failed'] += 1
                continue
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total images: {stats['total']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        if stats['avg_cloud_coverage']:
            print(f"Average cloud coverage: {np.mean(stats['avg_cloud_coverage']):.1f}%")
        print(f"\nInpainted images saved to: {self.output_folder}")
        
        # Save processing report
        self.save_report(stats)
    
    def save_report(self, stats):
        """Save processing report"""
        report_path = self.output_folder / f"inpainting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("Satellite Image Inpainting Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Folder: {self.input_folder}\n")
            f.write(f"Output Folder: {self.output_folder}\n\n")
            f.write(f"Total Images: {stats['total']}\n")
            f.write(f"Successfully Processed: {stats['processed']}\n")
            f.write(f"Failed: {stats['failed']}\n")
            if stats['avg_cloud_coverage']:
                f.write(f"Average Cloud Coverage: {np.mean(stats['avg_cloud_coverage']):.1f}%\n")
            f.write("\nMethods Used:\n")
            f.write("- Telea (Fast Marching Method)\n")
            f.write("- Navier-Stokes\n")
            f.write("- Statistical (Median + Gaussian)\n")
            f.write("- Hybrid (Best for satellite imagery)\n")
        
        print(f"Report saved to: {report_path}")

def main():
    """Main function"""
    print("=" * 60)
    print("SATELLITE IMAGE INPAINTING SYSTEM")
    print("Intelligent Cloud/Missing Data Removal")
    print("=" * 60)
    
    # Default to light_cloud folder or ask for input
    default_folder = "light_cloud"
    
    if os.path.exists(default_folder):
        use_default = input(f"\nFound '{default_folder}' folder. Use it? (y/n): ").strip().lower()
        if use_default == 'y':
            input_folder = default_folder
        else:
            input_folder = input("Enter folder path containing images: ").strip()
    else:
        input_folder = input("Enter folder path containing images: ").strip()
    
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' not found!")
        return
    
    # Options
    print("\nProcessing Options:")
    print("1. Quick (save only best method - Hybrid)")
    print("2. Comprehensive (save all methods + comparisons)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    save_all = (choice == '2')
    create_comp = (choice == '2')
    
    # Create inpainter and process
    inpainter = SatelliteImageInpainter(input_folder)
    inpainter.process_folder(save_all_methods=save_all, create_comparisons=create_comp)

if __name__ == "__main__":
    main()