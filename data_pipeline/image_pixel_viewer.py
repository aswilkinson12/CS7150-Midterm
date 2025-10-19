import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

class TifPixelViewer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.im = None
        self.image_array = None
        self.colorbar = None
        self.pixel_info = None
        
    def load_image(self, filepath):
        """Load a TIF image file"""
        try:
            # Open the image using PIL
            img = Image.open(filepath)
            
            # Convert to numpy array
            self.image_array = np.array(img)
            
            # Get image info
            print(f"Image loaded: {os.path.basename(filepath)}")
            print(f"Image shape: {self.image_array.shape}")
            print(f"Data type: {self.image_array.dtype}")
            print(f"Min value: {self.image_array.min()}")
            print(f"Max value: {self.image_array.max()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def display_image(self):
        """Display the image with interactive pixel value display"""
        if self.image_array is None:
            print("No image loaded")
            return
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Display the image
        if len(self.image_array.shape) == 2:
            # Grayscale image
            self.im = self.ax.imshow(self.image_array, cmap='gray', interpolation='nearest')
            self.colorbar = plt.colorbar(self.im, ax=self.ax)
        else:
            # Color image (RGB or RGBA)
            self.im = self.ax.imshow(self.image_array, interpolation='nearest')
        
        # Set title
        self.ax.set_title('TIF Image Viewer - Hover to see pixel values')
        
        # Add pixel info text
        self.pixel_info = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Connect mouse motion event
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Add grid for better pixel location
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.show()
    
    def on_mouse_move(self, event):
        """Handle mouse movement to display pixel values"""
        if event.inaxes != self.ax:
            self.pixel_info.set_text('')
            self.fig.canvas.draw_idle()
            return
        
        # Get cursor position
        x, y = int(event.xdata), int(event.ydata)
        
        # Check if position is within image bounds
        if (0 <= x < self.image_array.shape[1] and 
            0 <= y < self.image_array.shape[0]):
            
            # Get pixel value
            if len(self.image_array.shape) == 2:
                # Grayscale
                pixel_value = self.image_array[y, x]
                info_text = f'Position: ({x}, {y})\nValue: {pixel_value}'
            elif len(self.image_array.shape) == 3:
                # Color image
                pixel_value = self.image_array[y, x]
                if self.image_array.shape[2] == 3:
                    info_text = f'Position: ({x}, {y})\nRGB: ({pixel_value[0]}, {pixel_value[1]}, {pixel_value[2]})'
                elif self.image_array.shape[2] == 4:
                    info_text = f'Position: ({x}, {y})\nRGBA: ({pixel_value[0]}, {pixel_value[1]}, {pixel_value[2]}, {pixel_value[3]})'
                else:
                    info_text = f'Position: ({x}, {y})\nValue: {pixel_value}'
            
            self.pixel_info.set_text(info_text)
        else:
            self.pixel_info.set_text('')
        
        self.fig.canvas.draw_idle()

def select_file():
    """Open file dialog to select a TIF file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select a TIF image",
        filetypes=[("TIF files", "*.tif *.tiff"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path

def main():
    """Main function to run the application"""
    print("=== TIF Image Pixel Value Viewer ===")
    print("\nOptions:")
    print("1. Select file using dialog")
    print("2. Enter file path manually")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        filepath = select_file()
        if not filepath:
            print("No file selected")
            return
    elif choice == '2':
        filepath = input("Enter TIF file path: ").strip()
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return
    else:
        print("Invalid choice")
        return
    
    # Create viewer instance
    viewer = TifPixelViewer()
    
    # Load and display image
    if viewer.load_image(filepath):
        print("\nDisplaying image... Hover over the image to see pixel values")
        viewer.display_image()
    else:
        print("Failed to load image")

# Alternative simple function for quick use
def quick_view(filepath):
    """Quick function to view a TIF file with pixel values"""
    viewer = TifPixelViewer()
    if viewer.load_image(filepath):
        viewer.display_image()
    else:
        print(f"Failed to load: {filepath}")

if __name__ == "__main__":
    main()