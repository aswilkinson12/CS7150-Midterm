"""
    Data preprocessing package:
    - download raw satellite files
    - filter HAB months
    - select light-cloud images
    - perform inpainting
"""

from data_pipeline.fix_light_cloud import SatelliteImageInpainter