from PIL import Image
from pathlib import Path


def crop_lake_okeechobee(input_folder='data_imgs/habs_month_images', output_folder='raw'):

    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Find all .tif images in the input folder
    tif_files = list(Path(input_folder).glob('*.tif'))
    if not tif_files:
        print(f"No .tif files found in {input_folder}")
        return

    print(f"Found {len(tif_files)} .tif file(s)")


    left, top = 1485, 1105
    right, bottom = left + 200, top + 200

    for tif_file in tif_files:
        try:
            img = Image.open(tif_file)
            width, height = img.size
            print(f"\nProcessing: {tif_file.name} ({width}x{height})")

            # Crop and save
            cropped = img.crop((left, top, right, bottom))
            out_path = Path(output_folder) / f"{tif_file.stem}.tif"
            cropped.save(out_path)

            print(f"Saved: {out_path} ({cropped.size[0]}x{cropped.size[1]})")

        except Exception as e:
            print(f"Error processing {tif_file.name}: {e}")

    print(f"\n Done! Cropped Lake Okeechobee images saved to '{output_folder}'")


if __name__ == "__main__":
    crop_lake_okeechobee()
