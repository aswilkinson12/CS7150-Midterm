from pathlib import Path
import shutil


def copy_images_from_list(list_path: Path, destination_dir: Path) -> None:
    """
    Copy image files listed in a text file into a destination folder.

    The text file is expected to contain lines where the second whitespace-
    separated token is the relative path to an image file (e.g. "date<TAB>path").
    """
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    base_dir = list_path.parent
    copied_count = 0
    missing_paths: list[str] = []

    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Skipping malformed line {line_number}: {raw_line.rstrip()}")
                continue

            image_relative_path = parts[1].strip()
            image_path = (base_dir / image_relative_path).expanduser()

            if not image_path.exists():
                missing_paths.append(image_relative_path)
                print(f"Missing source (line {line_number}): {image_path}")
                continue

            destination_path = destination_dir / image_path.name
            shutil.copy2(image_path, destination_path)
            copied_count += 1

    print(f"Copied {copied_count} image(s) to {destination_dir}")

    if missing_paths:
        print("The following source files were not found:")
        for missing in missing_paths:
            print(f" - {missing}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    copy_images_from_list(
        list_path=project_root / "jun_to_oct_files.txt",
        destination_dir=project_root / "habs_month_image",
    )
