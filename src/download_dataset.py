import kagglehub
import shutil
import os


def move_dataset(source_path, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_dir)
    elif os.path.isdir(source_path):
        for item in os.listdir(source_path):
            src = os.path.join(source_path, item)
            dst = os.path.join(destination_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    else:
        raise FileNotFoundError(f"Source path {source_path} does not exist")


if __name__ == "__main__":
    path = kagglehub.dataset_download("uciml/adult-census-income")
    print("Path to dataset files:", path)

    target_directory = "../data/download"

    try:
        move_dataset(path, target_directory)
        print(f"Dataset successfully moved to: {target_directory}")
    except Exception as e:
        print(f"Error moving dataset: {e}")