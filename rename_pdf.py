import os

def replace_season(filename):
    seasons = {
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四"
    }
    for num, char in seasons.items():
        if f"第{num}季度" in filename:
            filename = filename.replace(f"第{num}季度", f"第{char}季度")
    return filename

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".PDF") or "第" in filename and "季度" in filename:
            new_name = filename.replace('.PDF', '.pdf')
            new_name = replace_season(new_name)
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Renamed '{filename}' to '{new_name}'")

if __name__ == "__main__":
    directory = '/path/to/your/directory'  # Replace with the path to your directory
    rename_files(directory)
