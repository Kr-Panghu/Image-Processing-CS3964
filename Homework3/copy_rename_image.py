import os
import shutil

# 源文件夹列表
source_folders = ["dataset0", "dataset1", "dataset2", "dataset3", "dataset4"]

# 目标文件夹
target_folder = "water_glass"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 计数器
counter = 1

# 遍历源文件夹
for source_folder in source_folders:
    # 构建完整路径
    source_folder_path = os.path.join(os.getcwd(), source_folder)

    # 遍历文件夹中的.jpg文件
    for filename in os.listdir(source_folder_path):
        if filename.endswith(".jpg"):
            # 构建源文件路径和目标文件路径
            source_file_path = os.path.join(source_folder_path, filename)
            target_file_path = os.path.join(target_folder, f"{counter}.jpg")

            # 复制并重命名文件
            shutil.copyfile(source_file_path, target_file_path)

            # 更新计数器
            counter += 1

print(f"共复制了 {counter - 1} 张图片到 {target_folder} 文件夹。")
