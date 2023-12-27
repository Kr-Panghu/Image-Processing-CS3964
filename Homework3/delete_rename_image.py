import os
import random
import shutil

# 目标文件夹
target_folder = "dataset"

# 获取目标文件夹中所有文件
all_files = os.listdir(target_folder)

# 从中随机选择 10 个文件
files_to_delete = random.sample(all_files, 5)

# 删除选定的文件
for file_to_delete in files_to_delete:
    file_path = os.path.join(target_folder, file_to_delete)
    os.remove(file_path)
    print(f"已删除文件: {file_path}")

all_files = os.listdir(target_folder)

# 重命名剩余的文件
for index, remaining_file in enumerate(all_files):
    old_path = os.path.join(target_folder, remaining_file)
    new_name = f"{index + 1}_tmp.jpg"
    new_path = os.path.join(target_folder, new_name)

    # 重命名文件
    os.rename(old_path, new_path)
    
all_files = os.listdir(target_folder)
    
for index, remaining_file in enumerate(all_files):
    old_path = os.path.join(target_folder, remaining_file)
    new_name = f"{index + 1}.jpg"
    new_path = os.path.join(target_folder, new_name)

    # 重命名文件
    os.rename(old_path, new_path)

print("操作完成。")
