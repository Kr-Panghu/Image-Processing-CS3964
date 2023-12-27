import os

# 目标文件夹
target_folder = "water_glass"

# 获取目标文件夹中所有文件
all_files = os.listdir(target_folder)

# 对文件进行按名称排序
all_files.sort()

# 重命名文件
for index, file_name in enumerate(all_files):
    old_path = os.path.join(target_folder, file_name)
    new_name = f"{index + 1}.jpg"
    new_path = os.path.join(target_folder, new_name)

    # 重命名文件
    os.rename(old_path, new_path)
    print(f"已重命名文件: {old_path} 为 {new_path}")

print("重命名操作完成。")
