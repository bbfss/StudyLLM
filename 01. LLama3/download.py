import os
import shutil
from modelscope import snapshot_download

# 1. 配置基础信息
model_id = 'LLM-Research/Meta-Llama-3-8B-Instruct'
base_dir = '/root/autodl-tmp'
cache_dir = os.path.join(base_dir, 'llama3_temp')

# 2. 指定只下载 original 文件夹下的这三个文件
allow_files = [
    'original/consolidated.00.pth',
    'original/params.json',
    'original/tokenizer.model'
]

print("🚀 正在从 ModelScope 精准下载 Llama-3 权重文件...")

try:
    # 3. 执行下载 (使用正确的参数名: allow_file_pattern)
    download_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_file_pattern=allow_files
    )
    
    # 4. 自动整理文件：将文件从深层目录移动到 /root/autodl-tmp 根目录
    source_folder = os.path.join(download_path, 'original')
    print(f"✅ 下载完成，正在整理文件至 {base_dir}...")

    for file_name in os.listdir(source_folder):
        src_file = os.path.join(source_folder, file_name)
        dst_file = os.path.join(base_dir, file_name)
        
        # 移动文件 (如果目标已存在则覆盖)
        shutil.move(src_file, dst_file)
        print(f"📦 已就绪: {file_name}")

    # 5. 清理下载产生的临时缓存文件夹
    shutil.rmtree(cache_dir)
    print("\n✨ 恭喜！所有文件已下载并整理完毕。")

except Exception as e:
    print(f"\n❌ 运行出错: {e}")