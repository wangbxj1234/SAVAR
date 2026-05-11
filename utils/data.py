import os.path as osp
import os
import PIL.Image as PImage
from pathlib import Path
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
from tokenizer import tokenize
import torch
# 尝试导入datasets库（用于读取arrow文件）
try:
    from datasets import Dataset as HFDataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# For CLIP
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img: PImage.Image = PImage.open(f).convert('RGB')
        return img
    except Exception as e:
        # 如果图片损坏，返回一个默认的黑色图片（512x512 RGB）
        print(f'[WARNING] Failed to load image {path}: {e}. Using default black image.')
        return PImage.new('RGB', (512, 512), (0, 0, 0))


def image_transform(final_reso: int, model='train',
    hflip=False, mid_reso=1.125,):
    """
    关键改进：直接resize到目标分辨率（不crop），保留全图信息
    这样与CLIP的预处理完全对齐：都是全图resize，不丢失边缘
    mid_reso参数保留用于兼容性，但实际不再使用
    """
    # 直接resize到目标分辨率，不crop
    train_aug, val_aug = [
        transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    if model == 'train':
       return train_aug
    else:
        return val_aug


def read_path_caption(file_txt):
    image_captions_list = []
    with open(file_txt, "r") as file:
        for line in file:
            line = line.strip()
            if line and ':' in line:  # Skip empty lines and lines without ':'
                parts = line.split(':')
                if len(parts) == 2:
                    a, b = parts
                    image_captions_list.append((a.strip(), b.strip()))
    return image_captions_list


# 文件查找缓存（避免重复搜索）
_file_cache = {}

def find_image_file(data_dir, filename):
    """
    在数据目录中智能查找图片文件
    支持多种可能的路径结构，使用缓存提高性能
    """
    # 使用缓存
    cache_key = (str(data_dir), filename)
    if cache_key in _file_cache:
        return _file_cache[cache_key]
    
    data_path = Path(data_dir)
    basename = Path(filename).name
    basename_lower = basename.lower()
    
    # 尝试多种路径组合 (优先尝试最可能的)
    # 1. 直接按caption中的相对路径查找
    rel_path = filename
    if rel_path.startswith('./'):
        rel_path = rel_path[2:]
    
    # 2. 判断是train还是val
    split_name = 'train' if '/train/' in filename or filename.startswith('train/') else 'val'
    
    # 3. 构造可能的路径列表
    possible_paths = [
        data_path / rel_path,                    # 原始相对路径
        data_path / split_name / basename,       # Flat 结构 (如 val/ILSVRC2012_val_00000293.JPEG)
        data_path / basename,                    # 直接在 root 下
    ]
    
    # 4. 尝试标准 ImageNet 结构 (train/class/file)
    if '_' in basename:
        possible_class = basename.split('_')[0]
        possible_paths.extend([
            data_path / split_name / possible_class / basename,
            data_path / 'train' / possible_class / basename,
            data_path / 'val' / possible_class / basename,
        ])
    
    # 5. 尝试从相对路径中提取 class (如 val/n01440764/xxx.JPEG -> class=n01440764)
    parts = Path(rel_path).parts
    if len(parts) >= 3: # split/class/file
        possible_paths.append(data_path / parts[-3] / parts[-2] / basename)
    elif len(parts) >= 2: # class/file or split/file
        possible_paths.append(data_path / parts[-2] / basename)

    # 遍历尝试
    for p in possible_paths:
        if p.exists():
            _file_cache[cache_key] = str(p)
            return str(p)
        # 尝试不同扩展名
        if p.suffix.lower() in ['.jpeg', '.jpg']:
            for ext in ['.JPEG', '.jpg', '.jpeg', '.JPG']:
                p_ext = p.with_suffix(ext)
                if p_ext.exists():
                    _file_cache[cache_key] = str(p_ext)
                    return str(p_ext)

    # 最后手段：递归搜索 (仅在第一次找不到时构建全量缓存)
    search_key = str(data_path)
    if not hasattr(find_image_file, '_full_scan_done'):
        find_image_file._full_scan_done = set()
    
    if search_key not in find_image_file._full_scan_done:
        print(f'[INFO] Starting full directory scan of {data_path} to locate missing images...')
        find_image_file._full_scan_done.add(search_key)
        # 扫描整个目录并填充缓存
        for root, dirs, files in os.walk(data_path):
            for f in files:
                if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                    # 我们可以为这个文件名建立一个通用的缓存
                    # 但由于 cache_key 是 (data_dir, filename)，我们需要更通用的缓存
                    _file_cache[(str(data_dir), f)] = str(Path(root) / f)
        
        # 重新检查缓存
        if cache_key in _file_cache:
            return _file_cache[cache_key]
        # 检查 basename
        if (str(data_dir), basename) in _file_cache:
            return _file_cache[(str(data_dir), basename)]

    # 如果都找不到，返回原始预期路径
    result = str(data_path / rel_path)
    _file_cache[cache_key] = result
    return result


# 读取数据
class ImageNet(Dataset):
    '''
    Args:
        root: image data root (real ImageNet dataset path, can be arrow files directory)
        caption_root: caption file root (VAR_dec/imagenet path)
        split: data split
    
        Returns:
            pic(PIL.Image.Image)   # return RGB
    '''

    def __init__(
            self,
            root: str,
            final_reso: int, model='train',
    hflip=False, mid_reso=1.125,
    caption_root: str = None,
    ) -> None:
        super(ImageNet, self).__init__()
        assert model in ["train", "val"]
        self.img_transform = image_transform(final_reso, model,
    hflip, mid_reso)
        # Image data directory (real ImageNet path)
        self.data_dir = Path(root)
        
        # Caption file directory (VAR_dec/imagenet path)
        if caption_root is None:
            # Default: use VAR_dec/imagenet as caption source
            var_dec_root = Path(__file__).parent.parent  # Go up from utils/ to VAR_dec/
            caption_root = str(var_dec_root / "imagenet")
        self.caption_dir = Path(caption_root)
        self.caption_file = str(self.caption_dir / f"{model}" / "image_captions.txt")
        self.reader = read_path_caption(self.caption_file)
        
        # 检查是否是arrow文件格式
        self.use_arrow = False
        self.arrow_dataset = None
        
        if HAS_DATASETS:
            # 检查数据目录中是否有arrow文件
            arrow_files = list(self.data_dir.glob(f'imagenet-1k-{model}-*.arrow'))
            if not arrow_files:
                # 也尝试validation作为val
                if model == 'val':
                    arrow_files = list(self.data_dir.glob('imagenet-1k-validation-*.arrow'))
            
            if arrow_files:
                print(f'[ImageNet] 检测到arrow文件格式，使用arrow文件加载图片数据')
                print(f'  找到 {len(arrow_files)} 个{model} arrow文件')
                self.use_arrow = True
                
                # 加载所有arrow文件
                try:
                    # 使用datasets库加载所有arrow文件
                    arrow_file_paths = [str(f) for f in sorted(arrow_files)]
                    
                    if len(arrow_file_paths) == 1:
                        self.arrow_dataset = HFDataset.from_file(arrow_file_paths[0])
                    else:
                        # 如果有多个文件，需要合并
                        print(f'  正在合并 {len(arrow_file_paths)} 个arrow文件...')
                        datasets_list = []
                        for i, arrow_file in enumerate(arrow_file_paths):
                            if i % 50 == 0:
                                print(f'    已加载 {i}/{len(arrow_file_paths)} 个文件...')
                            datasets_list.append(HFDataset.from_file(arrow_file))
                        from datasets import concatenate_datasets
                        self.arrow_dataset = concatenate_datasets(datasets_list)
                        print(f'  ✅ 合并完成!')
                    
                    print(f'  ✅ 成功加载arrow数据集，共 {len(self.arrow_dataset)} 个样本')
                    print(f'  特征: {list(self.arrow_dataset.features.keys())}')
                    
                    # 确保arrow数据集和caption文件的数量匹配（允许一些差异）
                    if abs(len(self.arrow_dataset) - len(self.reader)) > len(self.reader) * 0.1:
                        print(f'  ⚠️  警告: arrow数据集样本数({len(self.arrow_dataset)})与caption文件样本数({len(self.reader)})差异较大')
                    else:
                        print(f'  ✅ arrow数据集({len(self.arrow_dataset)})与caption文件({len(self.reader)})样本数匹配')
                        
                except Exception as e:
                    print(f'  ❌ 加载arrow文件失败: {e}')
                    print(f'  将回退到标准文件路径方式')
                    self.use_arrow = False
                    self.arrow_dataset = None

    def __len__(self):
        if self.use_arrow and self.arrow_dataset is not None:
            # 使用较小的长度，确保索引不会越界
            return min(len(self.arrow_dataset), len(self.reader))
        return len(self.reader)

    def __getitem__(self, indices):
        try:
            # 1. 读取路径和Caption
            img_path, captions = self.reader[indices]
            
            # 2. 获取图片 (处理 Arrow 或普通文件)
            if self.use_arrow and self.arrow_dataset is not None:
                arrow_sample = self.arrow_dataset[indices]
                img = arrow_sample['image']
                if not isinstance(img, PImage.Image):
                    # 如果是 numpy 数组，转为 PIL
                    img = PImage.fromarray(img) if hasattr(img, '__array__') else img
            else:
                # 普通文件读取逻辑
                if img_path.startswith("./"): img_path = img_path[2:]
                full_img_path = find_image_file(self.data_dir, img_path)
                
                # 这里的 pil_loader 内部虽然有 convert('RGB')，但为了统一逻辑，下面会再检查一次
                img = pil_loader(full_img_path)
            
            # === 修复核心：强制转换为 RGB ===
            # ImageNet 包含少量灰度图，Arrow读取时如果没转RGB，ToTensor后就是[1, H, W]
            # 这会导致和彩色图 [3, H, W] 无法 stack
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 3. 转换图片和文本
            img1 = self.img_transform(img)
            caption = tokenize(captions)
            
            # === 之前的内存修复 ===
            # 强制断开内存引用 (Deep Copy)
            if isinstance(img1, torch.Tensor):
                img1 = img1.contiguous().clone()
            
            if isinstance(caption, torch.Tensor):
                caption = caption.contiguous().clone()

            return img1, caption
        except Exception as e:
            # 如果加载失败，返回默认的黑色图片，避免训练中断
            print(f'[ERROR] Failed to load image at index {indices}: {e}. Using default black image.')
            default_img = PImage.new('RGB', (512, 512), (0, 0, 0))
            img1 = self.img_transform(default_img)
            caption = tokenize("")
            return img1, caption




  
def imagenet(root, final_reso, model, hflip, mid_reso):
    return ImageNet(root, final_reso, model, hflip, mid_reso)
