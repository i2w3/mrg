import os
import json

from collections import defaultdict
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.load(open(self.ann_path))
        self.examples = self.ann[self.split]
        self.masks = []
        self.reports = []

        for i in range(len(self.examples)):
            self.reports.append(tokenizer(self.examples[i]['En_Report'])[:self.max_seq_length])
            self.masks.append([1] * len(self.reports[-1]))

    def __len__(self):
        return len(self.examples)

class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        image_id = self.examples[idx]['id']
        image_path = eval(str(self.examples[idx]['Image_path']))
        images = []
        for ind in range(len(image_path)):
            image = Image.open(os.path.join(self.image_dir, image_path[ind])).convert('RGB')
            if self.transform is not None:
                images.append(self.transform(image))
        images = torch.stack(images, 0)
        report_ids = self.reports[idx]
        report_masks = self.masks[idx]
        seq_length = len(report_ids)
        sample = (image_id, images, report_ids, report_masks, seq_length)
        return sample


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        super().__init__()
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # 1. 获取每个样本的长度
        # 注意：这里我们直接访问 dataset.examples，如果你的结构不同，需要相应修改
        # 这一步可能会稍微增加初始化的时间，但只执行一次
        self.lengths = [len(eval(str(ex['Image_path']))) for ex in dataset.examples]
        
        # 2. 创建桶，将样本索引按长度分组
        self.buckets = defaultdict(list)
        for i, length in enumerate(self.lengths):
            self.buckets[length].append(i)
            
        # 3. 将桶（即样本索引列表）本身作为一个可迭代对象
        self.bucket_iters = []
        for bucket in self.buckets.values():
            # 将每个桶内的样本随机打乱
            random.shuffle(bucket)
            # 切分成小批次
            num_batches = len(bucket) // self.batch_size
            for i in range(num_batches):
                self.bucket_iters.append(bucket[i * self.batch_size : (i + 1) * self.batch_size])
            # 处理剩余不足一个batch的样本
            if not self.drop_last and len(bucket) % self.batch_size > 0:
                self.bucket_iters.append(bucket[num_batches * self.batch_size :])

    def __iter__(self):
        random.shuffle(self.bucket_iters)
        for batch in self.bucket_iters:
            yield batch

    def __len__(self):
        return len(self.bucket_iters)


class FFAIRDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        ## load the FFAIRDataset
        self.dataset = FFAIRDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        if self.split == 'train' and self.shuffle:
            batch_sampler = BucketBatchSampler(self.dataset, self.batch_size, drop_last=True)
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_sampler': batch_sampler,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'pin_memory': True,
            }
        else:
            self.init_kwargs = {
                'dataset': self.dataset,
                'batch_size': self.batch_size,
                'shuffle': False,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'pin_memory': True,
            }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        # we should pad the min
        max_images = max([x.shape[0] for x in images])
        images = [torch.cat((x, torch.zeros([max_images-x.shape[0], 3, 224, 224])), dim=0) for x in images]
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)