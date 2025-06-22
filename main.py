import argparse

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from modules import Tokenizer, FFAIRDataLoader, JointModel, compute_scores, Trainer


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/home/moment/PostGraduationProject/dataset/100/FFA-IR_images', help='the path to the images.')
    parser.add_argument('--ann_path', type=str, default='/home/moment/PostGraduationProject/dataset/annotation_mean_1.0_75.json', help='the path to the json file.')

    # Dataset settings
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='the number of samples for a batch')
    parser.add_argument('--max_seq_length', type=int, default=90, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/fair', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='CIDER', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--test', type=bool, default=False, help='whether to test the model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = FFAIRDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = FFAIRDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = FFAIRDataLoader(args, tokenizer, split='test', shuffle=False)

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointModel(tokenizer.get_vocab_size()).to(device)

    metrics = compute_scores
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = None

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, scheduler, args, train_dataloader, val_dataloader, test_dataloader)
    if args.test == False:
        trainer.train()
    else:
        trainer.test()
    