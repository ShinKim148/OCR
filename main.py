import os 
import sys
import torch  
from torch.utils.data import random_split
from argparse import ArgumentParser
from src.utils import AverageMeter, Eval, OCRLabelConverter
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import tqdm
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.trainer import OCRTrainer
from src.model import CRNN
from src.dataset import SynthDataset, SynthCollator
from src.utils import AverageMeter, Eval, OCRLabelConverter
from src.loss import CustomCTCLoss


import nsml
from nsml import DATASET_PATH


##############################################################################

def bind_model(model, class_to_save, optimizer=None):
    print('@@skc bind_model() +++')
    def load(filename, **kwargs):
        print('@@skc bind_model() - load filename=', filename)

    def save(filename, **kwargs):
        print('@@skc bind_model() - save filename=', filename)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))
        # with open(os.path.join(filename, 'class.pkl'), 'wb') as fp:
        #     pickle.dump(class_to_save, fp)

    def infer(input, top_k=100):
        print('@@skc bind_model() ++ TODO: impl')
        print('@@skc bind_model() - infer top_k=', top_k)
        print('@@skc bind_model() - infer type(input)=', type(top_k))
        print('@@skc bind_model() - infer input=', top_k)

    print('@@skc bind_model() call nsml.bind()')
    nsml.bind(save=save, load=load, infer=infer)

############################################################################




class Learner(object):
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.savepath = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available()
        self.cuda_count = torch.cuda.device_count()
        if self.cuda:
            self.model = self.model.cuda() 
        self.epoch = 0
        #if self.cuda_count > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    self.model = nn.DataParallel(self.model)
        self.best_score = None
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint['epoch']
            self.best_score=self.checkpoint['best']
            self.load()
        else:
            print('checkpoint does not exist')

    def fit(self, opt):
        opt['cuda'] = self.cuda
        opt['model'] = self.model
        opt['optimizer'] = self.optimizer
        #logging.basicConfig(filename="%s/%s.csv" %(opt['log_dir'], opt['name']), level=logging.INFO)
        #self.saver = EarlyStopping(self.savepath, patience=15, verbose=True, best_score=self.best_score) @@@@@@@@@@@@@@@@@@@@@@@
        opt['epoch'] = self.epoch
        trainer = OCRTrainer(opt)
        

        for epoch in range(opt['epoch'], opt['epochs']):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            trainer.count = epoch
            info = '%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f'%(epoch, train_result['train_loss'], 
                val_result['val_loss'], train_result['train_ca'],  val_result['val_ca'],
                train_result['train_wa'], val_result['val_wa'])
            #logging.info(info)
            self.val_loss = val_result['val_loss']
            print(self.val_loss)
            #if self.savepath:
            #    self.save(epoch)
            #if self.saver.early_stop:
            #    print("Early stopping")
            #    break

    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(self.savepath, self.checkpoint['epoch']))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])

    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)

# alphabet_list
math_list = []
with open('math.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    math_list.append(lines)
    math_list = sum(math_list, [])

math_list = ''.join(math_list)
alphabet = ''.join(set(math_list))




if __name__ == "__main__":

    args = {
        'name':'exp1',
        'path':'./data', #local path
        #'path':DATASET_PATH, #nsml path
        'imgdir': 'train/train_data',
        'imgH':32,
        'nChannels':1,
        'nHidden':256,
        'nClasses':len(alphabet),
        'lr':0.0001,        'epochs':2,
        'batch_size':8,
        'save_dir':'../checkpoints/',
        'log_dir':'../logs',
        'resume':False,
        'cuda': True,
        'schedule':False
        
    }

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('data_path:' + DATASET_PATH) #/data/CUBOX_test_ocr_small
    print('torch version:' + torch.__version__)#torch version:1.10.0+cu102
    #print(os.system('pip list'))
    print(os.system('nvidia-smi'))
    print(os.system('nvcc --version'))
    print(args)

    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    data = SynthDataset(args)
    args['collate_fn'] = SynthCollator()
    train_split = int(0.8*len(data))
    val_split = len(data) - train_split
    args['data_train'], args['data_val'] = random_split(data, (train_split, val_split))
    print('Traininig Data Size:{}\nVal Data Size:{}'.format(
        len(args['data_train']), len(args['data_val'])))
    args['alphabet'] = alphabet
    model = CRNN(args)
    args['criterion'] = CustomCTCLoss()
    savepath = os.path.join(args['save_dir'], args['name'])
    #gmkdir(savepath)
    #gmkdir(args['log_dir'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    learner = Learner(model, optimizer, savepath=savepath, resume=args['resume'])

    bind_model(model, None, optimizer=optimizer)
    # test mode
    
#    if args.pause:
#        print('@@skc args.pause is True!!!')
#        nsml.paused(scope=locals()) #@@skc called from nsml submit?
    
    nsml.report(
        summary=True,
        #epoch=learner.epoch,
        epoch_total=args['epochs'],

    )
    
    learner.fit(args)