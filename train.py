import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import hicomodel
import pl_bolts
import copy
from torch.utils.data import Dataset
import os
from test import ChromosomeDataset



def main():
    args = init_parser()
    init_training(args)

def init_parser():
  parser = argparse.ArgumentParser(description='C.Origami Training Module.')




  # Data and Run Directories
  parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--window', dest='window', default=128,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--length', dest='length', default=128,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--val_chr', dest='val_chr', default=1,
                        type=int,
                        help='Random seed for training')

  parser.add_argument('--itype', dest='itpe', default='Outward',
                        help='Path to the model checkpoint')
  
  
  parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')

  # Data directories
  parser.add_argument('--data-root', dest='dataset_data_root', default='/content/drive/MyDrive/corigamidata',
                        help='Root path of training data', required=True)


  # Model parameters
  parser.add_argument('--model-type', dest='model_type', default='ConvTransModel',
                        help='CNN with Transformer')

  # Training Parameters
  parser.add_argument('--patience', dest='trainer_patience', default=20,
                        type=int,
                        help='Epoches before early stopping')
  
  parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=50,
                        type=int,
                        help='Max epochs')
  
  parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=1,
                      
                        type=int,
                        help='Top n models to save')
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=2,
                        type=int,
                        help='Number of GPUs to use')

  # Dataloader Parameters
  parser.add_argument('--batch-size', dest='dataloader_batch_size', default=16, 
                        type=int,
                        help='Batch size')
  
  parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
  
  parser.add_argument('--num-workers', dest='dataloader_num_workers', default=8,
                        type=int,
                        help='Dataloader workers')


  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  return args

def init_training(args):

    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer_patience,
                                        verbose=False,
                                        mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                        save_top_k=args.trainer_save_top_n, 
                                        monitor='val_loss',filename=f'{args.val_chr}')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    #csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run_save_path}/csv')
    logger = pl.loggers.TensorBoardLogger(save_dir = f'{args.run_save_path}/tensorboard')
    all_loggers = logger
    
    # Assign seed
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args)
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger = all_loggers,
                            callbacks = [early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs
                            )
    val_chr,train_chr=pl_module.split_chromosomes(args.val_chr)
    print(val_chr,train_chr)
    trainloader = pl_module.get_dataloader(args, 'train',train_chr)
    print("trainloader ready")
    valloader = pl_module.get_dataloader(args, 'val',val_chr)
    print("valloader ready")
    pl_trainer.fit(pl_module, trainloader, valloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = self.get_model(args)
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        inputs, targets = batch
        inputs = inputs.transpose(1, 2).contiguous()
        inputs, targets = inputs.float().cuda(), targets.float().cuda()

        return inputs, targets
    
    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics



    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        return loss

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)
    

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss' : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args,chr):

        dataset=ChromosomeDataset(data_dir=args.dataset_data_root,window=args.window,length=args.length,chr=chr,itype=args.itpe)


        return dataset

    def get_dataloader(self, args, mode,chr):
        dataset = self.get_dataset(args,chr)

        if mode == 'train':
            shuffle = True
        else: # validation and test settings
            shuffle = False
        
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers

        if not args.dataloader_ddp_disabled:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus) 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,

            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True,
            drop_last=True,
        )
        return dataloader

    def get_model(self,args):
        model = hicomodel.ConvTransModel(True,args.window)
        return model
    
    def split_chromosomes(self,selected_chr):
        all_chromosomes = list(range(1, 17))
        return ([selected_chr], [chr for chr in all_chromosomes if chr != selected_chr])


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
