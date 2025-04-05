# early_stopping.py
import utils
import numpy as np
import os
import torch
import logging

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, checkpoint_name='', val_loss=np.inf):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = val_loss
        self.early_stop = False
        self.val_loss_min = val_loss
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.checkpoint_name = checkpoint_name

    def __call__(self, val_loss, model, epoch, current_step):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, current_step)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, current_step)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, epoch, current_step):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        checkpoint_filename = f'{self.checkpoint_name}--epoch-{epoch}.pth'
        checkpoint_filename = os.path.join(self.path, checkpoint_filename)
        logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
        checkpoint = {
            'network': model[0].state_dict(),
            'optimizer': model[1].state_dict(),
            'scheduler': model[2].state_dict(),
            'val_loss': val_loss,
            'epoch': epoch,
            'current_step': current_step
        }
        torch.save(checkpoint, checkpoint_filename)
        logging.info('Saving checkpoint done.')
        self.val_loss_min = val_loss