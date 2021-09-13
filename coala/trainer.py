'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import time

import sys
import torch
from torch import nn
from transformers import get_constant_schedule

from . import evaluation
from . import utils


class AS2Trainer():

    def __init__(
            self,
            model,
            optimizer = None,
            scheduler = None,
            epochs    = 3,
            patience  = 2,
            loss_fct  = nn.CrossEntropyLoss(),
            val_metric= 'P@1', #p@1, roc_auc, val_loss
            debug     = False,
            save_path = None,
            device    = 'cpu',
    ):
        assert val_metric in ['loss','val_loss','roc_auc','P@1','MAP','MRR','HIT@5','AUPC'], \
            'Evaluation Metric not recognized: %s' % val_metric
        self.logger = utils.get_logger(debug)

        self.model     = model
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.scheduler = scheduler or get_constant_schedule(self.optimizer)
        self.epochs    = epochs
        self.patience  = patience
        self.loss_fct  = loss_fct
        self.val_metric= val_metric
        self.debug     = debug
        self.save_path = save_path
        self.device    = device

        
    def fit(self, dataloader, dataloader_va, dataloaders_eval=[]):

        
        epochs_without_improvement = 0
        best_metrics = None

        self.logger.info('Start training with %d examples' % (len(dataloader.dataset)))
        self.logger.info('Configuration:')
        self.logger.info('    batch size: %d' % dataloader.batch_size)
        self.logger.info('    max epochs: %d' % self.epochs)
        self.logger.info('    val metric: %s' % self.val_metric)
        if self.save_path:
            self.logger.info('The trained model will be stored in: \'%s\'' % (self.save_path))
        if dataloader_va:
            self.logger.info('Validation data found (%d examples)' % len(dataloader_va.dataset))
            self.logger.info('The training will end when the %s computed on validation data worsens for %d consecutive epochs' % (self.val_metric, self.patience))
        if dataloaders_eval:
            self.logger.info('Evaluation data found (%d examples)' % sum(len(dd.dataset) for dd in dataloaders_eval))
        if self.debug:
            self.logger.debug('THE TRAINER IS RUNNING IN DEBUG MODE!')

            
        for epoch in range(1, self.epochs+1):
            time_epoch = time.time()
            time_all = time.time()
            loss_tr = .0
            y_true, y_scores = [], []
            
            self.model.train()
            for ib, batch in enumerate(dataloader):

                examples, labels = batch
                #self.logger.debug('Input and batch size:')
                self.logger.debug('  ' + ', '.join('%s:%s' % (k, examples[k].size()) for k in examples.keys()))
                
                labels = labels.to(self.device)
                examples = {k:v.to(self.device) for k,v in examples.items()}

                self.optimizer.zero_grad()
                outputs = self.model(**examples)

                logits  = outputs[0]
                loss    = self.loss_fct(logits, labels) 
                loss.backward()
                loss_tr += loss.item()

                self.optimizer.step()
                self.scheduler.step()

                y_true.extend(labels)
                y_scores.extend(logits[:,1].tolist())

                if self.debug and ib >2:
                    break
                
            time_epoch = time.time() - time_epoch
            loss_tr /= len(dataloader)
            
            time_eval = time.time()

            results = []
            for dloader in [dataloader_va] + dataloaders_eval:
                y_scores, loss_eval = self.predict(dloader, return_loss=True)
                y_true = dloader.dataset.labels[:len(y_scores)]
                questions_eval = dloader.dataset.questions[:len(y_scores)]
                ds_results = evaluation.report(y_true, y_scores, questions=questions_eval, return_metadata=False)
                ds_results['loss'] = loss_eval
                results.append(ds_results)
            time_eval = time.time() - time_eval
                
            if 'loss' in self.val_metric:
                op = lambda x,y: x > y
            else:
                op = lambda x,y: x < y

            epochs_without_improvement += 1
            is_saved = ''
            if not best_metrics or op(best_metrics[self.val_metric], results[0][self.val_metric]):
                best_metrics = results[0]
                epochs_without_improvement = 0
                if self.save_path:
                    #self.model.save(self.save_path)
                    if hasattr(self.model, 'module'):
                        self.model.module.save(self.save_path)
                    else:
                        self.model.save(self.save_path)
                    is_saved = ' (saved)'

            time_all = time.time() - time_all
            self.logger.info('Epoch %d - %s: %.4f%s' % (epoch, self.val_metric, results[0][self.val_metric], is_saved ))
            self.logger.info('    Execution time: %.1fs, of which %.1fs training and %.1fs evaluation' % (time_all, time_epoch, time_eval))
            self.logger.info('    Evaluation report:')
            self.logger.info('        training set - loss: %.4f, ' % loss_tr)
            for report_id, report in enumerate(results):
                report_str = ', '.join('%s:%d' % (k,v) if isinstance(v, int) else '%s:%.4f' % (k,v) for k,v in report.items())
                eval_set_name = ' ' * 8 + ('validation set - ' if not report_id else ('eval set [#%d] - ' % report_id))
                self.logger.info(eval_set_name + report_str)

            if epochs_without_improvement >= self.patience :
                self.logger.info ('Early stopping')
                sys.stdout.flush()
                break
            if self.debug and epoch > 3:
                self.logger.debug('Debug completed')
        return self

    
    
    def predict(self, dataloader_te, return_loss=False):
        self.model.eval()
        y_true, y_score = [], []
        loss_va = 0.
        
        with torch.no_grad():
            for Sb, Yb in dataloader_te: 
                Sb = {k:v.to(self.device) for k,v in Sb.items()}
                Yb = Yb.to(self.device)
                output = self.model(**Sb)
                logits = output[0]

                loss = self.loss_fct(logits, Yb) 
                loss_va += loss.item()

                y_true.extend(Yb.tolist())
                y_score.extend(logits[:,1].tolist())

                if self.debug:
                    break
                    
        loss_va /= len(dataloader_te)

        return y_score if not return_loss else (y_score, loss_va)
        
