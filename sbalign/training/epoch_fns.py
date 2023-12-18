import traceback
import torch
import numpy as np

from sbalign.utils.ops import to_numpy
from sbalign.utils.sb_utils import get_t_schedule
from sbalign.utils.sampling import sampling
from sbalign.utils.definitions import DEVICE


class ProgressMonitor:

    def __init__(self, metric_names=None):
        if metric_names is not None:
            self.metric_names = metric_names
            self.metrics = {metric: 0.0 for metric in self.metric_names}
        self.count = 0

    def add(self, metric_dict, batch_size: int = None):
        if not hasattr(self, 'metric_names'):
            self.metric_names = list(metric_dict.keys())
            self.metrics = {metric: 0.0 for metric in self.metric_names}

        self.count += (1 if batch_size is None else batch_size)

        for metric_name, metric_value in metric_dict.items():
            if metric_name not in metric_dict:
                self.metrics[metric_name] = 0.0
                self.metric_names.append(metric_name)
            
            self.metrics[metric_name] += metric_value * (1 if batch_size is None else batch_size)

    def summarize(self):
        return {k: np.round(v / self.count, 4) for k, v in self.metrics.items()}


def train_epoch_sbalign(
        model, loader, 
        optimizer, loss_fn,
        grad_clip_value: float = None, 
        ema_weights=None):

    model.train()
    monitor = ProgressMonitor()

    for idx, data in enumerate(loader):
        optimizer.zero_grad()

        try:
            data = data.to(DEVICE)
            drift_x, doobs_score_x, doobs_score_x_T = model(data)

            loss, loss_dict = loss_fn(drift_x_pred=drift_x,
                                      doobs_score_x_pred=doobs_score_x,
                                      doobs_score_xT_pred=doobs_score_x_T,
                                      data=data)
                                    
            monitor.add(loss_dict)

            loss.backward()

            if grad_clip_value is not None:
                grad_clip_value = 10.0
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            
            if ema_weights is not None:
                ema_weights.update(model.parameters())
            
        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                traceback.print_exc()
                continue

    return monitor.summarize()


def test_epoch_sbalign(model, loader, loss_fn):
    model.eval()
    monitor = ProgressMonitor()

    for idx, data in enumerate(loader):
        try:
            with torch.no_grad():
                data = data.to(DEVICE)
                drift_x, doobs_score_x, doobs_score_x_T = model(data)
                
                _, loss_dict = loss_fn(drift_x_pred=drift_x,
                                       doobs_score_x_pred=doobs_score_x,
                                       doobs_score_xT_pred=doobs_score_x_T,
                                       data=data)
                
                monitor.add(loss_dict)

        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e)
                continue

    return monitor.summarize()


# -------------------- Inference ------------------------

def inference_epoch_sbalign(model, g, dataset, inference_steps: int = 100, t_max: float =1.0):
    t_schedule = torch.from_numpy(get_t_schedule(inference_steps, t_max=t_max)).float().to(DEVICE)

    pos_T_preds = []
    pos_T = to_numpy(dataset['final'])

    n_samples = len(dataset['initial'])

    input_data = dataset['initial'].to(DEVICE)

    for idx in range(n_samples):
        pos_0 = input_data[idx: idx+1]
        assert pos_0.shape[0] == 1

        pos_T_pred = sampling(pos_0, model, g, inference_steps, t_schedule)
        pos_T_preds.append(to_numpy(pos_T_pred))
    
    pos_T_preds = np.asarray(pos_T_preds).reshape(n_samples, -1)

    rmsd = np.sqrt( ((pos_T_preds - pos_T)**2).sum(axis=1).mean(axis=0) )
    return {'rmsd': np.round(rmsd, 4)}