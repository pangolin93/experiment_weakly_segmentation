import sys
import torch
from tqdm import tqdm as tqdm
import segmentation_models_pytorch as smp
from smp.utils.meter import AverageValueMeter


class Epoch:

    def __init__(self, model_strong, model_weak, loss_strong, loss_weak, metrics_strong, metrics_weak, stage_name, device='cpu', verbose=True, enable_weak=False):
        self.model_strong = model_strong
        self.model_weak = model_weak
        
        self.loss_strong = loss_strong
        self.loss_weak = loss_weak
        
        self.metrics_strong = metrics_strong
        self.metrics_weak = metrics_weak

        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self.enable_weak = enable_weak

        self._to_device()

    def _to_device(self):
        self.model_strong.to(self.device)
        self.model_weak.to(self.device)

        self.loss_strong.to(self.device)
        self.loss_weak.to(self.device)

        for metric in self.metrics_strong:
            metric.to(self.device)

        for metric in self.metrics_weak:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, y_weak):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, y_weak in iterator:
                x, y, y_weak = x.to(self.device), y.to(self.device), y_weak.to(self.device)

                loss_strong, loss_weak, y_pred_strong, y_pred_weak = self.batch_update(x, y, y_weak)

                # update loss logs
                loss_strong_values = loss_strong.cpu().detach().numpy()
                loss_meter.add(loss_strong_values)

                loss_logs = {self.loss_strong.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # TODO: track loss weak?!?

                # update metrics logs
                for metric_fn in self.metrics_strong:
                    metric_value = metric_fn(y_pred_strong, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                for metric_fn in self.metrics_weak:
                    metric_value = metric_fn(y_pred_weak, y_weak).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
