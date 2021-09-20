import sys

import torch
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from tqdm import tqdm as tqdm


class SWEpoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        """[summary]

        Args:
            model ([type]): [description]
            loss (a PyTorch loss object or a an array of tuples like [(loss1, weight1), ...]): [description]
            metrics ([type]): [description]
            stage_name ([type]): [description]
            device (str, optional): [description]. Defaults to 'cpu'.
            verbose (bool, optional): [description]. Defaults to True.
        """
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if type(self.loss) == type([]):
            for l, w in self.loss:
                self.loss[i] = l.to(self.device)
        else:
            self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, epoch):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, w in iterator:
                x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
                loss, y_pred = self.batch_update(x, y, w, epoch)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs

class SWTrainEpoch(SWEpoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', 
                 verbose=True, weight_power=5):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.loss.reduction = 'none'
        self.weight_power = weight_power

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, w, epoch):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        power = ((epoch / 50) ** self.weight_power) 
        weighmap = (w ** power)
        if type(self.loss) == type([]):
            losses = [l(prediction, y) * weighmap * w for l, w in self.loss]
            loss = torch.sum(torch.stack(losses), dim=0)
        else:
            loss = self.loss(prediction, y) * weighmap
        loss = torch.sum(loss) / torch.sum(weighmap)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class SWValidEpoch(SWEpoch):

    def __init__(self, model, loss, metrics, device='cpu', 
                 verbose=True, weight_power=5):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.loss.reduction = 'none'
        self.weight_power = weight_power

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, w, epoch):
        with torch.no_grad():
            prediction = self.model.forward(x)
            power = ((epoch / 50) ** self.weight_power) 
            weighmap = (w ** power)
            if type(self.loss) == type([]):
                losses = [l(prediction, y) * weighmap * w for l, w in self.loss]
                loss = torch.sum(torch.stack(losses), dim=0)
            else:
                loss = self.loss(prediction, y) * weighmap
            loss = torch.sum(loss) / torch.sum(weighmap)
        return loss, prediction
