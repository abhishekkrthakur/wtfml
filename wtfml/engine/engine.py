"""
__author__: Abhishek Thakur
"""

import datetime
import torch
from tqdm import tqdm
from ..utils import AverageMeter

try:
    from torch.cuda import amp
    _amp_available = True
except ImportError:
    _amp_available = False

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    _xla_available = True
except ImportError:
    _xla_available = False


def reduce_fn(vals):
    return sum(vals) / len(vals)


class Engine:
    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        tpu_print=10,
        fp16=False,
        model_fn=None,
        use_mean_loss=False,
    ):
        """
        model_fn should take batch of data, device and model and return loss
        for example:
            def model_fn(data, device, model):
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                _, loss = model(images, targets)
                return loss
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.use_tpu = use_tpu
        self.tpu_print = tpu_print
        self.model_fn = model_fn
        self.fp16 = fp16
        if self.fp16 and not _amp_available:
            raise Exception(
                "You want to use fp16 but dont have amp installed"
            )
        self.use_mean_loss = use_mean_loss
        self.scaler = None

        if self.use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if self.fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if self.fp16:
            self.scaler = amp.GradScaler()

    def train(self, data_loader):
        losses = AverageMeter()
        self.model.train()
        print_idx = int(len(data_loader) * self.tpu_print / 100)
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        if self.use_tpu:
            para_loader = pl.ParallelLoader(data_loader, [self.device])
            tk0 = para_loader.per_device_loader(self.device)
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tk0):
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()

            if self.model_fn is None:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                _, loss = self.model(**data)
            else:
                if self.fp16:
                    with amp.autocast():
                        loss = self.model_fn(data, self.device, self.model)
                else:
                    loss = self.model_fn(data, self.device, self.model)

            if not self.use_tpu:
                with torch.set_grad_enabled(True):
                    if self.use_mean_loss:
                        loss = loss.mean()

                    if self.fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (b_idx + 1) % self.accumulation_steps == 0:
                        if self.fp16:
                            self.scaler.step(self.optimizer)
                        else:
                            self.optimizer.step()

                        if self.scheduler is not None:
                            self.scheduler.step()

                        if self.fp16:
                            self.scaler.update()

                        if b_idx > 0:
                            self.optimizer.zero_grad()

            else:
                loss.backward()
                xm.optimizer_step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                if b_idx > 0:
                    self.optimizer.zero_grad()
            if self.use_tpu:
                reduced_loss = xm.mesh_reduce("loss_reduce", loss, reduce_fn)
                losses.update(reduced_loss.item(), data_loader.batch_size)
            else:
                losses.update(loss.item(), data_loader.batch_size)

            if not self.use_tpu:
                tk0.set_postfix(loss=losses.avg)
            else:
                if b_idx % print_idx == 0 or b_idx == len(data_loader):
                    xm.master_print(
                        f"{datetime.datetime.now()}: Batch {b_idx} / {len(data_loader)}, loss={losses.avg}"
                    )
        if not self.use_tpu:
            tk0.close()
        return losses.avg

    def evaluate(self, data_loader, return_predictions=False):
        losses = AverageMeter()
        print_idx = int(len(data_loader) * self.tpu_print / 100)
        self.model.eval()
        final_predictions = []
        with torch.no_grad():
            if self.use_tpu:
                para_loader = pl.ParallelLoader(data_loader, [self.device])
                tk0 = para_loader.per_device_loader(self.device)
            else:
                tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                if self.fp16:
                    with amp.autocast():
                        batch_preds, loss = self.model(**data)
                else:
                    batch_preds, loss = self.model(**data)
                if return_predictions:
                    final_predictions.append(batch_preds)
                if self.use_tpu:
                    reduced_loss = xm.mesh_reduce("loss_reduce", loss, reduce_fn)
                    losses.update(reduced_loss.item(), data_loader.batch_size)
                else:
                    if self.use_mean_loss:
                        loss = loss.mean()
                    losses.update(loss.item(), data_loader.batch_size)
                if not self.use_tpu:
                    tk0.set_postfix(loss=losses.avg)
                else:
                    if b_idx % print_idx == 0 or b_idx == len(data_loader):
                        xm.master_print(
                            f"{datetime.datetime.now()}: Batch {b_idx} / {len(data_loader)}, loss={losses.avg}"
                        )
            if not self.use_tpu:
                tk0.close()
        return losses.avg, final_predictions

    def predict(self, data_loader):
        self.model.eval()
        final_predictions = []
        if self.use_tpu:
            raise Exception("TPU not available for predict yet!")
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                predictions, _ = self.model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions
