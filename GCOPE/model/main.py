from importlib import import_module
import torch
import torch_scatter


class Model(torch.nn.Module):
    def __init__(
                self,
                backbone,
                answering = torch.nn.Identity(),
                saliency = torch.nn.Identity(),
            ):
        super().__init__()
        self.backbone = backbone
        self.answering = answering
        self.saliency = saliency

    def forward(self, data, idx=None, task_type = None):
        # idx refine to original batch index as backbone using 'self.global_pool(h, batch)' - batch won't be changed by loader
        data.x = self.saliency((data.x))
        embed = self.backbone(data)
        if task_type == 'node':
            embed = embed[idx]
        elif task_type == 'edge':
            src_idx, dst_idx = idx
            embed = torch.cat([embed[src_idx], embed[dst_idx]], dim=-1)
        elif task_type == 'graph':
            embed = embed[idx]
        return self.answering(embed)


def get_model(
        backbone_kwargs,
        answering_kwargs = None,
        saliency_kwargs = None,
    ):

    backbone = import_module(f'model.backbone.{backbone_kwargs.pop("name")}').get_model(**backbone_kwargs)
    answering = torch.nn.Identity() if answering_kwargs is None else import_module(f'model.answering.{answering_kwargs.pop("name")}').get_model(**answering_kwargs)
    saliency = torch.nn.Identity() if saliency_kwargs is None else import_module(f'model.saliency.{saliency_kwargs.pop("name")}').get_model(**saliency_kwargs)

    return Model(backbone, answering, saliency)
