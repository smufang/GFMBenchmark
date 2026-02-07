import torch
from torch_geometric.utils import dropout_edge
import copy


def adversarial_aug_train(model_graph, model_text, node_attack, perturb_shapes, step_size, m):
    model_graph.train()
    model_text.train()
    perturbs = []
    for perturb_shape in perturb_shapes:
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size)
        perturb.requires_grad_()
        perturbs.append(perturb)

    loss = node_attack(perturbs)
    loss /= m

    for i in range(m-1):
        loss.backward()
        for perturb in perturbs:
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
        
        loss = node_attack(perturbs)
        loss /=  m

    return loss

# def adversarial_aug_train(model_graph, model_text, node_attack, perturb_shapes, step_size, m):
#     """
#     PGD-style adversarial training.

#     Compared to the previous version:
#     - The inner loop uses torch.autograd.grad() to compute gradients only for
#       perturbations, instead of calling loss.backward(). This prevents
#       accumulation of autograd graphs and stabilizes GPU memory usage.
#     - Intermediate losses are used only to update perturbations and are not
#       backpropagated to model parameters.
#     - Only the final forward pass contributes to the model update, so loss
#       normalization by m is no longer required.
#     """
#     model_graph.train()
#     model_text.train()
#     perturbs = []
#     for perturb_shape in perturb_shapes:
#         perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size)
#         perturb.requires_grad_()
#         perturbs.append(perturb)

#     # compute gradients w.r.t. perturbations only; no parameter backprop in the inner loop
#     for _ in range(max(m - 1, 0)):
#         cur_loss = node_attack(perturbs)
#         grads = torch.autograd.grad(
#             cur_loss,
#             perturbs,
#             retain_graph=False,
#             create_graph=False,
#             allow_unused=False,
#         )
#         with torch.no_grad():
#             for p, g in zip(perturbs, grads):
#           # L_inf step; optionally project to [-step_size, step_size] using clamp
#                 p.add_(step_size * g.sign())
#           # Optional projection: p.clamp_(-step_size, step_size)
#           # Clear grad reference to avoid unnecessary retention
#                 p.grad = None

#     # Return only the final forward loss; the outer loop performs a single backward to update model parameters
#     final_loss = node_attack(perturbs)
#     return final_loss

def graph_aug(g, f_p, e_p):
    new_g = copy.deepcopy(g)
    drop_mask = torch.empty(
        (g.x.size(1), ),
        dtype=torch.float32,
        device=g.x.device).uniform_(0, 1) < f_p
    
    new_g.x[:, drop_mask] = 0
    e, _ = dropout_edge(new_g.edge_index, p=e_p)
    new_g.edge_index = e
    return new_g