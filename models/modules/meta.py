import torch
import torch.nn as nn
import re
import warnings

from collections import OrderedDict


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix="", recurse=True):
        gen = self._named_members(
            lambda module: (
                module._parameters.items() if isinstance(module, MetaModule) else []
            ),
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r"^{0}\.(.+)".format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r"\1", k)
                    for k in all_names
                    if key_re.match(k) is not None
                ]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn(
                "Module `{0}` has no parameter corresponding to the "
                "submodule named `{1}` in the dictionary `params` "
                "provided as an argument to `forward()`. Using the "
                "default parameters for this submodule. The list of "
                "the parameters in `params`: [{2}].".format(
                    self.__class__.__name__, key, ", ".join(all_names)
                ),
                stacklevel=2,
            )
            return None

        return OrderedDict([(name, params[f"{key}.{name}"]) for name in names])


def grad_norm(params_list, grads, adaptive):
    # put everything on the same device, in case of model parallelism
    shared_device = params_list[0].device
    l = list(range(len(grads)))
    norm = torch.norm(
        torch.stack(
            [
                ((torch.abs(params_list[i]) if adaptive else 1.0) * grads[i])
                .norm(p=2)
                .to(shared_device)
                for i in l
                if grads is not None
            ]
        ),
        p=2,
    )
    return norm


# def gradient_update_parameters(model,
#                                loss,
#                                params=None,
#                                step_size=0.5,
#                                first_order=False):
#     """Update of the meta-parameters with one step of gradient descent on the
#     loss function.
#
#     Parameters
#     ----------
#     model : `torchmeta.modules.MetaModule` instance
#         The model.
#
#     loss : `torch.Tensor` instance
#         The value of the inner-loss. This is the result of the training dataset
#         through the loss function.
#
#     params : `collections.OrderedDict` instance, optional
#         Dictionary containing the meta-parameters of the model. If `None`, then
#         the values stored in `model.meta_named_parameters()` are used. This is
#         useful for running multiple steps of gradient descent as the inner-loop.
#
#     step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
#         The step size in the gradient update. If an `OrderedDict`, then the
#         keys must match the keys in `params`.
#
#     first_order : bool (default: `False`)
#         If `True`, then the first order approximation of MAML is used.
#
#     Returns
#     -------
#     updated_params : `collections.OrderedDict` instance
#         Dictionary containing the updated meta-parameters of the model, with one
#         gradient update wrt. the inner-loss.
#     """
#     if not isinstance(model, MetaModule):
#         raise ValueError('The model must be an instance of `torchmeta.modules.'
#                          'MetaModule`, got `{0}`'.format(type(model)))
#
#     if params is None:
#         params = OrderedDict(model.meta_named_parameters())
#
#     grads = torch.autograd.grad(loss,
#                                 params.values(),
#                                 create_graph=not first_order)
#
#     updated_params = OrderedDict()
#
#     if isinstance(step_size, (dict, OrderedDict)):
#         for (name, param), grad in zip(params.items(), grads):
#             updated_params[name] = param - step_size[name] * grad
#
#     else:
#         for (name, param), grad in zip(params.items(), grads):
#             updated_params[name] = param - step_size * grad
#
#     return updated_params


def gradient_update_parameters(
    model,
    loss,
    train_input,
    train_target,
    params=None,
    step_size=0.5,
    first_order=False,
    adaptive=False,
    alpha=0.0005,
    sam_lower=True,
):

    if not isinstance(model, MetaModule):
        raise ValueError(
            "The model must be an instance of `torchmeta.modules."
            "MetaModule`, got `{0}`".format(type(model))
        )

    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    key_list = params.keys()
    items_list = params.values()

    grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

    if sam_lower:
        params_list = list(params.values())
        gradnorm = grad_norm(params_list, grads, adaptive)
        scale = alpha / (gradnorm + 1e-12)

        l = list(range(len(grads)))
        old_p = []
        for i in l:
            old_p.append(torch.zeros_like(params_list[i]))

        for i in l:
            e_w = (
                (torch.pow(params_list[i], 2) if adaptive else 1.0)
                * grads[i]
                * scale.to(params_list[i])
            )
            params_list[i] = params_list[i].add(
                e_w
            )  # climb to the local maximum "w + e(w)"

        params_new = OrderedDict(zip(key_list, params_list))
        train_logit = model(train_input, params=params_new)
        inner_loss = F.cross_entropy(train_logit, train_target)
        model.zero_grad()
        grads_new = torch.autograd.grad(
            inner_loss, params_new.values(), create_graph=not first_order
        )

        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads_new):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads_new):
                updated_params[name] = param - step_size * grad

    else:
        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size * grad

    return updated_params
