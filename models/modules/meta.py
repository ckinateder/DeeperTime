import torch.nn as nn


class MetaModule(nn.Module):
    def __init__(self):
        super(MetaModule, self).__init__()

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
