"""MaskNet: Wang et al. (https://arxiv.org/abs/2102.07619)."""

import torch
from absl import logging


def _init_weights(module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(module.weight)
    torch.nn.init.constant_(module.bias, 0)


class MaskBlock(torch.nn.Module):
  def __init__(
    self, output_size: int, input_dim: int, mask_input_dim: int
  ) -> None:
    super(MaskBlock, self).__init__()

    self._input_layer_norm = torch.nn.LayerNorm(input_dim)

    #aggregation_size = int(mask_input_dim // 8)
    aggregation_size = int(mask_input_dim // 8)
    #self.mlp = True

    self._mask_layer = torch.nn.Sequential(
      torch.nn.Linear(mask_input_dim, aggregation_size),
      torch.nn.ReLU(),
      torch.nn.Linear(aggregation_size, input_dim),
    )
    self._mask_layer.apply(_init_weights)
    self._hidden_layer = torch.nn.Linear(input_dim, output_size)
    self._hidden_layer.apply(_init_weights)
    self._layer_norm = torch.nn.LayerNorm(output_size)

  def forward(self, input: torch.Tensor, mask_input: torch.Tensor):
    input = self._input_layer_norm(input)
    hidden_layer_output = self._hidden_layer(input * self._mask_layer(mask_input))
    return self._layer_norm(hidden_layer_output)


class MaskNet(torch.nn.Module):
  def __init__(self, output_size: int, in_features: int, mask_dim: int):
    super().__init__()

    mask_blocks = []
    n = 8

    for _ in range(n):
        mask_blocks.append(MaskBlock(output_size=output_size, input_dim=in_features, mask_input_dim=mask_dim))

    self._mask_blocks = torch.nn.ModuleList(mask_blocks)
    self.mlp = Mlp(input_size=output_size * n, output_size=output_size)

  def forward(self, input: torch.Tensor, mask_input: torch.Tensor):
      outputs = []
      for mask_layer in self._mask_blocks:
          outputs.append(mask_layer(input=input,mask_input=mask_input))
      # Share the outputs of the MaskBlocks.
      all_outputs = torch.cat(outputs, dim=1)
      output = self.mlp(all_outputs)
      return output

"""MLP feed forward stack in torch."""

class Mlp(torch.nn.Module):
  def __init__(self, output_size: int, input_size: int):
    super().__init__()

    layer_sizes = [output_size*8,output_size*4,output_size*2,output_size]
    modules = []
    for layer_size in layer_sizes[:-1]:
      modules.append(torch.nn.Linear(input_size, layer_size, bias=True))


      modules.append(
          torch.nn.BatchNorm1d(
            layer_size, affine=True, momentum=0.1
          )
      )

      modules.append(torch.nn.ReLU())

      input_size = layer_size
    modules.append(torch.nn.Linear(input_size, layer_sizes[-1], bias=True))
    modules.append(torch.nn.ReLU())
    self.layers = torch.nn.ModuleList(modules)
    self.layers.apply(_init_weights)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    net = x
    for i, layer in enumerate(self.layers):
      net = layer(net)

    return net