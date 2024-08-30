import torch


def _init_weights(module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(module.weight)
    torch.nn.init.constant_(module.bias, 0)

class MaskNet(torch.nn.Module):
  def __init__(self, in_features: int, output_size: int):
    super().__init__()
    mask_blocks = []

    input_size = in_features

    mask_blocks.append(MaskBlock(input_size, in_features, output_size))
    input_size = output_size
    mask_blocks.append(MaskBlock(input_size, in_features, output_size))

    self._mask_blocks = torch.nn.ModuleList(mask_blocks)

  def forward(self, inputs: torch.Tensor):
    net = inputs
    for mask_layer in self._mask_blocks:
        net = mask_layer(net=net, mask_input=inputs)
        # Share the output of the stacked MaskBlocks.
        output = net
    return output
    

class MaskBlock(torch.nn.Module):
  def __init__(
    self, input_dim: int, mask_input_dim: int, output_size: int
  ) -> None:
    super(MaskBlock, self).__init__()
    self._input_layer_norm = torch.nn.LayerNorm(input_dim)
    #self._input_layer_norm = None

    aggregation_size = int(mask_input_dim // 16)

    self._mask_layer = torch.nn.Sequential(
      torch.nn.Linear(mask_input_dim, aggregation_size),
      torch.nn.ReLU(),
      torch.nn.Linear(aggregation_size, input_dim),
    )
    self._mask_layer.apply(_init_weights)
    self._hidden_layer = torch.nn.Linear(input_dim, output_size)
    self._hidden_layer.apply(_init_weights)
    self._layer_norm = torch.nn.LayerNorm(output_size)

  def forward(self, net: torch.Tensor, mask_input: torch.Tensor):
    net = self._input_layer_norm(net)
    hidden_layer_output = self._hidden_layer(net * self._mask_layer(mask_input))
    return self._layer_norm(hidden_layer_output)