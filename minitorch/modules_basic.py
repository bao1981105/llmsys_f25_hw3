"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(rand((num_embeddings, embedding_dim)))
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # one hot + matmul is slow, but we use indexing directly to get the embedding vectors, we leave computation graph
        one_hot_x = one_hot(x, self.num_embeddings)  # (bs, seq_len, num_embeddings)
        # Reshape for matmul: (bs * seq_len, num_embeddings) @ (num_embeddings, emb_dim)
        flat_one_hot = one_hot_x.view(bs * seq_len, self.num_embeddings)
        emb = flat_one_hot @ self.weights.value  # (bs * seq_len, emb_dim)
        return emb.view(bs, seq_len, self.embedding_dim)
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        if self.training and self.p_dropout > 0:
            # Use numpy's random generator for deterministic results in tests
            mask = np.random.binomial(1, 1 - self.p_dropout, size=x.shape)
            # can we use np.random.binomial and tensor_from_numpy in forward? Will this break the autograd?
            #mask = rand(x.shape, backend=x.backend)
            mask = tensor_from_numpy(mask, backend=x.backend)
            output = x * mask
            output = output / (1 - self.p_dropout)
            return output
        else:
            return x
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.backend = backend
        bound = 1 / np.sqrt(in_size)
        self.weights = Parameter(
            tensor_from_numpy(
                np.random.uniform(-bound, bound, (in_size, out_size)),
                backend=backend
            )
        ) 
        # weights shape (in_size, out_size)
        if bias:
            self.bias = Parameter(
                tensor_from_numpy(
                    np.random.uniform(-bound, bound, (out_size,)),
                    backend=backend
                )
            )
        else:
            self.bias = None
        # bias shape (out_size,)
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        # batch, in_size = x.shape
        # output = x @ self.weights.value  # (batch, out_size)
        # if self.bias is not None:
        #     output = output + self.bias.value  # broadcasting (out_size, )
        # return output
        
        ### BEGIN ASSIGN3_2
        if len(x.shape) == 2:
            return x @ self.weights.value + (self.bias.value if self.bias is not None else 0)
        elif len(x.shape) == 3:
            # (batch, seq_len, in_size)
            batch, seq_len, in_size = x.shape
            x_flat = x.view(batch * seq_len, in_size)
            out_flat = x_flat @ self.weights.value  # (batch * seq_len, out_size)
            if self.bias is not None:
                out_flat = out_flat + self.bias.value
            out = out_flat.view(batch, seq_len, self.out_size)
            return out
        else:
            raise ValueError(f"Linear.forward: Unexpected input shape {x.shape}")
        ### END ASSIGN3_2

class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(tensor_from_numpy(np.ones(dim), backend=backend))
        self.bias = Parameter(tensor_from_numpy(np.zeros(dim), backend=backend))
        self.backend = backend
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        mean = x.mean(dim=1).view(batch, 1)  # (bs, 1)
        # for layer norm, we normalize over the last dimension
        # var = ((x - mean) ** 2).sum(axes=(1,), keepdims=True) / dim  # (bs, 1)
        var = x.var(dim=1).view(batch, 1)
        x_normalized = (x - mean) / (var + self.eps) ** 0.5  # (bs, dim)
        output = x_normalized * self.weights.value + self.bias.value  # broadcasting (dim, )
        return output
        ### END ASSIGN3_2
