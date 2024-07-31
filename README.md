# ChebyKAN
Implemetation of Kolmogorov Arnold Networks using Cheby Polynomials instead of B-splines

* B-splines are poor in performance and not very intutive to use
* using chebyshev polynomials to approximate functions

Chebyshev polynomials are orthogonal polynomials defined on the interval [-1, 1]. They are good at approximating functions and can be calculated recursively.

ChebyKAN = Linear + custom activation function

# Usage

```bash
from chebykanlayer import ChebyKANLayer

class MNISTChebyKAN(nn.Module):
    def __init__(self):
        super(MNISTChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 32, 4)
        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(32, 16, 4)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyKANLayer(16, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x
```

**Note**: Since Chebyshev polynomials are defined on the interval [-1, 1], we need to use tanh to keep the input in that range. We also use LayerNorm to avoid gradient vanishing caused by tanh. Removing LayerNorm will cause the network really hard to train.
