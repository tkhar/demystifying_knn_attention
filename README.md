# Fast Sampling Attention
This is documentation for the Fast Sampling Attention project.

## Code Structure
This is documentation for the Fast Sampling Attention project. Major files included only.

## Code Structure
```
.
├── config.py
├── backward_pass
│   ├── backward_pass.py: Implements the fast gradients
│   └── backward_pass_testing.py
├── forward_pass
│   ├── forward_pass.py: Implements fast attention
│   └── forward_pass_testing.py
├── naive_attention
│   ├── naive_attention.py: Implements forward pass naively
└── naive_backprop
    └── naive_backprop.py: Implements gradients naively.
└── random_walk_simulation
    └── random_walk.py: Implements the MCMC algorithm framework
└── softmax_expectation
    └── softmax_expectation.py: Implements the softmax expectation via Gumbel noise.
```

## TODOs
1. Improve the naive gradients so that they are more efficient.
2. Optimize the implementation of the fast algorithms.
3. Do a lot of validation testing to check the quality of the approximations.