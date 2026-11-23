# Backend API

Every nonlinear fit tries its model from a grid of starting values, and the grid
is fitted as one batch of array operations before the best few results are
polished on the CPU. `set_backend` chooses the array library for that batch. See
[Speed and the GPU](../getting-started.md#speed-and-the-gpu) for when the GPU
is worth it.

## set_backend

::: sars.set_backend

## get_backend

::: sars.get_backend
