import time
import torch
from functools import reduce
import logging


def get_slicer(slice_tensor: torch.Tensor, index_dim:int = 1) -> tuple:
    """
    Helper slice a tensor by the entries of another tensor.

    :param slice_tensor:
    :param: dimension to use for indexing
    :return: tuple for indexing.
    """
    try:
        indexes = range(slice_tensor.shape[index_dim])
    except (AttributeError, SyntaxError) as e:
        logging.error("")
        raise e

    return tuple(slice_tensor.T[i] for i in indexes)


def chain_compose(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Composes tensors by their first and last dimensions.

    e.g., if nxn tensors are given, you get the matrix multiplication.
    :param *tensors: tensors to compose
    :return: result of composition
    """
    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor), "All arguments must be pytorch tensors"
    def compose(x,y):
        return torch.tensordot(x,y,dims=1)

    return reduce(compose, tensors)


# inspired by the tntensors library
def optimize(tensors, loss_function, optimizer=torch.optim.Adam, tol=1e-4, max_iter=1e4, print_freq=500, verbose=True):
    """
    High-level wrapper for iterative learning.

    Default stopping criterion: either the absolute (or relative) loss improvement must fall below `tol`.
    In addition, the rate loss improvement must be slowing down.

    :param tensors: one or several tensors; will be fed to `loss_function` and optimized in place
    :param loss_function: must take `tensors` and return a scalar (or tuple thereof)
    :param optimizer: one from https://pytorch.org/docs/stable/optim.html. Default is torch.optim.Adam
    :param tol: stopping criterion
    :param max_iter: default is 1e4
    :param print_freq: progress will be printed every this many iterations
    :param verbose:
    """
    def log_iter(terminal: bool):
        if iteration % print_freq != 0 and not terminal:
            return None
        iter_spacing = len(str(max_iter))
        losses_str = ' + '.join([f"{l_i.item():10.6f}" for l_i in loss])
        if len(loss) > 1:
            losses_str += f" = {losses[-1].item():10.4}"
        log_str = [f"iter: {iteration: <{iter_spacing}}",
                   f"loss: {losses_str}",
                   f"total time: {time.time() - start:9.4f}"]
        if terminal:
            if converged:
                log_str.append(f"termination: converged (tol={tol})")
            else:
                log_str.append(f"termination:  max_iter was reached: {max_iter}")
        print(" | ".join(log_str))

    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    parameters = [t for t in tensors if t.requires_grad]
    if len(parameters) == 0:
        raise ValueError("There are no parameters to optimize. Did you forget a requires_grad=True somewhere?")

    optimizer = optimizer(parameters)
    losses = []
    converged = False
    start = time.time()
    iteration = 0
    while True:
        optimizer.zero_grad()
        loss = loss_function(*tensors)
        if not isinstance(loss, (tuple, list)):
            loss = [loss]
        total_loss = sum(loss)
        total_loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(total_loss.detach())

        delta_loss = losses[-1] - losses[-2] if len(losses) >= 2 else float('-inf')

        converged = (
                iteration >= 2
                and tol is not None
                and (losses[-1] <= tol or -delta_loss / losses[-1] <= tol)
                and losses[-2] - losses[-1] < losses[-3] - losses[-2])
        terminated = converged or iteration == max_iter
        if verbose:
            log_iter(terminated)
        if terminated:
            break
        iteration += 1
