import ignite.engine as engine
import torch
import torch.nn.utils as nnutils


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=engine._prepare_batch,
                              check_nan=False,
                              grad_clip=None,
                              output_predictions=False):
    """As ignite.engine.create_supervised_trainer, but may also optionally perform:
    - NaN checking on predictions (in a more debuggable way than ignite.handlers.TerminateOnNaN)
    - Gradient clipping
    - Record the predictions made by a model

    Arguments:
        (as ignite.engine.create_supervised_trainer, plus)
        check_nan: Optional boolean specifying whether the engine should check predictions for NaN values. Defaults to
            False. If True, and a NaN value is encountered, then a RuntimeError will be raised with attributes 'x', 'y',
            'y_pred', 'model', details the feature, label, prediction and model, respetively, on which this occurred.
        grad_clip: Optional number, boolean or None, specifying the value to clip the infinity-norm of the gradient to.
            Defaults to None. If False or None then no gradient clipping will be applied. If True then the gradient is
            clipped to 1.0.
        output_predictions: Optional boolean specifying whether the engine should record the predictions the model made
            on a batch. Defaults to False. If True then state.output will be a tuple of (loss, predictions). If False
            then state.output will just be the loss. (Not wrapped in a tuple.)
    """

    if device:
        model.to(device)

    if grad_clip is False:
        grad_clip = None
    elif grad_clip is True:
        grad_clip = 1.0

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

        if check_nan and torch.isnan(y_pred).any():
            e = RuntimeError('Model generated NaN value.')
            e.y = y
            e.y_pred = y_pred
            e.x = x
            e.model = model
            raise e

        loss = loss_fn(y_pred, y)
        loss.backward()

        if grad_clip is not None:
            nnutils.clip_grad_norm_(model.parameters(), grad_clip, norm_type='inf')

        optimizer.step()

        if output_predictions:
            return loss.item(), y_pred
        else:
            return loss.item()

    return engine.Engine(_update)
