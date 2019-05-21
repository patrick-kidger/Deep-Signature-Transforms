import candle
import ignite.engine as engine
import ignite.metrics as ignite_metrics
import sys
import time
import torch
import tqdm


def create_train_model_fn(max_epochs, optimizer_fn, loss_fn, train_dataloader, eval_dataloader, example_batch_x):

    def train_model(model, name, history, device=None):
        # Initialise all layers in model before passing parameters to optimizer
        # (necessary with the candle framework)
        model(example_batch_x)
        optimizer = optimizer_fn(model.parameters())

        history[name] = {'train_loss': [], 'train_mse': [], 'val_loss': [], 'val_mse': []}

        if device not in ('cuda', 'cpu'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        trainer = candle.create_supervised_trainer(model, optimizer, loss_fn, check_nan=True, grad_clip=1.0, device=device)
        evaluator = engine.create_supervised_evaluator(model, device=device, 
                                                       metrics={'mse': ignite_metrics.MeanSquaredError(), 
                                                                'loss': ignite_metrics.Loss(loss_fn)})
        
        log_interval = 10
        desc = "Epoch: {:4}{:12}"
        num_batches = len(train_dataloader)
        
        @trainer.on(engine.Events.STARTED)
        def log_results(trainer):

            # training
            evaluator.run(train_dataloader)
            train_mse = evaluator.state.metrics['mse']
            train_loss = evaluator.state.metrics['loss']

            # testing
            evaluator.run(eval_dataloader)
            val_mse = evaluator.state.metrics['mse']
            val_loss = evaluator.state.metrics['loss']


            tqdm.tqdm.write("train mse: {:5.4f} --- train loss: {:5.4f} --- val mse: {:5.4f} --- val loss: {:5.4f}"
                            .format(train_mse, train_loss, val_mse, val_loss), file=sys.stdout)

            model_history = history[name]
            model_history['train_loss'].append(train_loss)
            model_history['train_mse'].append(train_mse)
            model_history['val_loss'].append(val_loss)
            model_history['val_mse'].append(val_mse)
        
        @trainer.on(engine.Events.EPOCH_STARTED)
        def create_pbar(trainer):
            trainer.state.pbar = tqdm.tqdm(initial=0, total=num_batches, desc=desc.format(trainer.state.epoch, ''), 
                                           file=sys.stdout)

        @trainer.on(engine.Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            iteration = (trainer.state.iteration - 1) % len(train_dataloader) + 1
            if iteration % log_interval == 0:
                trainer.state.pbar.desc = desc.format(trainer.state.epoch, ' Loss: {:5.4f}'.format(trainer.state.output))
                trainer.state.pbar.update(log_interval)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def log_results_(trainer):
            trainer.state.pbar.n = num_batches
            trainer.state.pbar.last_print_n = num_batches
            trainer.state.pbar.refresh()        
            trainer.state.pbar.close()
            log_results(trainer)

        start = time.time()
        trainer.run(train_dataloader, max_epochs=max_epochs)
        end = time.time()
        tqdm.tqdm.write("Training took {:.2f} seconds.".format(end - start), file=sys.stdout)
        
    return train_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
