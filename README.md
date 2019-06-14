# Deep Signatures
Using signatures as layers in a neural network.

This is the code for the paper [Deep Signatures](https://arxiv.org/abs/1905.08494) by Bonnier, Kidger, Perez Arribas, Salvi, Lyons 2019.

To use signatures in your own projects, look at [Signatory](https://github.com/patrick-kidger/signatory), which was a spin-off from this project.

## Overview
If you're coming at this already knowing something about neural networks, then the idea is that the 'signature transform' is a transformation that does a particularly good job extracting features from streams of data, so it's a natural thing to try and build into our neural network models.

If you're coming at this already knowing something about signatures, then you probably know that they've previously only been used as a feature transformation, on top of which a model is built. But it is actually possible to backpropagate through the signature transform, so as long you design your model correctly (it has to be 'stream-preserving'; see the paper), then it actually makes sense to embed the signature within a neural network. Learning a nonlinearity before the signature transform provides a compact way to select which terms in the signature (of the original path) are useful for the given dataset.

## What are signatures?
The signature of a stream of data is essentially a collection of statistics about that stream of data. This collection of statistics does such a good job of capturing the information about the stream of data that it actually determines the stream of data uniquely. (Up to something called 'tree-like equivalance' anyway, which is really just a technicality. It's an equivalence relation that matters about as much as two functions being equal almost everywhere. That is to say, not much at all.) The signature transform is a particularly attractive tool in machine learning because it is what we call a 'universal nonlinearity': it is sufficiently rich that it captures every possible nonlinear function of the original stream of data. Any function of a stream is *linear* on its signature. Now for various reasons this is a mathematical idealisation not borne out in practice (which is why we put them in a neural network and don't just use a simple linear model), but they still work very well!

## Directory layout and reproducability
The `packages` directory contains the `candle` and `siglayer` packages, which were created for this project, but have standalone value. The `candle` package just provides various helpers for using PyTorch. The `siglayer` package provides the functionality necessary to perform signature calculations in PyTorch (and internally uses the `iisignature` package). It also provides some example models using the signature.

The `src` directory contains the scripts for our experiments. Reproducability should be easy: just run the `.ipynb` files.

## Dependencies
Python 3.7 was used. Virtual environments and packages were managed with [Miniconda](https://docs.conda.io/en/latest/miniconda.html). The following external packages were used, and may be installed via `pip3 install -r requirements.txt`.

[`fbm==0.2.0`](https://pypi.org/project/fbm/) for generating fractional Brownian motion.

[`gym==0.12.1`](https://gym.openai.com/)

[`pytorch-ignite==0.1.2`](https://pytorch.org/ignite/) is an extension to PyTorch.

[`iisignature==0.23`](https://github.com/bottler/iisignature) for calculating signatures. (Which was used as [Signatory](https://github.com/patrick-kidger/signatory) had not been developed yet.)

[`jupyter==1.0.0`](https://jupyter.org/)

[`matplotlib==2.2.4`](https://matplotlib.org/)

[`pandas==0.24.2`](https://pandas.pydata.org/)

[`torch==1.0.1`](https://pytorch.org/)

[`scikit-learn==0.20.3`](https://scikit-learn.org/)

[`sdepy==1.0.1`](https://pypi.org/project/sdepy/) for simulating solutions to stochastic differential equations.

[`tqdm==4.31.1`](https://github.com/tqdm/tqdm) for progress bars.
