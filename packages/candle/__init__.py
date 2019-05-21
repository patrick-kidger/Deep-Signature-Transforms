try:
    from .engine import create_supervised_trainer
except ImportError:  # no ignite
    def create_supervised_trainer(*args, **kwargs):
        raise ImportError('Ignite not installed')

from .modules import (Lambda,
                      Flatten,
                      View,
                      Concat,
                      Split,
                      SkipConnection,
                      NoInputSpec,
                      CannedNet,
                      CannedResNet)

from .recurrent import (Window,
                        Recur)

from .tracing import (convert_to_tensor,
                      Integer)

from .utils import (batch_fn,
                    outer_product,
                    flatten,
                    batch_flatten,
                    cat,
                    stack)
