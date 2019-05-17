"""
Implement the WPL module.

"""
from collections import OrderedDict

import IPython
import torch
import copy

import utils

logger = utils.get_logger()


def apply_fn_to_container(layer, container, mode='apply', fn=None):
    applyfn = fn or copy.deepcopy
    try:
        for idx, p in enumerate(layer.parameters()):
            name = str(idx)
            if mode == 'apply':
                setattr(container, str(idx), applyfn(p.data))
            elif mode == 'zero':
                setattr(container, str(idx), torch.zeros_like(p.data))
            elif mode == 'one':
                setattr(container, str(idx), torch.ones_like(p.data))
            elif mode == 'register':
                container.register_buffer(name, applyfn(p))
            elif mode == 'update':
                container.register_buffer(name, applyfn(p))
            else:
                raise ValueError(f"Not recognize mode {mode} apply function to container {fn} ")
    except ValueError as e:
        print(e)

    return container


def create_opt_weights(layer, fn=None):
    """
    Create a new module with no operation define, only as a container for parameters.
    should have exactly same format as layer.parameters()
    :param layer: module object
    :param fn: function when doing the register
    :return:
    """
    new_module = WPLModuleBufferContainer()
    apply_fn_to_container(layer, new_module, mode='register', fn=fn)
    return new_module


def update_opt_weights(layer, container):
    apply_fn_to_container(layer, container, 'update')
    return container


class WPLModuleBufferContainer(torch.nn.Module):
    # support a simple
    def __getitem__(self, item):
        return getattr(self, item)


class WPLModule(torch.nn.Module):

    """
    This module is a wrapper to normal pytorch module, with the additional function

    save_optimal_weights

    """

    ### WPL related implemenentation ###
    def __init__(self, args):
        super(WPLModule, self).__init__()
        self._modules_opt = OrderedDict()
        self._fisher = OrderedDict()
        self.args = args
        self.ignore_module_keys = []
        self.wpl_monitored_modules = self._modules
        self.build = False

    def _apply(self, fn):
        super(WPLModule, self)._apply(fn)

        for k, m in self.wpl_monitored_modules.items():
            if k not in self.ignore_module_keys:
                if isinstance(m, WPLModule):
                    m._apply(fn)
                else:
                    self._modules_opt[k] = apply_fn_to_container(m, self._modules_opt[k], mode='apply', fn=fn)
                    self._fisher[k] = apply_fn_to_container(m, self._fisher[k], mode='apply', fn=fn)

        return self

    def  init_wpl_weights(self, force=False):
        """
        Init for WPL operations.

        NOTE: only take care of all the weights in self._modules, and others.
        for self parameters and operations, please override later.

        :return:
        """
        # update all the modules
        for k, m in self.wpl_monitored_modules.items():
            if k not in self.ignore_module_keys:
                if isinstance(m, WPLModule):
                    if not m.build or force:
                        m.init_wpl_weights()
                else:
                    self._modules_opt[k] = create_opt_weights(m)
                    self._fisher[k] = create_opt_weights(m, torch.zeros_like)

        self.build = True

    def set_fisher_zero(self):

        for k, m in self.wpl_monitored_modules.items():
            if k not in self.ignore_module_keys:
                if isinstance(m, WPLModule):
                    m.set_fisher_zero()
                else:
                    self._fisher[k] = apply_fn_to_container(m, self._fisher[k],
                                                            mode='apply', fn=torch.zeros_like)

    def update_optimal_weights(self):
        """ Update the weights with optimal """
        for k, m in self.wpl_monitored_modules.items():

            if k not in self.ignore_module_keys:
                if isinstance(m, WPLModule):
                    m.update_optimal_weights()
                else:
                    self._modules_opt[k] = apply_fn_to_container(m, self._modules_opt[k])

    def update_fisher(self, keys, epoch=0):
        def _update_fisher(fisher, p):
            """ Update fisher information logic
            fisher: torch.Tensor
            p: torch.nn.Parameter
            """

            if p.grad is None:
                return p.data.clone().fill_(0.0)

            alpha = self.args.alpha_fisher
            if epoch >= self.args.alpha_decay_after:
                degree = max(epoch - self.args.alpha_decay_after + 1, 0)
                alpha = max(self.args.alpha_fisher - degree * self.args.alpha_decay, 0)

            if self.args.momentum:
                fisher *= (1 - self.args.lambda_fisher)
            fisher += self.args.lambda_fisher * p.grad.data.clone() ** 2
            if self.args.fisher_clip_by_norm > 0:
                if fisher.norm() > self.args.fisher_clip_by_norm:
                    fisher = fisher / fisher.norm() * self.args.fisher_clip_by_norm
            return fisher * alpha

        count = 0
        for k in keys:
            if k not in self.ignore_module_keys:
                m = self.wpl_monitored_modules.get(k)
                if isinstance(m, WPLModule):
                    m.update_fisher(k)
                else:
                    f_module = self._fisher.get(k)

                    for idx, p in enumerate(m.parameters()):
                        setattr(f_module, str(idx), _update_fisher(
                            getattr(f_module, str(idx)),
                            p
                        ))
                        count += 1
        logger.debug(f'update fisher with {keys} and param count {count}')
        return count

    def compute_weight_plastic_loss_with_update_fisher(self, keys):
        """
        Return the weight layer (can be freely accessed)

        based on dag figure
        - Update the gradient as fisher information
        - return loss

        :param dag: list of dags
        :return: loss function term. with Fisher information.
        """

        loss = 0
        count = 0
        for k in keys:
            if k not in self.ignore_module_keys:
                try:
                    layer = self.wpl_monitored_modules[k]
                except KeyError as e:
                    logger.warn(f'compute wpl but encounter key error {e}')
                    continue
                if isinstance(layer, WPLModule):
                    loss += layer.compute_weight_plastic_loss_with_update_fisher(k)
                else:
                    for idx, p in enumerate(layer.parameters()):

                        fisher = self._fisher[k][str(idx)]
                        diff = (p - self._modules_opt[k][str(idx)]) ** 2

                        try:
                            loss += (fisher * diff).sum()
                            count += 1
                        except RuntimeError as e:
                            print(f"Got error {e}")
                            self.cuda()

        return loss

