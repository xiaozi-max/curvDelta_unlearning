import torch
import torch.nn.functional as F

from model.crispedit import projection_cache_from_covariances


def _split(names, k):
    k = max(1, min(k, len(names)))
    return [names[i * len(names) // k : (i + 1) * len(names) // k] for i in range(k)]


class _Collector:
    """Hook capture of module inputs / output-grads; CPU-backed accumulators."""

    def __init__(self, modules, dtype):
        self.modules = modules
        self.dtype = dtype
        self.A = {n: torch.zeros(m.in_features, m.in_features, dtype=dtype) for n, m in modules.items()}
        self.B = {n: torch.zeros(m.out_features, m.out_features, dtype=dtype) for n, m in modules.items()}
        self.n = {n: 0 for n in modules}
        self._hooks = []
        self._inputs = {}
        self._grads = {}

    def activate(self, names, device):
        for name in names:
            self.A[name] = self.A[name].to(device)
            self.B[name] = self.B[name].to(device)
            self._hooks.append(self.modules[name].register_forward_hook(self._hook(name)))

    def deactivate(self, names, device):
        self._clear()
        for name in names:
            self.A[name] = self.A[name].cpu()
            self.B[name] = self.B[name].cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _hook(self, name):
        def forward(module, args, output):
            self._inputs[name] = args[0].detach()
            output.register_hook(
                lambda grad, _name=name: self._grads.__setitem__(_name, grad.detach())
            )

        return forward

    def add(self, names, mask):
        for name in names:
            x, g = self._inputs.get(name), self._grads.get(name)
            if x is None or g is None:
                continue
            device = self.A[name].device
            x = x.to(device)[:, :-1, :][mask].to(self.dtype)
            g = g.to(device)[:, :-1, :][mask].to(self.dtype)
            self.A[name].addmm_(x.T, x)
            self.B[name].addmm_(g.T, g)
            self.n[name] += x.shape[0]
        # Only drop this batch's captured tensors; the module hooks must stay
        # installed for the remaining batches (removed in deactivate()).
        self._inputs.clear()
        self._grads.clear()

    def _clear(self):
        self._inputs.clear()
        self._grads.clear()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


def collect_kfac_covariances(model, dataloader, target_modules, device, dtype, num_chunks=1):
    """Return ``{name: (A, B, n)}`` with CPU A (in,in), B (out,out); n = valid tokens."""
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    names = list(target_modules)
    collector = _Collector(target_modules, dtype)
    model.eval()
    try:
        for chunk in _split(names, num_chunks):
            collector.activate(chunk, device)
            with torch.enable_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    labels = batch.pop("labels")
                    mask = labels[..., 1:].ne(-100).to(device)
                    model.zero_grad(set_to_none=True)
                    logits = model(**batch, use_cache=False).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(device)
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    loss.backward()
                    with torch.no_grad():
                        collector.add(chunk, mask)
                    model.zero_grad(set_to_none=True)
            collector.deactivate(chunk, device)
    finally:
        model.train()
    result = {}
    for name in names:
        n = collector.n[name]
        if n == 0:
            raise RuntimeError(f"CrispEdit: no valid tokens collected for {name}")
        result[name] = ((collector.A[name] / n).cpu(), (collector.B[name] / n).cpu(), n)
    return result


def build_projection_cache(model, covariance_cache, energy_threshold):
    """Return ``{parameter: {"Ua", "Ub", "M"}}`` for each target ``.weight``."""
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    caches = {}
    for name, (A, B, _) in covariance_cache.items():
        cache = projection_cache_from_covariances(A.to(device), B.to(device), energy_threshold=energy_threshold)
        caches[modules[name].weight] = cache
    return caches
