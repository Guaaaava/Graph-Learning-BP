"""Experimental, inference-scoped log-likelihood cache; never a default backend.

The accepted ParticleBP owns the graph, support, synchronous cavity recursion,
moments and diagnostics. Only its pair-message implementation is overridden.
No probability-domain approximation, resampling, floor or precision change.
"""
from __future__ import annotations

from contextlib import nullcontext
import torch
from tools.bp_distance import distance_2d

from gib4cl.inference.particle_bp import ParticleBP, _normalise


class CachedParticleBP(ParticleBP):
    """Retain a bounded prefix of likelihood blocks during ONE infer() call.

    cache_bytes bounds retained tensor payload, not total working memory or CUDA
    allocator reservations. Blocks that do not fit are recomputed, without
    eviction or changing the particle integration. Finally clears all tensors,
    including on failure; no reuse across masks, supports, graphs or calls.
    Instances cannot serve concurrent infer() calls.
    """

    def __init__(self, *args, cache_bytes=0, profile_ranges=False, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError('cache_bytes must be a nonnegative integer')
        if not isinstance(profile_ranges, bool):
            raise ValueError('profile_ranges must be boolean')
        self.cache_bytes = cache_bytes
        self.profile_ranges = profile_ranges
        self._kernel_cache = None
        self.last_cache_stats = None

    def _phase(self, name):
        return torch.profiler.record_function(name) if self.profile_ranges else nullcontext()

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        if self._kernel_cache is not None:
            raise RuntimeError('Concurrent/nested infer on one cached BP instance is unsupported')
        self._kernel_cache = {}
        self.last_cache_stats = dict(
            cache_cap_bytes=self.cache_bytes, retained_payload_peak_bytes=0,
            cache_hit_blocks=0, cache_miss_blocks=0, uncached_blocks=0,
            computed_pair_distances=0, logical_pair_distances=0,
            status='running', released_after_infer=False,
        )
        try:
            result = super().infer(*args, **kwargs)
            self.last_cache_stats['status'] = 'complete'
            # The original pair_distance_evaluations remains a LOGICAL workload.
            # Actual computed distances are explicitly separate below.
            result.diagnostics['experimental_likelihood_cache'] = dict(self.last_cache_stats)
            result.diagnostics['experimental_likelihood_cache']['released_after_infer'] = True
            return result
        except Exception:
            self.last_cache_stats['status'] = 'failed'
            raise
        finally:
            self._kernel_cache.clear()
            self._kernel_cache = None
            self.last_cache_stats['released_after_infer'] = True

    def _pair_messages(self, left, right, left_cavity, right_cavity, z, variance):
        # Standalone private calls do not retain tensors beyond their lifetimes.
        if self._kernel_cache is None:
            return super()._pair_messages(left, right, left_cavity, right_cavity, z, variance)
        p, chunk = self.num_particles, self.chunk_size
        to_left = left.new_full((p,), -torch.inf)
        to_right = right.new_full((p,), -torch.inf)
        finite_kernel = torch.ones((), dtype=torch.bool, device=left.device)
        stats = self.last_cache_stats
        # Pointers are valid ONLY inside this infer: support and graph tensors
        # remain alive throughout. Measurement/variance addresses distinguish
        # edges even if a caller supplied aliased/expanded support point rows.
        pair_key = (left.data_ptr(), right.data_ptr(), left.stride(), right.stride(),
                    z.data_ptr(), variance.data_ptr())
        for i in range(0, p, chunk):
            for j in range(0, p, chunk):
                key = (*pair_key, i, j)
                kernel = self._kernel_cache.get(key)
                elements = min(chunk, p-i) * min(chunk, p-j)
                stats['logical_pair_distances'] += elements
                if kernel is None:
                    stats['cache_miss_blocks'] += 1
                    stats['computed_pair_distances'] += elements
                    with self._phase('bp/likelihood'):
                        kernel = distance_2d(left[i:i+chunk], right[j:j+chunk])
                        kernel.sub_(z).square_().mul_(-0.5 / variance)
                    payload = kernel.numel() * kernel.element_size()
                    if stats['retained_payload_peak_bytes'] + payload <= self.cache_bytes:
                        self._kernel_cache[key] = kernel
                        stats['retained_payload_peak_bytes'] += payload
                    else:
                        stats['uncached_blocks'] += 1
                else:
                    stats['cache_hit_blocks'] += 1
                with self._phase('bp/likelihood_check'):
                    finite_kernel &= torch.isfinite(kernel).all()
                with self._phase('bp/message_reduction'):
                    # Exactly the original log-domain block reductions. Cached
                    # kernel is read-only; cavity messages still change per round.
                    right_block = torch.logsumexp(kernel + left_cavity[i:i+chunk, None], dim=0)
                    left_block = torch.logsumexp(kernel + right_cavity[None, j:j+chunk], dim=1)
                    to_right[j:j+chunk] = torch.logaddexp(to_right[j:j+chunk], right_block)
                    to_left[i:i+chunk] = torch.logaddexp(to_left[i:i+chunk], left_block)
        if not finite_kernel:
            raise FloatingPointError('measurement log likelihood is not finite; use float64/check input scales')
        return _normalise(to_right), _normalise(to_left)
