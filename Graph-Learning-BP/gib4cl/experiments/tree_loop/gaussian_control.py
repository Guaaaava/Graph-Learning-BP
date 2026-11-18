"""Analytic auxiliary control, explicitly separate from nonlinear particle BP."""
from __future__ import annotations

import math
import numpy as np


def controlled_rings(lengths=range(3,9), unary_precision=1., edge_precisions=(1.,4.)):
    """Fixed local precision/coupling, degree two; covariance by dense inversion.

    Model: p(x) ∝ exp[-a/2 sum_i x_i² - b/2 sum_edges (x_i-x_j)²].
    Gaussian BP's symmetric precision message is iterated independently of
    the dense exact solver. This is NOT a two-dimensional ranging result.
    """
    a = float(unary_precision)
    if not math.isfinite(a) or a <= 0:
        raise ValueError('unary_precision must be finite and positive')
    rows = []
    for b in edge_precisions:
        b = float(b)
        if not math.isfinite(b) or b <= 0:
            raise ValueError('edge_precisions must be finite and positive')
        message = 0.
        for iteration in range(1,10001):
            updated = b*(a+message)/(a+b+message)
            if abs(updated-message) <= 1e-13:
                message = updated
                break
            message = updated
        else:
            raise ArithmeticError('Gaussian precision iteration did not converge')
        bp_variance = 1/(a+2*message)
        rho = 2*b/(a+2*b+math.sqrt(a*a+4*a*b))
        for length in lengths:
            if isinstance(length,bool) or not isinstance(length,int) or length < 3:
                raise ValueError('Ring lengths must be integers >= 3')
            precision = a*np.eye(length)
            for u in range(length):
                v = (u+1)%length
                precision[u,u] += b
                precision[v,v] += b
                precision[u,v] -= b
                precision[v,u] -= b
            exact_variance = float(np.linalg.inv(precision)[0,0])
            rows.append({'ring_length':length, 'unary_precision':a, 'edge_precision':b,
                         'exact_variance':exact_variance, 'gaussian_bp_variance':bp_variance,
                         'variance_ratio':bp_variance/exact_variance,
                         'closed_form_ratio':(1-rho**length)/(1+rho**length),
                         'message_iterations':iteration})
    return {'scope':'Auxiliary scalar Gaussian model; not the nonlinear two-dimensional particle experiment.',
            'controls':'Unary precision and per-edge coupling fixed within each curve; every node has degree two.',
            'rows':rows}
