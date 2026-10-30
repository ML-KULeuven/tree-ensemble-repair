"""
budro.py -- BuDRO baseline (Vargo, Zhang, Yurochkin & Sun, "Individually Fair Gradient
Boosting", ICLR 2021;  https://github.com/UMich-FATML/BuDRO).

Individual-fairness-aware gradient boosting via distributionally robust optimization: 
at every boosting round the training distribution is optimally transported within a Wasserstein
ball defined by a *fair metric* C[i, j] (a cost matrix over pairs of training individuals),
making the model robust to moving mass along directions the fair metric considers "cheap".

This module holds everything our code needs to run BuDRO, for both of its solver paths.
The only thing kept separate is budro/tf_dual.py -- our drop-in replacement for BuDRO's own
scripts/tf_dual.py, loaded by path and patched into the name `tf_dual` (see import_budro).

TWO SOLVER PATHS (which one a dataset uses is decided by n_train; see budro/tuning.py)
--------------------------------------------------------------------------------------------
  SINKHORN (small n)   fair_metric()       dense n x n cost matrix + entropic Sinkhorn
                                           (BuDRO's boost_sinkhorn_tf, unchanged).
                                           O(n^2) SPACE -- caps out around n = 10^4.
  DUAL     (large n)   fair_metric_lazy()  matrix-free LazyCost + root-find eta
                                           (BuDRO's boost_dual loop + budro/tf_dual.py).
                                           O(n*d) space; runs at any n.

Both paths use the IDENTICAL fair metric -- the squared-Euclidean distance on data projected
out of the sensitive subspace (see _project_out_sensitive).
They differ only in whether that distance matrix is materialised (Sinkhorn)
or rebuilt in blocks on demand (dual, our patch).

The dual path runs BuDRO's own boost_dual loop; only its per-round weight step (find_wts) is
replaced by budro/tf_dual.py. There, eta is root-found as in BuDRO's find_eta, but matrix-free,
and the transport weights are INTEGRAL (one argmax target per point) instead of BuDRO's exact
fractional split -- the integral approximation Pi-hat of Algorithm 7 (Vargo et al., App. B.3).
Details in tf_dual.py.
"""

import os
import sys
import importlib.util

import numpy as np

# silence Tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# local clone of https://github.com/UMich-FATML/BuDRO
from paths import BUDRO_DIR
_SCRIPTS_DIR = os.path.join(BUDRO_DIR, "scripts")
_TF_DUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_dual.py")    # our patch


# ==========================================================================================
# Import plumbing 
# ==========================================================================================

def _scripts_on_path():
    """
    Put BuDRO's scripts/ on sys.path so `import script` resolves. 
    """
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)


def _install_patched_tf_dual():
    """
    Bind budro/tf_dual.py to the module name `tf_dual`.

    BuDRO's tf_xgboost_log.py does a bare `import tf_dual`. Because _scripts_on_path()
    puts their scripts/ at sys.path[0], that would find THEIR (unmodified) copy.
    We seed sys.modules first, so their import resolves to ours.
    """
    existing = sys.modules.get("tf_dual")
    if existing is not None and getattr(existing, "_IS_OUR_PATCH", False):
        return existing
    spec = importlib.util.spec_from_file_location("tf_dual", _TF_DUAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tf_dual"] = mod
    spec.loader.exec_module(mod)
    return mod


def import_budro():
    """ Loads (tf_xgboost_log, script, patched solver)

    Runs the DUAL path through tf_xgboost_log.boost_dual (root-find find_wts, budro/tf_dual.py).
    `patched` is the installed module (its reset_state/probe_log/BUDGET_TOL for the caller).
    """
    _scripts_on_path()
    patched = _install_patched_tf_dual()
    import tf_xgboost_log as txl
    import script as budro_script
    return txl, budro_script, patched


# ==========================================================================================
# Fair metric 
# ==========================================================================================

def _project_out_sensitive(X, ordinal_cols, use_proxy=True):
    """Project X onto the orthogonal complement of the sensitive subspace.

    BuDRO's SenSR fair metric is the squared-Euclidean distance on this projected data, so
    both solver paths build on projData. The subspace is span{proxy direction, sensitive
    axes}, where the proxy is learned by LogisticRegressionCV predicting S from the other
    features (S zeroed first so it can't cheat).

    NOTE: penalising the proxy means the metric targets a BROADER notion than the causal
    discrimination we verify with veritas (which only flips S and holds everything else fixed).
    """
    _scripts_on_path()
    import script as budro_script
    from sklearn.linear_model import LogisticRegressionCV

    sens = list(ordinal_cols)
    Xd = np.asarray(X, dtype=np.float64)

    S = Xd[:, sens[0]]
    if not np.isin(S, [0, 1]).all():
        raise ValueError("BuDRO fair metric requires a binary sensitive attribute (values in {0, 1}).")

    if use_proxy:
        XR = Xd.copy()
        XR[:, sens] = 0.0                          # zero S so the proxy can't read it directly
        coef = LogisticRegressionCV(Cs=100, cv=5, max_iter=5000).fit(XR, S).coef_[0]

    off = 1 if use_proxy else 0
    A = np.zeros((Xd.shape[1], off + len(sens)))   # columns = directions to remove
    if use_proxy:
        A[:, 0] = coef                             # proxy direction
    for k, c in enumerate(sens):
        A[c, off + k] = 1.0                        # sensitive axes

    proj = budro_script.proj_matrix_gen(Xd, A)     # I - A(A^T A)^{-1} A^T
    return Xd @ proj


# ==========================================================================================
# 1) SINKHORN PATH -- dense cost matrix (small n)
# ==========================================================================================

def fair_metric(X, ordinal_cols, use_proxy=True, dataset=None):
    """Dense n x n cost matrix C[i, j] = ||p_i - p_j||^2 (float32).

    O(n^2) SPACE, so use only for small n; see fair_metric_lazy for the matrix-free version.
    This is the C passed to tf_xgboost_log.boost_sinkhorn_tf.
    """
    from scipy.spatial.distance import pdist, squareform

    projData = _project_out_sensitive(X, ordinal_cols, use_proxy)
    C = squareform(pdist(projData, metric="sqeuclidean")).astype(np.float32)

    # HACK: DefaultCreditCardClients' heavy-tailed features blow C up, 
    #       making Sinkhorn underflow --> we median-normalise.
    if dataset == "DefaultCreditCardClients":
        C /= np.median(C[C > 0])
    return C


# ==========================================================================================
# 2) DUAL PATH -- matrix-free cost matrix (large n)
# ==========================================================================================
#
# The cost matrix C is O(n^2) SPACE: dense, it can be too large to even allocate, 
# which is the wall that keeps BuDRO's released code at small n 
# (it materialises C = squareform(pdist(X @ proj))).
#
# The paper anticipates the fix. Appendix B.2 notes that the dual/SGD step only ever touches
# column subsets of C, so "R and C may be stored anywhere (or even computed on-the-fly) ...
# can be used for essentially arbitrarily large data sets." Their code never does this, but the
# observation is sound: C is never *data*. It is the squared-Euclidean distance matrix of the
# LINEAR projection P = X @ proj, so it is fully determined by the n x d matrix P (d = 10-137
# here). LazyCost stores P and rebuilds C in row blocks on demand: e.g., for ACSEmploy 
# 192.5 GB -> 13.9 MB.
#
# This is a MEMORY fix, and only that -- it is what makes large n runnable at all. The O(n^2)
# TIME per round (the adversary maximises over every pair) is inherent to the method and
# untouched here; the runtime work is done separately, in budro/tf_dual.py.


class LazyCost:
    """C[i, j] = scale * ||P_i - P_j||^2, materialised only in row blocks.

    Substitutable for the dense ndarray BuDRO expects. Supported indexing:
        C[a:b]              -> (b-a, n) float32 block
        C[int_array]        -> (len, n) float32 block
        C[i_array, j_array] -> (len,)   float32 elementwise

    Blocks are computed in float64 and cast to float32, which is exactly what BuDRO does:
    their prep builds Corig in float64 via pdist, and tf.constant(Corig, tf.float32) casts it.
    Computing in float32 instead loses ~1e-5 to cancellation, which propagates through the eta
    search and changes the trained model (on MEPSPanel21 it moved test scores by 5.4e-2). So:
    float64, always -- except P32, used only by tf_dual's argmax (see that property).
    """

    _IS_LAZY_COST = True

    def __init__(self, P, scale=1.0):
        self.P = np.ascontiguousarray(P, dtype=np.float64)
        self.sq = np.einsum("ij,ij->i", self.P, self.P)
        self.n = self.P.shape[0]
        self.d = self.P.shape[1]
        self.shape = (self.n, self.n)
        self.dtype = np.dtype(np.float32)
        self.scale = float(scale)
        self._P32 = None

    def __len__(self):
        return self.n

    @property
    def P32(self):
        """float32 P, for tf_dual's fused argmax (_fused_max_inds) -- and for nothing else.

        Safe there and only there. That kernel ranks targets by the inner product p_i . p_j
        and never forms sq_i + sq_j - 2 p_i.p_j, so the cancellation that forces float64 in
        _rows() cannot arise; and it needs the ORDER of the scores, not their value. Verified
        on DutchCensus to reproduce the float64 argmax with 100% identical indices. Every path
        that returns a C VALUE -- _rows() and the C[i_arr, j_arr] gather that measures
        transported distance -- stays float64.
        """
        if self._P32 is None:
            self._P32 = np.ascontiguousarray(self.P, dtype=np.float32)
        return self._P32

    @property
    def nbytes(self):
        return self.P.nbytes + self.sq.nbytes

    def _rows(self, idx):
        out = self.P[idx] @ self.P.T             # (b, n) float64 -- only big alloc
        out *= -2.0
        out += self.sq[idx][:, None]
        out += self.sq[None, :]
        np.maximum(out, 0.0, out)                # cancellation can go slightly negative
        if self.scale != 1.0:
            out *= self.scale
        return out.astype(np.float32)

    def __getitem__(self, key):
        if isinstance(key, tuple):                       # C[i_arr, j_arr]
            i, j = key
            diff = self.P[i] - self.P[j]
            v = np.einsum("ij,ij->i", diff, diff)
            return (v * self.scale if self.scale != 1.0 else v).astype(np.float32)
        if isinstance(key, (int, np.integer)):
            return self._rows(np.asarray([key]))[0]
        return self._rows(key)

    def to_dense(self):
        """Full n x n materialisation. For validation on small n only."""
        if self.n > 20000:
            raise MemoryError(f"refusing to densify n={self.n} ({self.n**2 * 4 / 2**30:.1f} GB)")
        return self._rows(slice(None))

    def sample_median(self, k=4096, seed=6):
        """Median of positive pairwise costs, estimated on a random row sample.

        Used only to reproduce fair_metric's DefaultCreditCardClients rescale without touching
        O(n^2). NOTE: that rescale has no counterpart in the authors' code; it changes the
        effective eps. Off by default.
        """
        rng = np.random.default_rng(seed)
        k = min(k, self.n)
        idx = rng.choice(self.n, size=k, replace=False)
        sub = LazyCost(self.P[idx])
        blk = sub.to_dense()
        pos = blk[blk > 0]
        return float(np.median(pos))


def fair_metric_lazy(X, ordinal_cols, use_proxy=True, normalize=None, seed=6):
    """Same fair metric as fair_metric(), returned as a LazyCost (matrix-free).

    Identical projection -- the only difference is that we stop before squareform(pdist(...))
    and hand back P wrapped in a LazyCost instead of the dense C.

    normalize=None      -> C as defined (matches the authors)
    normalize="median"  -> C /= median(C > 0), estimated by sampling. Reproduces the
                           DefaultCreditCardClients hack in fair_metric().
    """
    projData = _project_out_sensitive(X, ordinal_cols, use_proxy)
    lazy = LazyCost(projData)
    if normalize == "median":
        lazy.scale = 1.0 / lazy.sample_median(seed=seed)
    elif normalize is not None:
        raise ValueError(f"unknown normalize={normalize!r}")
    return lazy
