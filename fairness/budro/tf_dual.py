"""Drop-in replacement for BuDRO's scripts/tf_dual.py -- the "dual" solver path,
which we adapt to work with large datasets.

budro._install_patched_tf_dual() binds this module to the name `tf_dual`, so BuDRO's
tf_xgboost_log.boost_dual calls into it unchanged. The entry point is find_wts().


WHAT WE CHANGED vs BuDRO's original tf_dual (details in the sections below):
  1. eta-finding: BuDRO's find_eta root-finds eta on a DENSE n x n C, so it does not scale.
     We keep the root-find but make it MATRIX-FREE (LazyCost + tiled argmax) and warm-started
     across boosting rounds, so it runs at large n.
  2. weight-finding: BuDRO's find_wts builds the EXACT transport plan (fractional split of one
     boundary point). We use INTEGRAL weights instead (one argmax target per point) -- BuDRO's
     own Pi-hat approximation (Algorithm 7), the same weights dual_sgd.py generates.


The dual variable eta
---------------------
Each boosting round, the adversary transports every training point subject to one constraint:
the mean distance moved equals the budget eps. The dual of that transport problem has a single
scalar variable eta -- a price per unit distance -- under which each point picks its target
greedily:

    j*(i) = argmax_j [ loss_{y_i}(j) - eta * C[i, j] ].

eta is determined by eps: it is the price at which the adversary spends exactly its budget,

    mean_dist(eta*) = eps,   where   mean_dist(eta) := (1/n) sum_i C[i, j*(i)].

mean_dist is monotone non-increasing in eta (a higher price keeps points at home), so eta* is
the root of a monotone scalar function.

Solving for eta
---------------
BuDRO's original tf_dual.find_eta already root-finds this (scipy root_scalar where
mean_dist(eta) = eps), but it operates on a DENSE n x n cost matrix, so it does not fit at large
n. We keep the root-find and make it scale:

  * That pass never builds the n x n cost matrix: it computes the argmax matrix-free, in cache-
    sized tiles, straight from the projected points (see _fused_max_inds). This is the change
    that lets the dual path run at large n.
  * Each probe -- one such matrix-free pass -- yields both the transport plan and the spend it
    costs, so the plan we owe every round comes for free from the search (see _probe).
  * eta* drifts only slowly between rounds, so we predict it from the previous rounds and correct
    with a few steps, rather than searching from scratch (see _predict, solve_eta).

solve_eta returns a price that is within budget (mean_dist <= eps): the adversary may underspend by
up to BUDGET_TOL but never overspends. The spend is returned every round, and it is what certifies a
run really used the eps it claims.

Building the weights
--------------------
BuDRO's original tf_dual.find_wts builds the EXACT transport plan, 
splitting one boundary point's mass fractionally so mean_dist == eps exactly
(Vargo et al., App. B.3). We keep the INTEGRAL weights (one argmax target per point, weights k/n)
-- BuDRO's own approximation Pi-hat and Algorithm 7. So this is boost_dual's ALGORITHM (root-find)
with boost_dual_sgd's WEIGHTS (integral): the same weights BuDRO's dual_sgd.dual builds from a plan.
"""

import time

import numpy as np

_IS_OUR_PATCH = True

# We accept the first eta whose plan spends in [(1 - BUDGET_TOL)*eps, eps]:
# within tolerance of the budget, never above it.
BUDGET_TOL = 0.05

# ---- per-run state  --------------------------------------------------------
_eta_prev = None       # eta* from the previous boosting round
_eta_prev2 = None      # ...and the one before, for extrapolating the drift
_probes = []           # full argmax passes spent per round
_spent = []            # mean_dist actually spent per round -> eff_eps


def reset_state():
    """Reset per-run state. Call once before each BuDRO run so eta is not warm-started across
    runs and the logs start empty."""
    global _eta_prev, _eta_prev2, _probes, _spent
    _eta_prev = None
    _eta_prev2 = None
    _probes = []
    _spent = []


def probe_log():
    """Full argmax passes spent per boosting round."""
    return list(_probes)


def spend_log():
    """mean_dist actually spent per round -> eff_eps"""
    return list(_spent)


# Tile sizes for the block-by-block argmax in _fused_max_inds. We never build the whole n x n score
# matrix; we compute it in small ROW_BLOCK x COL_TILE pieces and keep only a running best per row.
# Small tiles are the whole point: each one fits in fast CPU cache. Build the full matrix instead
# and the pass spends its time shuffling that matrix to and from slow RAM -- fine for one fit, a
# real bottleneck when several share a machine.
# Tuning: keep one tile (ROW_BLOCK * COL_TILE * 4 bytes) within L2 cache; raise ROW_BLOCK for very large n.
ROW_BLOCK = 1024
COL_TILE = 512


def _fused_max_inds(a, lazy, l0, l1, ya):
    """For every point i, pick its best target j = argmax_j [ l_{y_i}(j) - eta*C[i,j] ].
    Returns one target index per point. 

    This is the matrix-free version of BuDRO's argmax (dual_inds in dual_sgd.py, dual_inds_tf in
    tf_dual.py): the same formula, but while the original forms the whole n x n score matrix
    L - eta*C (thus needing a dense C in memory) we never build C at all. The score of target j for
    point i is an inner product of two short (length d+1) vectors, so one matrix-multiply of all
    points against all targets gives a block of scores and we take each row's argmax. We stream
    that product in ROW_BLOCK x COL_TILE tiles, so the full score matrix never exists at once.

    Detail -- the inner-product form. The score is l_{y_i}(j) - eta*C[i,j]. 
    With C[i,j] = scale*(sq_i + sq_j - 2 p_i.p_j)  (sq_i = ||p_i||^2) and a = eta*scale, that is
        l(j) - a*sq_i - a*sq_j + 2a*(p_i.p_j).
    The -a*sq_i term is the same for every target j, so it drops out of the argmax, leaving
        argmax_j [ l(j) - a*sq_j + 2a*(p_i.p_j) ] = argmax_j  [p_i, 1] . [2a*p_j, l(j) - a*sq_j].

    Detail -- why float32 is safe. This form ranks targets by the inner product p_i.p_j and never
    evaluates sq_i + sq_j - 2 p_i.p_j, so the cancellation that forces float64 for C's actual
    values cannot arise. Only the ordering of scores matters here, and it matches float64 exactly.
    """
    n = lazy.n
    out = np.empty(n, dtype=np.int64)                   # winning target index per point

    if a == 0.0:
        # Special case: price 0 (eta = 0). With no cost term the score is just the loss l(j), so
        # every source goes to the single highest-loss target of its label -- argmax(l0) for label-0
        # sources, argmax(l1) for label-1. 
        out[:] = int(np.argmax(l0))            # default: everyone -> highest-loss point under l0
        out[ya == 1] = int(np.argmax(l1))      # label-1 sources -> highest-loss point under l1
        return out

    # p_i = point i's feature vector in the projected fair-metric space (row i of lazy.P32);
    # sq_i = ||p_i||^2. For a source i, the score of target j is a dot product (docstring has the
    # algebra):
    #     score(i, j) ~ [p_i, 1] . [2a*p_j, l(j) - a*sq_j].
    # So if we stack every source into a matrix A (row i = [p_i, 1]) and every target into a matrix
    # B (row j = [2a*p_j, l(j) - a*sq_j]), then A @ B.T is the whole score matrix and the argmax of
    # each row is that source's best target. We compute it tile by tile so the n x n matrix is never
    # actually stored.

    # A -- the source half of the dot product. Row i is [p_i, 1]; the trailing 1 multiplies the
    # target's l(j) - a*sq_j entry. Built once, shared by both labels.
    P32 = lazy.P32
    d = P32.shape[1]
    A = np.empty((n, d + 1), dtype=np.float32)
    A[:, :d] = P32
    A[:, d] = 1.0

    # Split sources by their own label: the loss uses it (l0 for y=0, l1 for y=1), so each label
    # needs its own B. rid = indices of the sources with the current label.
    for lab, ll in ((0, l0), (1, l1)):
        rid = np.nonzero(ya == lab)[0]
        if rid.size == 0:
            continue

        # B -- the target half of the dot product. Row j is [2a*p_j, l(j) - a*sq_j]; u is that last
        # entry. Subtracting u's mean shifts every score equally, so it changes no argmax, but keeps
        # the numbers small enough to be safe in float32.
        u = ll - a * lazy.sq                            # float64
        u -= u.mean()
        B = np.empty((n, d + 1), dtype=np.float32)
        B[:, :d] = P32 * np.float32(2.0 * a)            # 2a*p_j
        B[:, d] = u                                     # l(j) - a*sq_j, mean-shifted
        Bt = np.ascontiguousarray(B.T)                  # (d+1, n), so A @ Bt is the score matrix

        # The core: a streaming argmax. We want each source's best target over ALL targets, but
        # without ever holding a whole score row. So we walk the targets one tile at a time and, per
        # source, keep the best score seen so far (best) and the target that gave it (bidx); after
        # the last tile, bidx IS the argmax. Outer loop = blocks of sources; inner = tiles of targets.
        for r0 in range(0, rid.size, ROW_BLOCK):
            blk = rid[r0:r0 + ROW_BLOCK]                # this block of source rows
            Ab = A[blk]
            best = np.full(blk.size, -np.inf, dtype=np.float32)   # best score seen so far, per source
            bidx = np.zeros(blk.size, dtype=np.int64)             # target index that gave it
            rows = np.arange(blk.size)
            for c0 in range(0, n, COL_TILE):
                S = Ab @ Bt[:, c0:c0 + COL_TILE]        # scores of these sources vs this tile of targets
                # NOTE: BuDRO breaks argmax ties RANDOMLY (Appendix B.3); 
                # instead, np.argmax takes the lowest index.
                # However, original source code uses argmax.
                j = np.argmax(S, axis=1)                # best target within the tile, per source
                v = S[rows, j]                          # and its score
                hit = v > best                          # sources this tile improved on
                best[hit] = v[hit]
                bidx[hit] = j[hit] + c0                 # +c0: tile-local index -> global target index
            out[blk] = bidx                             # winning target per source in this block
    return out


def _probe(eta, C, l0, l1, ya, n):
    """Probe one candidate price. Returns (transport plan, average distance it spends).

    This is the single pass BuDRO's root-find makes to evaluate a price (f_grad/f_grad_tf in
    tf_dual.py: run the argmax, then reduce to the mean distance). 
    
    We return the plan and mean_dist: the plan feeds the transport weights, 
    and mean_dist is what solve_eta compares to the budget eps to tell whether the price is too high or too low.
    """
    inds = _fused_max_inds(float(eta) * C.scale, C, l0, l1, ya)   # a = price x cost-scale (scale normally 1)
    dist = np.asarray(C[inds, np.arange(n)], dtype=np.float64)    # C[...] already applies scale
    return inds, float(dist.mean())


# -------------------------------------------------------------------------------------------
# eta* is the price at which the average distance moved equals the budget eps --
# that is, the value of eta for which mean_dist(eta) = eps.
#
# The original dual_sgd looks for eta* with gradient descent (SGD). We found that unreliable
# here: the search keeps wandering around eta* without ever settling on it, so the adversary
# moves points by the wrong amount -- spending a budget different from the eps we asked for.
#
# Instead we solve mean_dist(eta) = eps directly (which is what BuDRO's other solver,
# tf_dual.find_eta, already does). To keep it fast on large datasets (the authors had this root-find
# but abandoned it at scale as too slow), we avoid searching from scratch
# every round: eta* changes only a little between boosting rounds, so we guess it from
# the previous rounds (_predict) and then refine that guess with a few correction steps.
#
# The one expensive operation is a "probe": testing a single candidate eta costs one full pass
# over all pairs of points (_probe). solve_eta is written to reach eta* in as few probes as
# possible.
# ---------------------------------------------------------------------------------------------

ALPHA0 = 1.35          # initial steepness of the price->spend curve; refined per round (see Stage 2)
MAX_NEWTON = 8         # Stage-2 refine steps before falling back to bisection
MAX_PROBES = 40        # hard cap on full argmax passes per round, across all stages

def solve_eta(C, l0, l1, ya, eps, n, eta0=None, tol=None):
    """Find eta* solving mean_dist(eta) = eps. Returns (eta, inds, mean_dist, n_probes):

        eta        the price found (>= 0).
        inds       the transport plan at that eta -- one target per point -- so the caller
                   needs no extra pass to build the weights.
        mean_dist  the average distance that plan actually spends.
        n_probes   how many full argmax passes it took (the cost of the call).

    The returned eta never overspends: mean_dist(eta) <= eps always. It also lands within `tol`
    of eps, with one exception. mean_dist changes in discrete jumps -- the plan assigns each point
    to a whole target, so nudging eta does nothing until some point flips to a new target -- so
    occasionally no eta spends inside the [(1 - tol)*eps, eps] band at all. Then we return the
    closest eta that still does not overspend.

    The search runs in three stages, each reusing the range the earlier ones have narrowed:
        1. start from a predicted eta (extrapolated from earlier rounds, passed in as eta0);
        2. refine it with a few quick correction steps;
        3. if those miss the band, fall back to repeatedly halving the range (bisection).

    tol defaults to BUDGET_TOL. We read it here rather than in the signature so that changing the
    module constant takes effect (a signature default would bind once, at import).
    """
    if tol is None:
        tol = BUDGET_TOL
    count = [0]

    # We keep the search narrowed to a bracket of two prices; eta* sits between them:
    #   lo -- a price that still OVERSPENDS (mean_dist > eps): too cheap, points move too far.
    #   hi -- a price that is WITHIN budget (mean_dist <= eps): expensive enough to hold the line.
    # `hi` is what we return -- it never overspends -- together with its plan (hiI) and spend (hiD).
    st = {"lo": None, "loD": None, "hi": None, "hiD": None, "hiI": None}

    def probe(e):
        # One trial at price e. File the result as the new lo or hi, keeping the bracket tightest.
        count[0] += 1
        inds, D = _probe(e, C, l0, l1, ya, n)
        if D > eps:                              # overspent -> raise the lower (too-cheap) end
            if st["lo"] is None or e > st["lo"]:
                st["lo"], st["loD"] = e, D
        else:                                    # within budget -> lower the upper (feasible) end
            if st["hi"] is None or e < st["hi"]:
                st["hi"], st["hiD"], st["hiI"] = e, D, inds
        return inds, D

    def ok(D):
        # Close enough? Inside the accept band [(1 - tol)*eps, eps], and never over budget.
        return (1.0 - tol) * eps <= D <= eps

    # Is the budget even binding? Price 0 means distance is free. 
    # If the adversary underspends even then, no price is needed and eta* = 0.
    inds0, D0 = probe(0.0)
    if D0 <= eps:
        return 0.0, inds0, D0, count[0]

    # Stage 1: start from the predicted price (extrapolated from earlier rounds),  
    #          or 0.1 at first (from the original dual_sgd). 
    e = float(eta0) if (eta0 is not None and eta0 > 0.0) else 0.1
    inds, D = probe(e)
    if ok(D):
        return e, inds, D, count[0]         # prediction already in tolerance: no search needed

    # ---- Stage 2: refine the guess ----
    # Raising the price lowers the spend in a predictable way: roughly spend ~ 1 / price^alpha,
    # where alpha is the steepness of that curve (seeded at ALPHA0 above).
    # Knowing that shape, a probe tells us more than "too high" or "too low": from how far this
    # price over- or underspent, we can estimate the price that would hit eps and jump to it,
    #     new_price = price * (spend / eps) ** (1 / alpha)
    # overspent (spend > eps) -> factor > 1 -> raise the price; underspent -> lower it.
    prev = (e, D)
    alpha = ALPHA0
    # Each pass makes one such correction (a Newton step) then re-probes.
    # At most MAX_NEWTON of them; if none lands in tolerance we fall back to bisection (Stage 3).
    for _ in range(MAX_NEWTON):
        if count[0] >= MAX_PROBES:          # hard cap on full argmax passes per round, across all stages
            break
        e_last, D_last = prev
        if D_last <= 0.0:
            e_new = e_last * 0.25           # nothing transported: no scale to extrapolate from
        else:
            e_new = e_last * (D_last / eps) ** (1.0 / alpha)   # one-step jump toward eta*
        # The jump is only an estimate and can overshoot. 
        # --> Clamp it back to the range we already know holds eta* i.e., (lo,hi).
        lo, hi = st["lo"], st["hi"]
        if lo is not None and hi is not None:
            if not (lo < e_new < hi):                   # jumped outside the bracket -> use a point inside it
                e_new = np.sqrt(max(lo, hi * 1e-9) * hi)
            # else: already inside (lo, hi) -> keep the Newton estimate
        elif hi is not None:
            e_new = min(e_new, hi * 0.9)                # only the feasible end known -> stay just below it
        elif lo is not None:
            e_new = max(e_new, lo * 1.1)                # only the too-cheap end known -> stay just above it
        if not np.isfinite(e_new) or e_new <= 0.0:      # jump blew up (inf/nan/<=0) -> give up, go bisect
            break
        # Probe the new price and see if it is within tolerance
        i2, D2 = probe(e_new)
        if ok(D2):
            return e_new, i2, D2, count[0]
        # Re-measure alpha (the slope of spend vs price) from the last two probes.
        # Clamp to [0.2, 5] against noise.
        if D2 > 0 and D_last > 0 and D2 != D_last and e_new != e_last:
            a = -(np.log(D2) - np.log(D_last)) / (np.log(e_new) - np.log(e_last))
            if 0.2 < a < 5.0:
                alpha = a
        prev = (e_new, D2)

    # ---- Stage 3: fall back to bisection (Newton did not reach tolerance within MAX_NEWTON) ----
    # Bisection repeatedly halves a price range known to contain eta*. The cheap end is always known
    # (lo: price 0 overspent, or we'd have returned earlier). If Newton never found a within-budget
    # price (no hi yet), raise the price until one appears, then halve the range down.
    if st["hi"] is None:                    # only overspending prices seen -> raise price to find hi
        p = max(st["lo"], 1e-12) * 2.0      # start just above the highest price known to overspend
        while count[0] < MAX_PROBES:
            _, D2 = probe(p)
            if ok(D2):
                return p, st["hiI"], D2, count[0]   # within tolerance -> done
            if D2 <= eps:
                break                       # within budget: hi is now set -> go bisect
            p *= 2.0                        # still overspending -> double and retry
        if st["hi"] is None:                # never within budget (mean_dist -> 0 as price -> inf):
            return st["lo"], inds, st["loD"], count[0]   # unreachable in practice; best effort

    # ---- Stage 3, bisection: halve the range [lo, hi] until a probe lands within tolerance ----
    while count[0] < MAX_PROBES:
        lo, hi = st["lo"], st["hi"]
        m = 0.5 * (lo + hi)                 # midpoint price
        if not (lo < m < hi):               # range too tight to split further -> stop
            break
        _, D2 = probe(m)                    # probe updates lo or hi, shrinking the range
        if ok(D2):
            break                           # within tolerance -> done

    # Tolerance can be unreachable: mean_dist moves in discrete steps, so it may jump across the
    # whole accept band between two prices -- e.g. it steps across eps right at eta=0 (overspends at
    # price 0, within budget for every positive price), so no positive price lands in the band.
    # Either way st["hi"] is the feasible side; we return it and the caller reports the budget spent.
    return st["hi"], st["hiI"], st["hiD"], count[0]


# -----------------------------------------------------------------------------
# Entry point, called once per boosting round by tf_xgboost_log.boost_dual.
# -----------------------------------------------------------------------------

def _predict():
    """Predicted eta for the current round, or None on the first round.

    eta* rises across rounds as boosting shrinks the losses, so the previous round's value is
    biased low; using it unadjusted lands outside BUDGET_TOL on most rounds. We extrapolate the
    trend from the last two rounds, clamping the step ratio to [0.5, 2] so a single anomalous
    round cannot throw the prediction far off.
    """
    if _eta_prev is None or _eta_prev <= 0.0:
        return None
    if _eta_prev2 is None or _eta_prev2 <= 0.0:
        return _eta_prev
    ratio = min(max(_eta_prev / _eta_prev2, 0.5), 2.0)
    return _eta_prev * ratio


def find_wts(loss0, loss1, y, C, eps,
             lowg=0.0, highg=10.0, method="brentq",
             roottol=1e-8, dtol=1e-25,
             dtype=None, idtype=None,
             debug=False, verbose=False):
    """Root-find eta and return INTEGRAL transport weights -- the entry point that
    tf_xgboost_log.boost_dual expects (replacing BuDRO's exact-split find_wts).

    Signature matches BuDRO's tf_dual.find_wts so boost_dual's keyword call is unchanged. The
    root-bracket / tolerance knobs (lowg, highg, method, roottol, dtol) and dtype/idtype are
    accepted for compatibility and IGNORED: solve_eta brackets and tolerances the root itself
    (BUDGET_TOL), matrix-free.

    Returns (wts0, wts1, sol, xport_map, split_data), matching BuDRO's find_wts:
        wts0, wts1  integral transport weights over the augmented support -- the only values
                    boost_dual uses (xgb sample weights, and max_loss). Identical to what
                    dual_sgd.dual returns for the same inputs, so the trained model is the same.
        sol         the root eta as a plain float (boost_dual ignores this slot).
        xport_map   one target index per point (the transport plan).
        split_data  always None -- we do NOT do the fractional split (integral weights).
    """
    global _eta_prev, _eta_prev2

    if not getattr(C, "_IS_LAZY_COST", False):
        raise TypeError("find_wts() needs a budro.LazyCost (the dual path is matrix-free)")

    n = int(C.n)
    ya = np.asarray(y).reshape(-1)
    l0 = np.asarray(loss0, dtype=np.float64).reshape(-1)
    l1 = np.asarray(loss1, dtype=np.float64).reshape(-1)
    epsf = float(np.asarray(eps))

    # The core algorithm: root-find eta so mean_dist(eta) = eps, matrix-free, warm-started from the
    # previous rounds. solve_eta returns the plan at eta* for free (no extra pass).
    t0 = time.time()
    eta, max_inds, D, probes = solve_eta(C, l0, l1, ya, epsf, n, eta0=_predict())

    _eta_prev2, _eta_prev = _eta_prev, eta
    _probes.append(probes)
    _spent.append(D)

    if verbose:
        print(f"***Dual: eta={eta:.6g}  spent={D:.6g}/{epsf:g} ({D / epsf:.1%})  "
              f"{probes} probes, {time.time() - t0:.1f}s", flush=True)

    # Integral transport weights: how often each point is chosen as a target, per source label
    # (identical to dual_sgd.dual). This is Algorithm 7's Pi-hat.
    wts0 = np.bincount(max_inds[ya == 0], minlength=n).astype(np.float64) / n
    wts1 = np.bincount(max_inds[ya == 1], minlength=n).astype(np.float64) / n

    return wts0, wts1, eta, max_inds, None
