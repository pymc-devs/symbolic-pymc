import numpy as np
import theano
import theano.tensor as tt

from symbolic_pymc.theano.opt import ScanArgs
from symbolic_pymc.theano.random_variables import CategoricalRV, DirichletRV, NormalRV


def create_test_hmm():
    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_init_state = rng_state.get_state()
    rng_tt = theano.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    N_tt = tt.iscalar("N")
    N_tt.tag.test_value = 10
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2

    mus_tt = tt.matrix("mus")
    mus_tt.tag.test_value = np.stack([np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1).astype(
        theano.config.floatX
    )

    sigmas_tt = tt.ones((N_tt,))
    sigmas_tt.name = "sigmas"

    pi_0_rv = DirichletRV(tt.ones((M_tt,)), rng=rng_tt, name="pi_0")
    Gamma_rv = DirichletRV(tt.ones((M_tt, M_tt)), rng=rng_tt, name="Gamma")

    S_0_rv = CategoricalRV(pi_0_rv, rng=rng_tt, name="S_0")

    def scan_fn(mus_t, sigma_t, S_tm1, Gamma_t, rng):
        S_t = CategoricalRV(Gamma_t[S_tm1], rng=rng, name="S_t")
        Y_t = NormalRV(mus_t[S_t], sigma_t, rng=rng, name="Y_t")
        return S_t, Y_t

    (S_rv, Y_rv), scan_updates = theano.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv, rng_tt],
        outputs_info=[{"initial": S_0_rv, "taps": [-1]}, {}],
        strict=True,
        name="scan_rv",
    )
    Y_rv.name = "Y_rv"

    scan_op = Y_rv.owner.op
    scan_args = ScanArgs.from_node(Y_rv.owner)

    Gamma_in = scan_args.inner_in_non_seqs[0]
    Y_t = scan_args.inner_out_nit_sot[0]
    mus_t = scan_args.inner_in_seqs[0]
    sigmas_t = scan_args.inner_in_seqs[1]
    S_t = scan_args.inner_out_sit_sot[0]
    rng_in = scan_args.inner_out_shared[0]

    rng_updates = scan_updates[rng_tt]
    rng_updates.name = "rng_updates"
    mus_in = Y_rv.owner.inputs[1]
    mus_in.name = "mus_in"
    sigmas_in = Y_rv.owner.inputs[2]
    sigmas_in.name = "sigmas_in"

    # The output `S_rv` is really `S_rv[1:]`, so we have to extract the actual
    # `Scan` output: `S_rv`.
    S_in = S_rv.owner.inputs[0]
    S_in.name = "S_in"

    return locals()
