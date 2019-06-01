from collections.abc import Mapping


def assert_ops_equal(a, b, compare_fn=lambda a, b: a.op.type == b.op.type):
    if hasattr(a, 'op') or hasattr(b, 'op'):
        assert hasattr(a, 'op') and hasattr(b, 'op')

        assert compare_fn(a, b)

        assert len(a.op.inputs) == len(b.op.inputs)

        if isinstance(a.op, Mapping):
            a_inputs = list(a.op.inputs.values())
        else:
            a_inputs = list(a.op.inputs)

        if isinstance(b.op, Mapping):
            b_inputs = list(b.op.inputs.values())
        else:
            b_inputs = list(b.op.inputs)

        for i_a, i_b in zip(a_inputs, b_inputs):
            assert_ops_equal(i_a, i_b)
