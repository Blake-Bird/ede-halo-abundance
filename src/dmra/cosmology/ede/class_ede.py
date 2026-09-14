"""CLASS_EDE adapter using the shared execution and provenance boundary."""

from dmra.cosmology.ede.runner import CLASSRunner


class CLASSEDESolver(CLASSRunner):
    """Run CLASS_EDE and preserve its native output and execution manifest."""

    name = "CLASS_EDE"
    checkout = "class_ede"
    command_name = "class_ede"
