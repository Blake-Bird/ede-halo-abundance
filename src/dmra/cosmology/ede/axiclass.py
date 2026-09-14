"""AxiCLASS adapter using the shared execution and provenance boundary."""

from dmra.cosmology.ede.runner import CLASSRunner


class AxiCLASSSolver(CLASSRunner):
    """Run AxiCLASS and preserve its native output and execution manifest."""

    name = "AxiCLASS"
    checkout = "AxiCLASS"
    command_name = "class_axiclass"
