
def get_runner_cls(algorithm):
    if algorithm == "nfsp":
        from algorithms import RunNFSP

        return RunNFSP

    if algorithm == "escher":
        from algorithms import RunEscherParallel

        return RunEscherParallel

    if algorithm == "psro":
        from algorithms import RunPSRO

        return RunPSRO

    if algorithm == "rnad":
        from algorithms import RunRNaD

        return RunRNaD

    if algorithm == "ppo":
        from algorithms import RunPPO

        return RunPPO

    if algorithm == "mmd":
        from algorithms import RunMMD

        return RunMMD

    if algorithm == "ppg":
        from algorithms import RunPPG

        return RunPPG

    raise ValueError(f'Unrecognized algorithm: {algorithm}')
