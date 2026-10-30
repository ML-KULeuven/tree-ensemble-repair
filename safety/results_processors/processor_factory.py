from results_processors.processor import Processor

def create_results_processor(exp_mode: str, **kwargs) -> Processor:
    """
    Factory to create the correct fault fixing class
    
    :param exp_mode: What experiment do we need the results for?
    :type exp_mode: str
    :param kwargs: arguments to initialize the processor
    :return: A class which is responsible for writing and parsing results
    :rtype: Processor
    """
    if exp_mode == "repair":
        from results_processors.repair_loop import RepairLoopResults
        return RepairLoopResults(**kwargs)
    
    elif exp_mode == "count_faults":
        from results_processors.count_faults import CountFaultsResults
        return CountFaultsResults(**kwargs)
    elif exp_mode == "percent_fixed":
        from results_processors.percent_fixed import PercentFixed
        return PercentFixed(**kwargs)
    else:
        raise ValueError(
            f"Unknown experiment mode: {exp_mode}. "
            f"Available: ['repair', 'count_faults', 'percent_fixed']"
        )