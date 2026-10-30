from fault_fixing_algorithms.fault_fixer import FaultFixer

def create_fault_fixer(repair_method: str, **kwargs) -> FaultFixer:
    method = repair_method.split("_", 1)[0]
    if method == "milp-l1":
        from fault_fixing_algorithms.milp_l1 import MILPL1Fixer
        return MILPL1Fixer(**kwargs)

    elif method == "penalty":
        from fault_fixing_algorithms.penalty_tree import PenaltyTreeFixer
        return PenaltyTreeFixer(**kwargs)
    
    elif method == "retraining-loss":
        from fault_fixing_algorithms.fa_loss import FALossFixer
        return FALossFixer(**kwargs)
    
    elif method == "retraining-augment":
        from fault_fixing_algorithms.model_hardening import ModelHardeningFixer
        return ModelHardeningFixer(**kwargs)
    
    elif method == "milp":
        from fault_fixing_algorithms.milp import MILPFixer
        return MILPFixer(**kwargs)
    else:
        raise ValueError(
            f"Unknown fixer type: {repair_method}. "
            f"Available: ['milp-l1', 'retraining-loss', 'retraining-augment', 'milp', 'penalty']"
        )
