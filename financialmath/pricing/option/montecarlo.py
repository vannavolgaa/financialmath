from dataclasses import dataclass
import time
import numpy as np
from typing import NamedTuple
from financialmath.instruments.option import Option 
from financialmath.tools.tool import MainTool
from financialmath.model.blackscholes.montecarlo import (
    MonteCarloBlackScholes,
    MonteCarloBlackScholesInput
    )
from financialmath.pricing.numericalpricing.option.montecarlo import (
    MonteCarloGreeks,
    MonteCarloLeastSquare,
    MonteCarloLookback, 
    MonteCarloPricing
    )
from financialmath.pricing.schemas import (
    OptionGreeks, 
    OptionValuationResult
    )

class MonteCarloBlackScholesParameters(NamedTuple):
    pass

@dataclass
class MonteCarloBlackScholesValuation: 

    option : Option 
    parameters : MonteCarloBlackScholesParameters
    first_order_greek : bool = True
    second_order_greek : bool = True
    third_order_greek : bool = True
    max_workers : int = 4