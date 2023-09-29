from instruments.option import CreateOption, OptionPayoffList

opt = CreateOption.european_vanilla_call(strike=100, expiry=1)

opt.payoff_name()

