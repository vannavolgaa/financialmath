from instruments.option import CreateOption, OptionPayoffList

opt = CreateOption.european_vanilla_call(strike=100, expiry=1)

opt.payoff_name()


listA = [1,2,3,4,5]
ListB = [True, False, False, True]
[a for a,b in zip(listA, ListB) if b is True]