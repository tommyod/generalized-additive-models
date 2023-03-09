

# generalized-additive-models

## About

## Installation

```
pip install generalized-additive-models
```

## Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from generalized_additive_models import GAM, Spline, Categorical
    
# Load data
data = load_diabetes(as_frame=True)
df, y = data.data, data.target

# Create model
terms = Spline("bp") + Spline("bmi", constraint="increasing") + Categorical("sex")
gam = GAM(terms)

# Cross validate
scores = cross_val_score(gam, df, y, scoring="r2")
print(scores) # array([0.26, 0.4 , 0.41, 0.35, 0.42])
```

## Documentation

## Contributing

Contributions are very welcome.
You can correct spelling mistakes, write documentation, clean up code, implement new features, etc.

Some guidelines:

- Code must comply with the standard. See the GitHub action pipeline for more information.
- If possible, use existing algorithms from `numpy`, `scipy` and `scikit-learn`.
- Write tests, especically regression tests if a bug is fixed.
- Take backward compatibility seriously. API changes require good reason.

## Citing