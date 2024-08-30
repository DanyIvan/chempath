# Chempath


<p align="center">
  <img src="figures/chemnpath.pdf">
</p>



Chempath is a pathway analysis program that automatically builds the the most important pathways of a reaction system. This algorithm was originally developed by Lehmann (2004). Chempath is an open-source, python implementation of this algorithm


## How to use Chempath

Clone this repository with and go to the repository main directory. Try running the tests:

```python
python chempath_tests.py
```

If there are no errors, you can go ahead and use Chempath.

- See the [tutorial](tutorial.ipynb) jupyter-notebook to learn how to use Chempath.
- See an [example](examples/box_model_pathways/box_model_pathways_example.ipynb) of how to use Chempath in a photochemical box model
- See an [example](examples/photochem_modern_earth/pathways_in_photochem.ipynb) of how to use Chempath in a 1D photochemical model

# References

Lehmann (2004): An Algorithm for the Determination of All Significant Pathways in Chemical Reaction Systems, Journal of Atmospheric Chemistry, 47, 45–78.