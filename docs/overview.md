---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: '1.4.1'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Overview

This is a extended web version of the [journal article](https://ieeexplore.ieee.org/abstract/document/9534655/?casa_token=k0nfUkc0KfsAAAAA:t6co-nT1B_q8jei3sQNvOprr33wIwCsCgkeh0GJMq9oSvv7s0NdV0HuYBVswxNnZVFnJOFAlTeU) titled *Combined neural network and adaptive DSP training for long-haul optical communications* published in *Journal of Lightwave Technology* with DOI:10.1109/JLT.2021.3111437

With additional remarks, references, interactive figures and executable source notebooks, we hope this site can improve the reading experience.

## Source codes
``````{margin}
```{note}
Authors' remarks stay at this sidebar along with the table-of-content. We choose washed font-color to minimize distracition.
```
``````

At source notebook page, you can execute the source codes directly in [Colab](https://research.google.com/colaboratory/) by hovering on <i class="fas fa-rocket"></i> at the top and choose `Colab`. You may also download the notebook by hovering on <i class="fas fa-download"></i> and click `.ipynb` so that you can run them locally.



## Figures
Most figures in this article are reproduced into interactive graphs.
Below is an example.

You can:
- drag to zoom in and double click to resume
- move the cursor around to see the number indicators
- single click on a legend item to hide that trace, double click to isolate other traces
- see more options in the toolbar when showing hover cursor on the it


```{code-cell} ipython3
---
tags: [remove-input]
---
import plotly.io as pio
import plotly.express as px
import plotly.offline as py

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", size="sepal_length")
fig
```

## Equations

Equations rendering is done by [MathJax](https://www.mathjax.org/), which allows checking $\LaTeX$ commands in the right click menu of each equation. 

## Feedbacks
You are welcome to discuss or raise issue in the Github repository (<i class="fab fa-github"></i> link at the top).

## Cite
    @article{fan2021combined,
      title={Combined neural network and adaptive DSP training for long-haul optical communications},
      author={Fan, Qirui and Lu, Chao and Lau, Alan Pak Tao},
      journal={Journal of Lightwave Technology},
      year={2021},
      publisher={IEEE}
    }
