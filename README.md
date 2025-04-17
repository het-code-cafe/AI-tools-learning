# README.md

This is a repository used to generate plots and other learning materials 
for AI courses, including ML and Data Science.

This repository contains a variety of diagrams (Mermaid & Napkin), plots (plots folder, all generated with the CCPlots module) and code examples (ray.so folder). 
Most of these plots have also been created with some AI assistance!

## Contents

The following sections are included in this file:

- [Plots generated with Python](#python-plots)
- [Code snippets (in presentation)](#code-snippets-rayso)
- [Mermaid diagrams (in presentation)](#mermaid-diagrams)
- [Styling](#styling)
- [Plot Showcase](#plot-showcase)

Also see the `docs` folder.

## Python Plots

A small module named CCPlots generates the plots for this repository.

1. Install the requirements using `pip install -r requirements.txt`.
2. `python main.py`: contains a function to render all plots again. You can also
any file in the implementation folder to re-do that specific plot only.

The examples in the implementation folder all follow the PlotExample interface, 
allowing them to all be run using the main script in the same manner. Please make 
sure any new implementations adhere to this interface.

Note: many of these implementations were written with AI assistance. They are to be used 
to illustrate concepts around AI only.

## Code Snippets (ray.so)

The code examples for the slides can be found in ray.so_images. They are, 
of course, made with ray.so.

Settings:
- Theme: meadow
- Background: off
- Margin: 16px
- Languages used: Python, Markdown

This folder also contains code snippets used for the presentation. Some are 
just small snippets, others are fully functional scripts to play around with.

## Mermaid Diagrams

In the `mermaid` folder we store the syntax for the mermaid diagrams in `.md` files and store 
`.svg` and `.png` renders.

## Styling

Everything here uses styling from `colour_reference.md`, which contains the 
CodeCafé colours as well as some supporting colours to complete the palette.

## Plot Showcase

Enjoy some cool plots! That need somewhat better styling in the sizing and letters 
and margins.

### Classification Example

![Linear Regression Animation](plots/linear_regression_animation.gif)
![Linear Regression Animation](plots/multivariate_regression_animation.gif)
![KMeans Clustering Animation](plots/kmeans_animation.gif)
![KNN Regression Animation](plots/knn_visualization_animation.gif)
