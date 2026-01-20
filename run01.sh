#!/bin/bash
set -e

./advection2D

gnuplot plot_final01
gnuplot plot_initial01
gnuplot plot_avg01