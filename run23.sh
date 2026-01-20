#!/bin/bash
set -e

./advection2D

gnuplot plot_final23
gnuplot plot_initial23
gnuplot plot_avg