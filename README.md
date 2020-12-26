# Einstein Telescope
Created by Jan Harms (Gran Sasso Science Institute)

This repository contains functions to simulate signal-detection and parameter estimation for the Einstein Telescope and other current and future detectors.

## CBC_SNR
Functions to calculate SNRs of BNS and BBH for a cosmological distribution. It implements the method of Marsat/Baker (https://arxiv.org/abs/1806.10734) to simulate changing antenna patterns in frequency domain (based on the stationary-phase approximation). This is a very fast code for SNR analyses, easily allowing for 1e6 BNS signals to be simulated in less than two hours even though these signals last for about a day in the ET observation band and need to be sampled up to kHz.

The code implements the (spinless) TaylorF2 waveform model. The SNR of CBCs calculated with this code deviates insignificantly from the SNR that you get from more sophisticated waveform models (unless in more exotic cases, e.g., larger intermediate-mass BBH where the SNR can significantly depend on the detection of higher-order modes.

The code requires an hdf5 file containing a simulated population of CBC signals. An example for 10^6 BBH signals can be found here: https://1drv.ms/u/s!AvvaaLvjiAhkkMYzQUXChATPeyrw0A?e=xtjY34

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
