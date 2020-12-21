# Einstein Telescope
Created by Jan Harms (Gran Sasso Science Institute)

This repository contains functions to simulate signal-detection and parameter-estimation problems for the Einstein Telescope and other current and future detectors.

## CBC_SNR
Functions to calculate SNRs of BNS and BBH for a cosmological distribution. It implements the method of Marsat/Baker (https://arxiv.org/abs/1806.10734) to simulate changing antenna patterns in frequency domain (based on the stationary-phase approximation). This is a very fast code for SNR analyses, easily allowing for 1e6 BNS signals to be simulated within a few hours. 

The code only implements (spinless) TaylorF2 and TaylorT2 waveforms, but with the Marsat/Baker method, I strongly discourage people from using the time-domain waveform TaylorT2. The frequency does not evolve close enough to merger frequencies, and therefore you can significantly lose SNR with TaylorT2 when you have very heavy or very distant BBHs (which appear as brief transients at low frequencies). TaylorF2 does not have this problem since you can extend the frequency band as you like, and as implemented now, the SNR of CBCs calculated with this code deviates insignificantly from the SNR that you get from more sophisticated waveform models. 

Some codes require an hdf5 file containing a simulated population of CBC signals. An example for 10^6 BBH signals can be found here: https://1drv.ms/u/s!AvvaaLvjiAhkkMYzQUXChATPeyrw0A?e=xtjY34

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
