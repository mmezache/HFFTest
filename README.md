# High-frequency features detection algorithm documentation


#### Dependencies
Python modules required to run the python script:

---
Numpy (probably already installed):
```sh
$ pip install --user numpy
```
---
cvxopt:
```sh
$ pip install --user cvxopt
```
---
pywt:
```sh
$ pip install --user PyWavelets
```
---
Matplotlib:
```sh
$ pip install --user matplotlib
```
#### Main features
The main features of the library are:
- the detection and computation of the high-frequency features of a 1D signal
- the Monte Carlo procedure to compute the p-value of the test of hypothesis described in [DHMR]

##### File HFF_v1.py
Library of functions in order to compute the examples. The functions are organized in four categories in the library:
* the procedure to compute the HF features parameters,
* the procedure to simulate the null hypothesis,
* the Monte-Carlo procedure to compute the p-value,
* the procedure to compute test signals.

##### File outputFigs.py
Library of functions to display the numerical results.

##### Examples
The file "Example_HFF.py" is a python program which compute the complete procedure for a test signal. The users may change at will the following parameters:
* the length of the signal,
* the standard deviation of the noise,
* the amplitude of the oscillations,
* the parameter of the $\ell^1$-trend filtering,
* the number of iteration of the Monte-Carlo procedure,
* the choice of the test signal.

The program displays the test signal obtained, the trend estimate, the cloud of points corresponding of the HF features of the null (blue dots) and the point corresponding to the HF features of the tested signal (red dot) and the single-sided amplitude spectrum of the signal which emphasizes the points allowing the computations of the HF features.
The time of computations may be significantly long if the number of iteration of the Monte-Carlo procedure is big (over 100). However the Monte-Carlo procedure can be computed in a parallelized framework which reduces drastically the time of computations.
Moreover the automatic choice of the smoothing parameter is efficient for signals which display oscillations of "high" frequency, i.e. if the spike corresponding to the oscillations in the single sided amplitude spectrum is located away from the low-frequency components (example 2 in "Example_HFF.py"). If the signal tested has oscillations located in the low-frequencies, the users are advised to fix the smoothing parameters (example 1 in "Example_HFF.py"). 

## Run on local machine
```sh
$ python Example_HFF.py
```

## Documentation links

* [cvxopt](https://cvxopt.org/) - Optimisation library (Python)
* [pywt](https://pywavelets.readthedocs.io/en/latest/) - Wavelet Transforms library (Python)
* [DHMR](https://hal.archives-ouvertes.fr/hal-02263522) - Testing high-frequency features in a noisy signal (Preprint)
* [l1tf](https://github.com/bugra/l1) - l1 trend filtering python library (Python)
