import numpy as np
import pandas as pd

import time
import progressbar

import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

class Interferometer:

    def __init__(self, name='ET', interferometer=0, plot=False):
        self.plot = plot
        self.ifo_id = interferometer
        self.name = name+str(interferometer)

        self.R_earth = 6730000.

        self.setProperties()

    def setProperties(self):

        k = self.ifo_id

        if self.name[0:2] == 'ET':
            # the lat/lon/azimuth values are just approximations (for Sardinia site)
            self.lat = (40+31.0/60) * np.pi/180.
            self.lon = (9+25.0/60) * np.pi/180.
            self.arm_azimuth = 87.*np.pi/180.+k*np.pi*2./3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth)*e_long + np.sin(self.arm_azimuth)*e_lat
            self.e2 = np.cos(self.arm_azimuth+np.pi/3.)*e_long + np.sin(self.arm_azimuth+np.pi/3.)*e_lat

            self.psd_data = np.loadtxt('ET_D_psd.txt')

            self.duty_factor = 0.85
        elif self.name == 'H':
            self.lat = 46.5 * np.pi/180.
            self.lon = -119.4 * np.pi/180.
            self.arm_azimuth = 126. * np.pi/180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth+np.pi/2.) * e_long + np.sin(self.arm_azimuth+np.pi/2.) * e_lat

            self.psd_data = np.loadtxt('LIGO_O5_psd.txt')

            self.duty_factor = 0.85
        elif self.name == 'CE1':
            self.lat = 46.5 * np.pi/180.
            self.lon = -119.4 * np.pi/180.
            self.arm_azimuth = 126. * np.pi/180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth+np.pi/2.) * e_long + np.sin(self.arm_azimuth+np.pi/2.) * e_lat

            self.psd_data = np.loadtxt('CE1_psd.txt')

            self.duty_factor = 0.85
        elif self.name == 'CE2':
            self.lat = 46.5 * np.pi/180.
            self.lon = -119.4 * np.pi/180.
            self.arm_azimuth = 126. * np.pi/180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth+np.pi/2.) * e_long + np.sin(self.arm_azimuth+np.pi/2.) * e_lat

            self.psd_data = np.loadtxt('CE2_psd.txt')

            self.duty_factor = 0.85
        elif self.name == 'CEA':    # hypothetical CE2 in Australia
            self.lat = -20.517
            self.lon = 131.061
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('CE2_psd.txt')

            self.duty_factor = 0.85
        elif self.name == 'L':
            self.lat = 30.6 * np.pi/180.
            self.lon = -90.8 * np.pi/180.
            self.arm_azimuth = -162. * np.pi/180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth+np.pi/2.) * e_long + np.sin(self.arm_azimuth+np.pi/2.) * e_lat

            self.psd_data = np.loadtxt('LIGO_O5_psd.txt')

            self.duty_factor = 0.85
        elif self.name == 'V':
            self.lat = 43.6 * np.pi/180.
            self.lon = 10.5 * np.pi/180.
            self.arm_azimuth = 71. * np.pi/180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth+np.pi/2.) * e_long + np.sin(self.arm_azimuth+np.pi/2.) * e_lat

            self.psd_data = np.loadtxt('Virgo_O5_psd.txt')

            self.duty_factor = 0.85

        self.Sn = interp1d(self.psd_data[:, 0], self.psd_data[:, 1], bounds_error=False, fill_value=1.)

        if self.plot:
            plt.figure()
            plt.loglog(self.psd_data[:, 0], np.sqrt(self.psd_data[:, 1]))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Strain noise')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('Sensitivity_' + self.name + '.png')
            plt.close()


class Detector:

    def __init__(self, name='ET', plot=False):
        self.interferometers = []
        if name=='ET':
            for k in np.arange(3):
                self.interferometers.append(Interferometer(name=name, interferometer=k, plot=plot))
        else:
            self.interferometers.append(Interferometer(name=name, interferometer='', plot=plot))

class Network:

    def __init__(self, detector_ids=['ET'], plot=False):
        self.name = ''
        for id in detector_ids:
            self.name += id

        self.detectors = []
        for d in np.arange(len(detector_ids)):
            print(detector_ids[d])
            detectors = Detector(name=detector_ids[d], plot=plot).interferometers
            for i in np.arange(len(detectors)):
                self.detectors.append(detectors[i])

def fisco(parameters):
    c = 299792458.
    G = 6.674e-11

    M = (parameters['mass_1']+parameters['mass_2']) * 1.9885e30 * (1+parameters['redshift'])

    return 1/(np.pi)*c**3/(G*M)/6**1.5  # frequency of last stable orbit

def TaylorF2(parameters, cosmo, frequencyvector, maxn=8, plot=None):
    ff = frequencyvector
    ones = np.ones((len(ff),1))

    phic = parameters['phase']
    tc = parameters['geocent_time']
    z = parameters['redshift']
    r = cosmo.luminosity_distance(z).value * 3.086e22
    iota = parameters['iota']

    # define necessary variables, multiplied with solar mass, parsec, etc.
    M = (parameters['mass_1'] + parameters['mass_2']) * 1.9885*10**30 * (1+z)
    mu = (parameters['mass_1'] * parameters['mass_2'] / (
            parameters['mass_1'] + parameters['mass_2'])) * 1.9885*10**30 * (1+z)

    # define constants
    c = 299792458.
    G = 6.674e-11

    Mc = G*mu**0.6*M**0.4/c**3

    # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf)
    hp = c/(2.*r)*np.sqrt(5.*np.pi/24.)*Mc**(5./6.)/(np.pi*ff)**(7./6.)*(1.+np.cos(iota)**2.)
    hc = c/(2.*r)*np.sqrt(5.*np.pi/24.)*Mc**(5./6.)/(np.pi*ff)**(7./6.)*2.*np.cos(iota)

    C = 0.57721566  # Euler constant
    eta = mu/M

    f_isco = fisco(parameters)

    v = (np.pi*G*M/c**3*ff)**(1./3.)
    v_isco = (np.pi*G*M/c**3*f_isco)**(1./3.)

    # coefficients of the PN expansion (https://arxiv.org/pdf/0907.0700.pdf)
    pp = np.hstack((1.*ones, 0.*ones, 20./9.*(743./336.+eta*11./4.)*ones, -16*np.pi*ones,
                   10.*(3058673./1016064.+5429./1008.*eta+617./144.*eta**2)*ones,
                   np.pi*(38645./756.-65./9.*eta)*(1+3.*np.log(v/v_isco)),
                    11583231236531./4694215680.-640./3.*np.pi**2-6848./21.*(C+np.log(4*v))
                     +(-15737765635./3048192.+2255./12.*np.pi**2)*eta+76055./1728.*eta**2-127825./1296.*eta**3,
                    np.pi*(77096675./254016.+378515./1512.*eta-74045./756.*eta**2)*ones))

    psi = 0.

    for k in np.arange(maxn):
        PNc = pp[:,k]
        psi += PNc[:,np.newaxis]*v**k

    psi *= 3./(128.*eta*v**5)
    psi += 2.*np.pi*ff*tc-phic-np.pi/4.
    phase = np.exp(1.j*psi)
    polarizations = np.hstack((hp*phase, hc*1.j*phase))
    polarizations[np.where(ff>3*f_isco),:] = 0.j   # very crude high-f cut-off

    # t(f) is required to calculate slowly varying antenna pattern as function of instantaneous frequency.
    # This FD approach follows Marsat/Baker arXiv:1806.10734v1; equation (22) neglecting the phase term, which does not
    # matter for SNR calculations.
    t_of_f = np.diff(psi,axis=0)/(2.*np.pi*(ff[1]-ff[0]))
    t_of_f = np.vstack((t_of_f,[t_of_f[-1]]))

    if plot != None:
        plt.figure()
        plt.semilogx(ff,t_of_f-tc)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('t(f)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('t_of_f_' + plot + '.png')
        plt.close()

    return polarizations, t_of_f

def GreenwichMeanSiderealTime(gps):
    # calculate the Greenwhich mean sidereal time
    sidereal_day = 23.9344696
    return np.mod(9.533088395981618 + (gps-1126260000.)/3600.*24./sidereal_day, 24) * np.pi/12.

def projection(parameters, detectors, polarizations, timevector):
    """
    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note1: there is a typo in the definition of the wave-frame in Nishizawa et al.
    Note2: it is computationally more expensive to use numpy.einsum instead of working with several vector quantities
    """

    nt = len(polarizations[:, 0])

    if np.iscomplex(np.sum(polarizations)):
        proj = np.zeros((nt, len(detectors)),dtype=complex)
    else:
        proj = np.zeros((nt, len(detectors)))

    if timevector.ndim==1:
        timevector = timevector[:,np.newaxis]

    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']

    gmst = GreenwichMeanSiderealTime(timevector)

    phi = ra-gmst
    theta = np.pi/2.-dec

    #start_time = time.time()
    ux = np.cos(theta)*np.cos(phi[:,0])
    uy = np.cos(theta)*np.sin(phi[:,0])
    uz = -np.sin(theta)

    vx = -np.sin(phi[:,0])
    vy = np.cos(phi[:,0])
    vz = 0
    #print("Creating vectors u,v: %s seconds" % (time.time() - start_time))

    #start_time = time.time()
    mx = -ux*np.sin(psi) - vx*np.cos(psi)
    my = -uy*np.sin(psi) - vy*np.cos(psi)
    mz = -uz*np.sin(psi) - vz*np.cos(psi)

    nx = -ux*np.cos(psi) + vx*np.sin(psi)
    ny = -uy*np.cos(psi) + vy*np.sin(psi)
    nz = -uz*np.cos(psi) + vz*np.sin(psi)
    #print("Creating vectors m, n: %s seconds" % (time.time() - start_time))

    #start_time = time.time()
    hxx = polarizations[:, 0]*(mx*mx-nx*nx)+polarizations[:, 1]*(mx*nx+nx*mx)
    hxy = polarizations[:, 0]*(mx*my-nx*ny)+polarizations[:, 1]*(mx*ny+nx*my)
    hxz = polarizations[:, 0]*(mx*mz-nx*nz)+polarizations[:, 1]*(mx*nz+nx*mz)
    hyy = polarizations[:, 0]*(my*my-ny*ny)+polarizations[:, 1]*(my*ny+ny*my)
    hyz = polarizations[:, 0]*(my*mz-ny*nz)+polarizations[:, 1]*(my*nz+ny*mz)
    hzz = polarizations[:, 0]*(mz*mz-nz*nz)+polarizations[:, 1]*(mz*nz+nz*mz)
    #print("Calculation GW tensor: %s seconds" % (time.time() - start_time))

    #start_time = time.time()
    for k in np.arange(len(detectors)):
        e1 = detectors[k].e1
        e2 = detectors[k].e2
        n = k
        e1 = np.array([np.cos(n * np.pi * 2. / 3.), np.sin(n * np.pi * 2. / 3.), 0.])
        e2 = np.array([np.cos(np.pi / 3. + n * np.pi * 2. / 3.), np.sin(np.pi / 3. + n * np.pi * 2. / 3.), 0.])

        proj[:, k] = 0.5 * (e1[0]**2-e2[0]**2)*hxx\
                     + 0.5 * (e1[1]**2-e2[1]**2)*hyy\
                     + 0.5 * (e1[2]**2-e2[2]**2)*hzz\
                     + (e1[0]*e1[1]-e2[0]*e2[1])*hxy\
                     + (e1[0]*e1[2]-e2[0]*e2[2])*hxz\
                     + (e1[1]*e1[2]-e2[1]*e2[2])*hyz
    #print("Calculation of projection: %s seconds" % (time.time() - start_time))

    return proj

def SNR(detectors, signals, T, fs, duty_cycle=False, plot=None):

    if signals.ndim==1:
        signals = signals[:,np.newaxis]

    ff = np.linspace(0.0, fs/2.0, int(fs*T)//2+1)
    if ff.ndim==1:
        ff = ff[:,np.newaxis]

    SNRs = np.zeros(len(detectors))
    for k in np.arange(len(detectors)):
        signal = np.abs(signals[1:,k])**2
        SNRs[k] = 2*np.sqrt(np.sum(np.abs(signals[1:,k])**2/detectors[k].Sn(ff[1:,0]),axis=0)/T)
        if plot!=None:
            plt.figure()
            plt.semilogx(ff[1:], 2*np.sqrt(np.abs(signals[1:,k])**2/detectors[k].Sn(ff[1:,0])/T))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('SNR spectral density')
            plt.xlim((1, fs/2))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('SNR_density' + detectors[k].name + '_' + plot + '.png')
            plt.close()

        # set SNRs to zero if interferometer is not operating (according to its duty factor [0,1])
        if duty_cycle:
            operating = np.random.rand()
            if detectors[k].duty_factor<operating:
                SNRs[k] = 0.

    return SNRs

def horizon(detectors, parameters, cosmo, frequencyvector, SNRmin):
    ff = frequencyvector
    T = 1./(ff[1]-ff[0])
    fs = 2*ff[-1]

    def dSNR(z, SNR0):
        z = np.max([0.1,z[0]])

        r = cosmo.luminosity_distance(z).value * 3.086e22

        # define necessary variables, multiplied with solar mass, parsec, etc.
        M = (parameters['mass_1'] + parameters['mass_2']) * 1.9885e30 * (1+z)
        mu = (parameters['mass_1'] * parameters['mass_2'] / (
                parameters['mass_1'] + parameters['mass_2'])) * 1.9885e30 * (1+z)

        parameters['redshift'] = z
        f_isco_z = fisco(parameters)

        # define constants
        c = 299792458.
        G = 6.674e-11

        Mc = G * mu ** 0.6 * M ** 0.4 / c ** 3

        # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf) with optimal orientation
        hp = c/r * np.sqrt(5.*np.pi/24.) * Mc**(5./6.) / (np.pi*ff[1:])**(7./6.)
        hp[ff[1:] > 3*f_isco_z] = 0  # very crude, but reasonable high-f cut-off; matches roughly IMR spectra (in qadrupole order)
        hp = np.vstack(([0],hp))

        hc = 1.j*hp

        hpij = np.array([[1,0,0],[0,-1,0],[0,0,0]])
        hcij = np.array([[0,1,0],[1,0,0],[0,0,0]])

        # project signal onto the detector
        proj = np.zeros((len(hp), len(detectors)), dtype=complex)

        for k in np.arange(len(detectors)):
            if detectors[k].name[0:2] == 'ET':
                n = detectors[k].ifo_id
                az = n*np.pi*2./3.
                e1 = np.array([np.cos(az),np.sin(az),0.])
                e2 = np.array([np.cos(az+np.pi/3.),np.sin(az+np.pi/3.),0.])
            else:
                e1 = np.array([1., 0., 0.])
                e2 = np.array([0., 1., 0.])

            proj[:, k] = 0.5 * hp[:,0] * (e1 @ hpij @ e1 - e2 @ hpij @ e2) \
                        +0.5 * hc[:,0] * (e1 @ hcij @ e1 - e2 @ hcij @ e2)
            #proj[:, k] = 0.5 * hp[:, 0] * (e1 @ hpij @ e1 - e2 @ hpij @ e2)
            #proj[:, k] = 0.5j * hc[:, 0] * (e1 @ hcij @ e1 - e2 @ hcij @ e2)

        SNRs = SNR(detectors, proj, T, fs)
        SNRtot = np.sqrt(np.sum(SNRs**2))

        #print('z = ' + str(z) + ', r = ' + str(cosmo.luminosity_distance(z).value) + 'Mpc, SNR = '+str(SNRtot))

        return SNRtot-SNR0

    zmax = np.zeros_like(SNRmin)
    for k in np.arange(len(SNRmin)):
        zmax[k] = optimize.root(lambda x: dSNR(x, SNRmin[k]), 5).x[0]

    return zmax

def parameterHistograms(parameters, population):
    hist, zz = np.histogram(parameters['redshift'].to_numpy(), np.linspace(0, 100, 1001))
    plt.bar(0.5*(zz[0:-1]+zz[1:]), hist, align='center', alpha=0.5, width=0.9*(zz[1]-zz[0]))
    plt.xlabel('Redshift of analyzed signals')
    plt.ylabel('Count')
    plt.xlim(0, 10)#np.ceil(np.max(parameters['redshift'].to_numpy())))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Histogram_' + population + '_redshift.png', dpi=300)
    plt.close()

    plt.hist(parameters['mass_1'].to_numpy(), np.linspace(1.15, 1.5, 101))
    plt.xlabel('M1')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Histogram_' + population + '_M1.png', dpi=300)
    plt.close()

    plt.hist(parameters['mass_2'].to_numpy(), np.linspace(1.15, 1.3, 101))
    plt.xlabel('M1')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Histogram_' + population + '_M2.png', dpi=300)
    plt.close()

def analyzeDetections(network_SNR, threshold_ii, SNR0, parameters, population, tag=''):
        ns = len(network_SNR)
        nSNR = len(threshold_ii[0,:])
        maxz = np.zeros_like(SNR0)
        for k in np.arange(nSNR):
            ii = np.where(threshold_ii[:,k] == 1)[0]
            ndet = len(ii)
            maxz[k] = np.max(parameters['redshift'].iloc[ii].to_numpy())
            maxSNR = np.max(network_SNR)
            print('Detected signals with SNR>{:.3f}: {:.3f} ({:} out of {:})'.format(SNR0[k], ndet/ns, ndet, ns))
            print('Maximum detected redshift %.3f' % maxz[k])

        print('SNR: {:.1f} (min) , {:.1f} (max) '.format(np.min(network_SNR), np.max(network_SNR)))

        hist_tot, zz = np.histogram(parameters['redshift'].iloc[0:ns].to_numpy(), np.linspace(0, 100, 1001))

        for k in np.arange(nSNR):
            ii = np.where(threshold_ii[:,k]==1)
            hist_det, zz = np.histogram(parameters['redshift'].iloc[ii].to_numpy(), zz)
            plt.bar(0.5*(zz[0:-1]+zz[1:]), hist_det, align='center', alpha=0.5, width=0.9*(zz[1]-zz[0]))
        plt.xlabel('Redshift of detected signals')
        plt.ylabel('Count')
        plt.xlim(0, np.ceil(np.max(maxz)))
        plt.grid(True)
        plt.tight_layout()
        plt.legend(['SNR>' + s for s in SNR0.astype(str)])
        plt.savefig('Histogram_' + population + '_z_' + tag + '.png', dpi=300)
        plt.close()

        hist_tot[np.where(hist_tot<1)] = 1e10
        for k in np.arange(nSNR):
            ii = np.where(threshold_ii[:,k]==1)
            hist_det, zz = np.histogram(parameters['redshift'].iloc[ii].to_numpy(), zz)
            plt.bar(0.5*(zz[0:-1]+zz[1:]), hist_det/hist_tot, align='center', alpha=0.5, width=0.9*(zz[1]-zz[0]))
        plt.xlabel('Redshift')
        plt.ylabel('Detection efficiency')
        plt.xlim(0, np.ceil(np.max(maxz)))
        plt.grid(True)
        plt.tight_layout()
        plt.legend(['SNR>' + s for s in SNR0.astype(str)])
        plt.savefig('Efficiency_' + population + '_z_' + tag + '.png', dpi=300)
        plt.close()

        n, bins, patches = plt.hist(network_SNR, np.linspace(0, np.ceil(maxSNR), 51), facecolor='g', alpha=0.75)
        plt.xlabel('SNR')
        plt.ylabel('Count')
        plt.xlim(0, np.ceil(maxSNR))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Histogram_' + population + '_SNR_' + tag + '.png', dpi=300)
        plt.close()

def main():

    #fs = 1024    # good for BBH in ET
    fs = 2048   # good for BNS in ET
    T = 4    # good for ET with TaylorF2

    ns = 10000 #820000  # number of signals to simulate (1e6 is maximum for BBH, 820000 is maximum for BNS)

    detSNR = np.array([8., 12., 25., 50.])  #SNR detection thresholds

    population = 'BNS'
    networks = [['ET', 'CE1'], ['ET']]

    cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

    parameters = pd.read_hdf('BBH_injections_1e6.hdf5').iloc[0:ns]
    parameters_BNS = pd.read_csv('BNS_injections_8e5.txt',
                                 names=['mass_1', 'mass_2', 'redshift', 'luminosity_distance'],
                                 delimiter=' ').sample(frac=1).iloc[0:ns]
    if population == 'BNS':
        # here it is neglected that p(z) is different for BNS and BBH
        parameters['mass_1'] = parameters_BNS['mass_1'].to_numpy()
        parameters['mass_2'] = parameters_BNS['mass_2'].to_numpy()
        parameters['redshift'] = parameters_BNS['redshift'].to_numpy()

    parameterHistograms(parameters, population)

    frequencyvector = np.linspace(0.0, fs / 2.0, (fs * T) // 2 + 1)
    frequencyvector = frequencyvector[:, np.newaxis]

    for n in np.arange(len(networks)):
        network = Network(networks[n])

        if len(detSNR)==1:
            population = 'BNS_'+network.name+'_SNR'+str(detSNR)
        else:
            population = 'BNS_'+network.name+'_mSNR'

        zmax = horizon(network.detectors, parameters.iloc[0], cosmo, frequencyvector, detSNR)
        for k in np.arange(len(detSNR)):
            print(population + ' horizon (time-invariant antenna pattern; M='
                  + str(parameters['mass_1'].iloc[0]+parameters['mass_2'].iloc[0]) + '; SNR>'
                  + str(detSNR[k]) + '): {:.3f}'.format(zmax[k]))

        network_SNR = np.zeros(ns)
        threshold_ii = np.zeros((ns, len(detSNR)))

        print('Processing CBC population')
        bar = progressbar.ProgressBar(max_value=ns)
        for k in np.arange(ns):
            #GW150914 = {'mass_1': 36.2, 'mass_2': 29.1, 'geocent_time': 1126259642.413, 'phase': 1.3, 'ra': 1.375, 'dec': -1.2108,
            #                  'psi': 2.659, 'iota': 0.3, 'redshift': 0.093}
            #parameters.iloc[k] = GW150914

            one_parameters = parameters.iloc[k]

            # characteristic time scale
            dT = 1./fisco(one_parameters)

            wave, t_of_f = TaylorF2(one_parameters, cosmo, frequencyvector[1:,:], maxn=8)
            t_of_f = np.zeros_like(t_of_f)
            signal = projection(one_parameters, network.detectors, wave, t_of_f)
            # fill the first sample of Fourier spectra with value 0, since it was left out to avoid division at f=0
            signal = np.vstack((np.zeros(len(network.detectors)),signal))

            SNRs = SNR(network.detectors, signal, T, fs, duty_cycle=True) #, plot='FD'+str(k))[0]
            network_SNR[k] = np.sqrt(np.sum(SNRs**2))
            for l in np.arange(len(detSNR)):
                if (network_SNR[k]>detSNR[l]):
                    threshold_ii[k,l] = 1

            bar.update(k)

        bar.finish()

        analyzeDetections(network_SNR, threshold_ii, detSNR, parameters, population, tag='')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
