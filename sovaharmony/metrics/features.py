from sovaharmony.metrics.coh import get_coherence
from sovaharmony.metrics.sl import get_sl_1band
from sovaharmony.metrics.p_entropy import get_entropy_freq
#from sovaharmony.metrics.pme import get_pme_freq
#from sovaharmony.metrics.pme import Modulation_Bands_Decomposition,Modulation_Bands_Spectrum,Modulation_Bands_Decomposition_Hamming
from sovaharmony.metrics.pme import Amplitude_Modulation_Analysis
import numpy as np
from sovaflow.flow import fit_spatial_filter
from sovaflow.utils import createRaw
from sovachronux.qeeg_psd_chronux import qeeg_psd_chronux
from sovaharmony.utils import _verify_epoch_continuous,_verify_epochs_axes
import mne
import numpy as np

channels_reduction={'cresta':['F3','F4','C3','C4','P3','P4','O1','O2'],
                    'openBCI':['FP1','FP2','C3','C4','P7','P8','O2','O1'],# #https://docs.openbci.com/Deprecated/UltracortexMark3_NovaDep/
                    'paper':['F3','F4','C3','C4','TP7','TP8','O1','O2'] # En el paper esta T3 y T4, lo cambiamos por TP7 y TP8
                    # Quantitative electroencephalography in mild cognitive impairment: longitdinal changes and possible prediction of ALzheimer's disease
                    }


## USE PYTHON >3.7 , fundamental to guarantee dict order

# each metric has an internal function that is executed internally in get_derivative
# that function should receive signal (mne.Epochs) and kwargs, its arguments
#_get_[derivative_name]

### INTERNAL FEATURES FUNCTIONS ###
def qeeg_psd_irasa(data, sf,bands,ch_names,osc=True, aperiodic=True,fmin=1,fmax=45,win_sec=5):
    '''
    Function responsible for calculating PSD distribution at specific frequency intervals, using YASA library.
    
    Input parameters:
        - data:
            numpy array, required.
            2-D matrix [samples, trials]. It contains the data to process.
        - sf: int
            Sample frequencie 
        - bands:

        - ch_names
        - osc
        - aperiodic
        - fmin
        - fmax
        - win_sec
        
    Output:
        - power_normalized 
        - fit_params
    '''
   
    power = {}
    # PSD using YASA
    import yasa
    freqs, psd_aperiodic, psd_osc,fit_params = yasa.irasa(data, sf, ch_names=None, band=(fmin, fmax), win_sec=win_sec, return_fit=True)
    if osc:
        # To avoid the negative values that YASA gets, we use the following:
        psd=psd_osc-np.min(psd_osc)
    if aperiodic:
         # To avoid the negative values that YASA gets, we use the following:
        psd=psd_aperiodic-np.min(psd_aperiodic)
    else: 
        psd = psd_aperiodic + psd_osc
    
    for band_label,vals in bands.items():
        fmin,fmax = vals
        idx_band = np.logical_and(fmin <= freqs, freqs < fmax)
        pot_band = sum(psd.T[idx_band == True])
        power[band_label]=pot_band
        
    total_pot = sum(list(power.values()))
    power_normalized = {}
    
    #Calculate the density power for each interval.
    for band_label,pot_band in power.items():
        power_normalized[band_label]=pot_band/total_pot

    return power_normalized,fit_params,psd,freqs


def _get_power(signal_epoch,bands,irasa=False,osc=False, aperiodic=False):
    signal = np.transpose(signal_epoch.get_data(),(1,2,0)) # epochs spaces times -> spaces times epochs
    _verify_epochs_axes(signal_epoch.get_data(),signal)
    space_names = signal_epoch.info['ch_names']
    spaces,times,epochs = signal.shape
    output = {}
    if osc==False and aperiodic==False:
        type='power'
    else:
        type='irasa'
    output['metadata'] = {'type':type,'kwargs':{'bands':bands}}
    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),spaces))
    output['metadata']['axes']={'bands':bands_list,'spaces':space_names}
    nchans,points,epochs = signal.shape
    signalCont = np.reshape(signal,(nchans,points*epochs),order='F')
    
    if irasa:
        if aperiodic:
            output['fit_params']={}
            output['fit_params']['values']=[]
        welch_matrix=[]
        for space in space_names:
            space_idx = space_names.index(space)
            dummy,fit_params,psd,freqs = qeeg_psd_irasa(signalCont[space_idx,:], signal_epoch.info['sfreq'],bands,signal_epoch.ch_names,osc=osc, aperiodic=aperiodic,fmin=1,fmax=30)
            welch_matrix.append(psd)
            if aperiodic:
                output['fit_params']['axes']=list(fit_params.keys())[1:]
                output['fit_params']['values']+=[[fit_params['Intercept'][0],fit_params['Slope'][0],fit_params['R^2'][0],fit_params['std(osc)'][0]]]
            for b in bands.keys():
                band_idx = bands_list.index(b)
                values[band_idx,space_idx]=dummy[b] #if there is an error in this line update sovachornux
        output['values'] = values
        
    else:
        for space in space_names:
            space_idx = space_names.index(space)
            dummy = qeeg_psd_chronux(signal[space_idx,:,:],signal_epoch.info['sfreq'],bands,spectro=True)         
            
            for b in bands.keys():
                band_idx = bands_list.index(b)
                values[band_idx,space_idx]=dummy[2][b] #if there is an error in this line update sovachornux
        output['values'] = values
    if irasa==False and osc==False and aperiodic==False: 
        return output
    else:
        return output, welch_matrix

def _get_psd(signal_epoch,bands):
    signal = np.transpose(signal_epoch.get_data(), (1, 2, 0))  # epochs spaces times -> spaces times epochs
    _verify_epochs_axes(signal_epoch.get_data(), signal)

    space_names = signal_epoch.info['ch_names']
    spaces, times, epochs = signal.shape

    output = {}
    output['metadata'] = {'type': 'psd', 'kwargs': {'bands': bands}}
    output['metadata']['axes']={'spaces':space_names}
    values = None  # Inicialización para asignar más tarde
    freqs = None

    for space_idx, space in enumerate(space_names):
        dummy = qeeg_psd_chronux(signal[space_idx, :, :], signal_epoch.info['sfreq'], bands, spectro=True)
        
        # Inicialización dinámica de valores y frecuencias al detectar la forma de dummy
        if values is None or freqs is None:
            values = np.empty((spaces, *dummy[0].shape))  # Dimensiones dinámicas
            freqs = np.empty((spaces, *dummy[1].shape))
        
        # Asignación de valores y frecuencias
        values[space_idx] = dummy[0]
        freqs[space_idx] = dummy[1]

    output['values'] = values
    output['freqs'] = freqs
    return output

    

def _get_sl(signal_epoch,bands):
    space_names = signal_epoch.info['ch_names']
    epochs,spaces,times = signal_epoch.get_data().shape

    output = {}
    output['metadata'] = {'type':'sl','kwargs':{'bands':bands}}

    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),spaces,spaces))
    output['metadata']['axes']={'bands':bands_list,'spaces1':space_names,'spaces2':space_names}

    for b,brange in bands.items():
        dummy = get_sl_1band(signal_epoch,brange[0],brange[1])
        band_idx = bands_list.index(b)
        values[band_idx,:,:]=dummy
    output['values'] = values
    return output

def _get_coh(signal_epoch,window,bands):
    chs = signal_epoch.info['ch_names']
    blist=list(bands.keys())
    _,Cfxy = get_coherence(signal_epoch,bands,window)
    axes = {'bands':blist,'spaces1':chs,'spaces2':chs}
    output = {}
    dim0 = list(axes.keys())[0]
    output['metadata'] = {'type':f'coherence-{dim0}','kwargs':{'window':window,'bands':bands}}
    output['metadata']['axes']=axes
    output['values']=Cfxy
    return output

def _get_pme(signal_epoch,bands):
    signal = np.transpose(signal_epoch.get_data(),(1,2,0)) # epochs spaces times -> spaces times epochs
    _verify_epochs_axes(signal_epoch.get_data(),signal)
    space_names = signal_epoch.info['ch_names']
    spaces,times,epochs = signal.shape
    output = {}
    output['metadata'] = {'type':'crossfreq','kwargs':{'bands':bands}}
    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),spaces))
    output['metadata']['axes']={'spaces':space_names,'bands':bands_list,'bands':bands_list}
    values = Amplitude_Modulation_Analysis(signal,signal_epoch.info['sfreq'],Bands=list(bands.values()))
    output['values'] = values
    return output

def _get_entropy(signal_epoch,bands,D):
    space_names = signal_epoch.info['ch_names']
    epochs,spaces,times = signal_epoch.get_data().shape

    output = {}
    output['metadata'] = {'type':'entropy','kwargs':{'bands':bands,'D':D}}

    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),spaces))
    output['metadata']['axes']={'bands':bands_list,'spaces':space_names}

    for b,brange in bands.items():
        fmin,fmax=brange
        dummy = get_entropy_freq(signal_epoch,fmin=fmin,fmax=fmax,D=D)
        band_idx = bands_list.index(b)
        values[band_idx,:]=dummy
    output['values'] = values
    return output

#### generalizado#######

foo_map={
    'absPower':_get_power,
    'power':_get_power,
    'power_osc':_get_power,
    'power_ape':_get_power,
    'sl':_get_sl,
    'cohfreq':_get_coh,
    'crossfreq':_get_pme,
    'entropy':_get_entropy,
    'psd':_get_psd
}
def get_derivative(in_signal,feature,kwargs,spatial_filter=None,portables=False, montage_select=str):
    """
    Returns derivative
    If spatial_filter is not None, it will be computed over ics, otherwise over channels

    signal: mne.Epochs object
    feature: str, the feature you want
    kwargs: arguments for the fuction that calculates that feature
    spatial_filter: tuple (A,W,spatial_filter_chs)
    """
    signal = in_signal.copy()
    
    if spatial_filter is not None:
        # ICs powers
        A,W,spatial_filter_chs,sf_name = spatial_filter['A'],spatial_filter['W'],spatial_filter['ch_names'],spatial_filter['name']
        intersection_chs = list(set(spatial_filter_chs).intersection(signal.ch_names))
        W_adapted = fit_spatial_filter(W,spatial_filter_chs,intersection_chs,mode='demixing')
        signal.reorder_channels(intersection_chs)
        signal2 = np.transpose(signal.get_data(),(1,2,0)) # epochs spaces times -> spaces times epochs
        _verify_epochs_axes(signal.get_data(),signal2)
        nchans,points,epochs = signal2.shape
        signalCont = np.reshape(signal2,(nchans,points*epochs),order='F')
        _verify_epoch_continuous(signal.get_data(),signalCont,('epochs','spaces','times'))
        if portables:
            resample= signal.info['sfreq']/4 
            signal_resample=signalCont[:,::4] # Signal chx resample- continue
            ics = W_adapted @ signal_resample
            comps = ics.shape[0]
            ics_epoch = np.reshape(ics,(comps,int(points/4),epochs),order='F')
            
        else:
            ics = W_adapted @ signalCont
            comps = ics.shape[0]
            ics_epoch = np.reshape(ics,(comps,points,epochs),order='F')
        
        _verify_epoch_continuous(ics_epoch,ics,('spaces','times','epochs'))
        ics_epoch2 = np.transpose(ics_epoch,(2,0,1))
        _verify_epochs_axes(ics_epoch2,ics_epoch)
        del ics_epoch
        del ics
        del signalCont
        del signal2
        info_epochs=mne.create_info(['C'+str(x+1) for x in range(comps)], in_signal.info['sfreq'], ch_types='eeg')
        signal = mne.EpochsArray(ics_epoch2,info_epochs)
    
    if spatial_filter==None and portables:
        intersection_chs =list(set(channels_reduction[montage_select]).intersection(signal.ch_names))
        signal.reorder_channels(intersection_chs)
    
    if feature in ('power_ape', 'power_osc', 'power_irasa'):
        output,psd=foo_map[feature](signal,**kwargs)
    else:
        output=foo_map[feature](signal,**kwargs)
    if spatial_filter is not None:
        output['metadata']['space']='ics'
        output['metadata']['W'] = W_adapted
        output['metadata']['W_channels']=intersection_chs
        output['metadata']['spatial_filter_name']=sf_name
    else:
        output['metadata']['space']='sensors'
    return output
    
