from sovaharmony.datasets import BIOMARCADORES_OE, BIOMARCADORES_CE, CHBMP, SRM, DUQUE
from sovaharmony.datasets import BIOMARCADORES_OE_server, BIOMARCADORES_CE_server
from sovaharmony.datasets import BIOMARCADORES_CE_54X10, CHBMP_54X10, SRM_54X10, DUQUE_54X10
from sovaharmony.datasets import test_portables
from sovaharmony.pipeline import pipeline
#from .misc.neuroharmonaze import neurosovaHarmonize

#joblib para paralelizar flujos 
# THE_DATASETS=[
#     CHBMP_54X10,
#     SRM_54X10,
#     DUQUE_54X10,
#     BIOMARCADORES_CE_54X10
#     ]

THE_DATASETS=[
    BIOMARCADORES_CE_server,
    BIOMARCADORES_OE_server
]

spatial=['58x25']
metrics=['osc']


# Inputs not dataset dependent
bands ={'delta':(1.5,6),
        'theta':(6,8.5),
        'alpha-1':(8.5,10.5),
        'alpha-2':(10.5,12.5),
        'beta1':(12.5,18.5),
        'beta2':(18.5,21),
        'beta3':(21,30),
        'gamma':(30,45)
        }

# pipeline(THE_DATASETS,
#          prep=False,
#          post=True,
#          portables=False,
#          tmontage='openBCI',
#          prepdf=False,
#          propdf=False,
#          spatial_matrix=spatial,
#          metrics=metrics,
#          IC=True, 
#          Sensors=False,
#          OVERWRITE=True,
#          bands=bands,
#          norm='False'
#         )

spatial=[None]
metrics=['power']
pipeline(THE_DATASETS,
         prep=False, # if you need preprocessing 
         post=False, # if you need postprocessing
         portables=True, # if you need reduce the number of the sensors
         tmontage='openBCI', # select reductor montage 
         prepdf=False, # If you need the dataframe in preprocessing
         propdf=True, # If you need the dataframe in postprocessing
         spatial_matrix=spatial, # Select the spatial matrix  or None in case not aplied
         metrics=metrics, # List with the metrics extract
         IC=False, # Select the 
         Sensors=True,
         OVERWRITE=True,
         bands=bands,
         norm='False'
        )