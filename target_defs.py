import numpy as np


archetype_sequences = {'liquid': 'BAABAABAABAABABBAAAB',
                    'membrane': 'ABAAABBAAAABBABBBAAA',
                    'spherical micelle': 'BBBBBBBBAAAAAAAAAAAA',
                    'string': 'ABBABABAAAAABABBABAA',
                    'vesicle': 'BABBBAABAAAABAABAABA',
                    'wormlike micelle': 'BBAAABBBAAAABBBAAAAA',
                     }

archetype_predictions = {'liquid': np.array([ 18.177, 2.932]),
                        'membrane': np.array([ 1.132, 8.499]),
                        'spherical micelle': np.array([ 2.901, -4.351]),
                        'string': np.array([ -4.079, 6.662]),
                        'vesicle': np.array([ 6.498, 8.414]),
                        'wormlike micelle': np.array([ -3.771, 0.172]),
                         }

archetype_plaintext = {'spherical micelle': 'uniform spherical micelles',
                       'membrane': 'sheet-like membranes',
                       'vesicle': 'hollow vesicles',
                       'wormlike micelle': 'heterogeneous, worm-like micelle aggregates',
                       'liquid': 'amorphous liquid droplets',
                       'string': 'string-like aggregates',
                       }
