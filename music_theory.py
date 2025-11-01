# Basic music theory (with help from my music teacher!)
SCALES = {
    'C_major': [0, 2, 4, 5, 7, 9, 11],      
    'G_major': [7, 9, 11, 0, 2, 4, 6],      
    'D_major': [2, 4, 6, 7, 9, 11, 1],      
    'A_major': [9, 11, 1, 2, 4, 6, 8],
    'E_major': [4, 6, 8, 9, 11, 1, 3],
    'B_major': [11, 1, 3, 4, 6, 8, 10],
    'F_major': [5, 7, 9, 10, 0, 2, 4],
    'Bb_major': [10, 0, 2, 3, 5, 7, 9],
    'Eb_major': [3, 5, 7, 8, 10, 0, 2],
    'Ab_major': [8, 10, 0, 1, 3, 5, 7],
    'Db_major': [1, 3, 5, 6, 8, 10, 0],
    'Gb_major': [6, 8, 10, 11, 1, 3, 5],
    'A_minor': [9, 11, 0, 2, 4, 5, 7],      
    'E_minor': [4, 6, 7, 9, 11, 0, 2],      
    'D_minor': [2, 4, 5, 7, 9, 10, 0],
    'B_minor': [11, 1, 2, 4, 6, 7, 9],
    'F#_minor': [6, 8, 9, 11, 1, 2, 4],
    'C#_minor': [1, 3, 4, 6, 8, 9, 11],
    'G_minor': [7, 9, 10, 0, 2, 3, 5],
    'C_minor': [0, 2, 3, 5, 7, 8, 10],
    'F_minor': [5, 7, 8, 10, 0, 1, 3],
}


PROGRESSIONS = [
    {
        'name': 'I-IV-V-I (Classic)',
        'chords': ['I', 'IV', 'V', 'I']
    },
    {
        'name': 'I-vi-IV-V (50s progression)',
        'chords': ['I', 'vi', 'IV', 'V']
    },
    {
        'name': 'I-V-vi-IV (Pop progression)',
        'chords': ['I', 'V', 'vi', 'IV']
    },
    {
        'name': 'vi-IV-I-V (Sad to hopeful)',
        'chords': ['vi', 'IV', 'I', 'V']
    },
    {
        'name': 'I-IV-vi-V (Optimistic)',
        'chords': ['I', 'IV', 'vi', 'V']
    },
]

def get_chord_notes(chord_symbol, root_note=60):
    

    numeral_map = {
        'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
        'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6
    }
    
    
    is_minor = chord_symbol[0].islower()
    numeral = chord_symbol.upper()
    
    if numeral not in numeral_map:
        return [0, 4, 7]  
    
    degree = numeral_map[numeral]
    
    scale_degrees_in_c = [0, 2, 4, 5, 7, 9, 11]
    chord_root = scale_degrees_in_c[degree]
    
    
    if is_minor:
       
        return [chord_root, (chord_root + 3) % 12, (chord_root + 7) % 12]
    else:
   
        return [chord_root, (chord_root + 4) % 12, (chord_root + 7) % 12]


def get_chord_notes_old(root, chord_type='major'):

    if chord_type == 'major':
        return [root, root + 4, root + 7]  
    elif chord_type == 'minor':
        return [root, root + 3, root + 7]  
    return [root, root + 4, root + 7]

def get_progression_chords(scale_name='C_major', progression_idx=2):

    scale = SCALES.get(scale_name, SCALES['C_major'])
    prog = PROGRESSIONS[progression_idx % len(PROGRESSIONS)]
    
    chords = []
    for degree in prog:
        root = scale[degree % len(scale)]
 
        chord_type = 'minor' if degree in [5, 2] else 'major'
        chords.append(get_chord_notes(root, chord_type))
    
    return chords

def get_melody_notes(scale_name='C_major', num_notes=8):
   
    scale = SCALES.get(scale_name, SCALES['C_major'])

    extended = []
    for octave in [-12, 0, 12]:
        extended.extend([n + octave for n in scale])
    return sorted(set(extended))

def note_fits_chord(note, chord_notes):

    note_class = note % 12
    chord_classes = [n % 12 for n in chord_notes]
    return note_class in chord_classes
