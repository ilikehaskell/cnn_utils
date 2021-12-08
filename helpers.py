import numpy as np
import PIL
import IPython.display as ipd

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def play_sound():
    beep = np.sin(2*np.pi*400*np.arange(100000*2)/10000)
    ipd.Audio(beep, rate=100000, autoplay=True)