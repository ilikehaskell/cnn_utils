import numpy as np, pandas as pd
import PIL
import IPython.display as ipd

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def save_as_submission_csv(model, test_holder,test_csv, save_path):
    results = test_holder.get_input_label_preds(model)
    preds = [( c+1) for a,b,c in results]
    test_df = pd.read_csv(test_csv) 
    test_df['label'] = preds
    test_df.to_csv(save_path, index=False)
    return test_df

# def play_sound():
#     beep = np.sin(2*np.pi*400*np.arange(100000*2)/10000)
#     ipd.Audio(beep, rate=100000, autoplay=True)