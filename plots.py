from .helpers import tensor_to_image

def get_images_with_labels(ilp):
    imgs, labels, preds = list(zip(*ilp))
    labels = [str(int(label)+1) for label in labels]
    preds = [str(int(pred)+1) for pred in preds]
    plot_labels = [''.join([il, ip]) for il, ip in zip(labels, preds)]
    imgs = [tensor_to_image(img.permute(1,2,0).cpu()) for img in imgs]
    return imgs, plot_labels

