import numpy as np

def adjust_image(img):
    if(len(np.shape(img))>2):
        min_values = np.reshape(np.amin(img, axis=(1,2)),(-1,1,1))
        img_adj = img-min_values
        max_values = np.reshape(np.amax(img_adj, axis=(1,2)),(-1,1,1))
        img_adj = (img_adj.astype(np.float32)/max_values)*255
    else:
        min_values = np.amin(img)
        img_adj = img-min_values
        max_values = np.amax(img_adj)
        img_adj = (img_adj.astype(np.float32)/max_values)*255

    return img_adj.astype(np.uint8)
