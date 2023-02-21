import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import torch

from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage import transform as tf

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(img):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    plt.title('Brain Image')
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.tight_layout()
    graph = get_graph()
    return graph

def get_subtraction(img1, img2):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    plt.title('Brain Image')
    plt.imshow(img1-img2, cmap='gray')
    plt.show()
    plt.tight_layout()
    graph = get_graph()
    return graph

def origin_pic(input, output):

    plt.switch_backend('AGG')
    plt.figure(figsize=(6,6))
    
    plt.subplot(121)
    plt.imshow(torch.rot90(torch.sum(input[0,0,:,:,:].detach().cpu(),dim=2)),cmap='gray')
    plt.title("Origin")
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(torch.rot90(torch.sum(output[0,1,:,:,:].detach().cpu(),dim=2)),cmap='jet')
    plt.title('DL output')
    plt.axis('off')

    plt.show()
    plt.tight_layout()
    graph = get_graph()

    return graph


def new_pic(L0):

    plt.switch_backend('AGG')
    plt.figure(figsize=(6,6))
    
    plt.imshow(np.rot90(np.sum(L0>0,axis=2)),cmap='jet',interpolation='nearest')
    plt.title('DL output-refined')
    plt.axis('off')

    plt.show()
    plt.tight_layout()
    graph = get_graph()

    return graph