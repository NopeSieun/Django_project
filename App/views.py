from django.contrib.auth.decorators import login_required

import shutil
import tempfile
import pandas as pd
import time

import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg', depending on your system

import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.utils import set_determinism, first
import skimage

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    GaussianSmoothd,
    RandRotate90d,
    ToTensord,
    RandSpatialCropd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGaussianNoised,
)

from monai.config import print_config
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from sklearn.metrics import roc_auc_score
#############################
from django.shortcuts import render
import io
import os
import json

from .models import Info

# from _mysql import connection

from django.http import HttpResponse
from django.http.response import HttpResponseRedirect

from torchvision import models
from torchvision import transforms
import torch
from PIL import Image
from django.conf import settings

from io import BytesIO

from django import forms
import base64
from .forms import UploadFileForm, UploadFileForm2
import nibabel as nib
import numpy as np
import imageio
from .utils import get_plot, get_subtraction, origin_pic, new_pic

from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.core.files.uploadedfile import UploadedFile
from django.core.files.storage import FileSystemStorage

import einops
import monai
from monai.inferers import sliding_window_inference
from monai.transforms import Resize
from scipy.stats import mode

from skimage import measure

import torch
import glob

image_uri = None
predicted_volume = None
chart = None
chart2 = None
seg = None
form1 = None
form2 = None

context = {
    'form1': form1,
    'form2': form2,
    'image_uri': image_uri,
    'seg': seg,
    'chart': chart,
    'chart2': chart2,
    'predicted_volume': predicted_volume,
}

device = "cpu"
model = monai.networks.nets.SwinUNETR(img_size=(128, 128, 64),
                                      in_channels=1, out_channels=2,
                                      feature_size=24).to(device)
model.load_state_dict(torch.load("staticfiles/swinunetr_tmp.pth", map_location=torch.device('cpu')))


def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'L')
    return img


def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "PNG")  # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:pil_img/PNG;base64,' + data64.decode('utf-8')


def register_submit(img, result):
    info = Info()
    info.image = img

    if result == None:
        info.result = "We can't calculate. Sorry."
    else:
        info.result = result

    info.save()
def ListFunc(request):
    data = Info.objects.all()
    return render(request, 'App/imgList.html', {'data': data})

def result_info(request):
    global context
    return render(request, 'App/results.html', context)

@login_required
def index(request):
    global image_uri
    global predicted_volume
    global chart
    global chart2
    global seg
    global form1
    global form2
    global context

    if request.method == 'POST':
        form1 = UploadFileForm(request.POST, request.FILES)
        form2 = UploadFileForm2(request.POST, request.FILES)

        if form1.is_valid():

            nii_file_img = form1.cleaned_data['file']

            image = nib.load(nii_file_img.temporary_file_path()).get_data()
            h = len(image[0, 0]) // 2
            data = image[:, :, h]  # 50]
            data_for_chart = image[:, :, h]

            img_array = np.ascontiguousarray(np.array(data))  # array, float32
            img_array = img_array.astype('float')

            image_array = Image.fromarray(img_array)
            print(type(image_array))  # PIL mode=F size=128x128

            np_image = np.asarray(data_for_chart)
            pil_image = to_image(np_image)

            chart = get_plot(np_image)

            nii_file_seg = form1.cleaned_data['seg_file']

            seg_image = nib.load(nii_file_seg.temporary_file_path()).get_data()
            seg_data = seg_image[:, :, 20]

            seg_array = np.ascontiguousarray(np.array(seg_data))  # array, float32
            image_array = Image.fromarray(seg_array)  # PIL mode=F size=128x128

            np_seg = np.asarray(seg_data)
            pil_seg = to_image(np_seg)
            seg = to_data_uri(pil_seg)

            chart2 = get_subtraction(np_image, np_seg)

            predicted_volume = "This mode doesn't calculate volume."
            register_submit(chart2, predicted_volume)

            form2 = UploadFileForm2()


        elif form2.is_valid():

            nii_file_img = form2.cleaned_data['file2']

            # 저장
            fs = FileSystemStorage()
            fs.save("origin_nifti.nii", nii_file_img)
            plist = glob.glob("media/origin_nifti.nii")

            valid_idx = np.arange(0, len(plist))
            data_dicts = [
                {
                    "image1": os.path.join(plist[idx]),
                }
                for idx in valid_idx
            ]
            valid_Data = data_dicts

            test_transforms = Compose(
                [
                    LoadImaged(keys=["image1"]),
                    EnsureChannelFirstd(keys=["image1"]),
                    ToTensord(keys=["image1"]),
                    ScaleIntensityd(
                        keys=["image1"],
                        minv=0.0,
                        maxv=1.0,
                    ),
                ]
            )

            test_ds = Dataset(
                data=valid_Data,
                transform=test_transforms,
            )

            test_loader = DataLoader(
                test_ds, batch_size=1, shuffle=False,
            )

            with torch.no_grad():
                for step, batch in enumerate(test_loader):
                    val_inputs = (batch["image1"]).to(device)
                    sz = batch["image1"].shape[2:]
                    R = Resize(spatial_size=(sz[0] * 2, sz[1] * 2, sz[2] * 4))
                    R0 = Resize(spatial_size=(sz[0], sz[1], sz[2]), mode="nearest")

                    K = R(val_inputs[0])
                    kz = K.shape[1:]
                    x1 = int(0.5 * (kz[0] - 192))
                    y1 = int(0.5 * (kz[1] - 192))
                    z1 = int(0.5 * (kz[2] - 96))
                    val_inputs = K[:, x1:x1 + 192, y1:y1 + 192, z1:z1 + 96].unsqueeze(0)

                    val_outputs1 = sliding_window_inference(
                        val_inputs, [128, 128, 64], 1, model, overlap=0.25, mode='gaussian')

                    val_outputs = val_outputs1.softmax(1)

                    chart = origin_pic(val_inputs, val_outputs)

                    r0 = val_outputs[0, 1, :, :, :].detach().cpu() > .05
                    r = val_outputs[0, 1, :, :, :].detach().cpu() > .25

                    r0[val_inputs[0, 0] == 0] = 0
                    r[val_inputs[0, 0] == 0] = 0

                    SN = np.zeros_like(r)
                    L0, N0 = measure.label(r0, connectivity=3, return_num=True)
                    L, N1 = measure.label(r, connectivity=3, return_num=True)

                    if N1 > 1:
                        a, b = (mode(L[L > 0]))
                        SN[L == a[0]] = 1
                        L[L == a[0]] = 0
                        a, b = (mode(L[L > 0]))
                        SN[L == a[0]] = 1
                    for sn in range(1, N0 + 1):
                        if np.sum(SN * (L0 == sn)) == 0:
                            L0[L0 == sn] = 0

                    chart2 = new_pic(L0)

                    val_outputs_ = np.zeros((sz[0] * 2, sz[1] * 2, sz[2] * 4))
                    val_outputs_[x1:x1 + 192, y1:y1 + 192, z1:z1 + 96] = val_outputs[0, 1, :, :,
                                                                         :].detach().cpu().numpy()

                    val_masks_ = np.zeros((sz[0] * 2, sz[1] * 2, sz[2] * 4))
                    val_masks_[x1:x1 + 192, y1:y1 + 192, z1:z1 + 96] = L0 > 0

                    val_masks_ = R0(np.expand_dims(val_masks_, 0))
                    val_outputs_ = R0(np.expand_dims(val_outputs_, 0))

                    os.mkdir("C:/your_nifti")
                    #dpath = "C:/Users/user/Downloads"
                    dpath = "C:/your_nifti"
                    tmp = nib.Nifti1Image(100 * val_outputs_[0].cpu().numpy(),
                                          batch['image1_meta_dict']['affine'][0].numpy())
                    nib.save(tmp, os.path.join(dpath, 'your_nii_prob_SN.nii.gz').replace("\\","/"))
                    tmp = nib.Nifti1Image(val_masks_[0].cpu().numpy(), batch['image1_meta_dict']['affine'][0].numpy())
                    nib.save(tmp, os.path.join(dpath, 'your_nii_mask_SN.nii.gz').replace("\\","/"))  # 적어도 애는 저장이 되야 함

            predicted_volume = np.rot90(np.sum(L0 > 0, axis=2)).sum()
            register_submit(chart2, predicted_volume)
            form1 = UploadFileForm()

    else:

        form1 = UploadFileForm()
        form2 = UploadFileForm2()

    context = {
        'form1': form1,
        'form2': form2,
        'image_uri': image_uri,
        'seg': seg,
        'chart': chart,
        'chart2': chart2,
        'predicted_volume': predicted_volume,
    }

    result_info(request)
    return render(request, 'App/index.html', context)