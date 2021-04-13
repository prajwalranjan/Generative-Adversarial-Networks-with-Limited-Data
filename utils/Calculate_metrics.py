# calculates  fid, kid,is

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from scipy.linalg import sqrtm
from tensorflow import convert_to_tensor
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from keras.datasets import cifar10
from skimage.transform import resize
import cv2
import os
import tensorflow
from tensorflow.python.ops import array_ops 
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow import map_fn


def kernel_classifier_distance_and_std_from_activations(real_activations,
                                                        generated_activations,
                                                        max_block_size=1024,
                                                        dtype=None):
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)
  #real_activations.shape[1].assert_is_compatible_with(
  #    generated_activations.shape[1])

  if dtype is None:
    dtype = real_activations.dtype
    assert generated_activations.dtype == dtype
  else:
    real_activations = math_ops.cast(real_activations, dtype)
    generated_activations = math_ops.cast(generated_activations, dtype)

  # Figure out how to split the activations into blocks of approximately
  # equal size, with none larger than max_block_size.
  n_r = array_ops.shape(real_activations)[0]
  n_g = array_ops.shape(generated_activations)[0]

  n_bigger = math_ops.maximum(n_r, n_g)
  n_blocks = math_ops.to_int32(math_ops.ceil(n_bigger / max_block_size))

  v_r = n_r // n_blocks
  v_g = n_g // n_blocks

  n_plusone_r = n_r - v_r * n_blocks
  n_plusone_g = n_g - v_g * n_blocks

  sizes_r = array_ops.concat([
      array_ops.fill([n_blocks - n_plusone_r], v_r),
      array_ops.fill([n_plusone_r], v_r + 1),
  ], 0)
  sizes_g = array_ops.concat([
      array_ops.fill([n_blocks - n_plusone_g], v_g),
      array_ops.fill([n_plusone_g], v_g + 1),
  ], 0)

  zero = array_ops.zeros([1], dtype=dtypes.int32)
  inds_r = array_ops.concat([zero, math_ops.cumsum(sizes_r)], 0)
  inds_g = array_ops.concat([zero, math_ops.cumsum(sizes_g)], 0)

  dim = math_ops.cast(real_activations.shape[1], dtype)

  def compute_kid_block(i):
    'Compute the ith block of the KID estimate.'
    r_s = inds_r[i]
    r_e = inds_r[i + 1]
    r = real_activations[r_s:r_e]
    m = math_ops.cast(r_e - r_s, dtype)

    g_s = inds_g[i]
    g_e = inds_g[i + 1]
    g = generated_activations[g_s:g_e]
    n = math_ops.cast(g_e - g_s, dtype)

    k_rr = (math_ops.matmul(r, r, transpose_b=True) / dim + 1)**3
    k_rg = (math_ops.matmul(r, g, transpose_b=True) / dim + 1)**3
    k_gg = (math_ops.matmul(g, g, transpose_b=True) / dim + 1)**3
    return (-2 * math_ops.reduce_mean(k_rg) +
            (math_ops.reduce_sum(k_rr) - math_ops.trace(k_rr)) / (m * (m - 1)) +
            (math_ops.reduce_sum(k_gg) - math_ops.trace(k_gg)) / (n * (n - 1)))

  ests = map_fn(
      compute_kid_block, math_ops.range(n_blocks), dtype=dtype, back_prop=False)

  mn = math_ops.reduce_mean(ests)

  # nn_impl.moments doesn't use the Bessel correction, which we want here
  n_blocks_ = math_ops.cast(n_blocks, dtype)
  var = control_flow_ops.cond(
      math_ops.less_equal(n_blocks, 1),
      lambda: array_ops.constant(float('nan'), dtype=dtype),
      lambda: math_ops.reduce_sum(math_ops.square(ests - mn)) / (n_blocks_ - 1))

  return mn, math_ops.sqrt(var / n_blocks_)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return numpy.array(images)


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid_kid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    #print(act1.shape)#debug
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    kact1 = convert_to_tensor(act1)
    kact2 = convert_to_tensor(act2)
    kid, _ = kernel_classifier_distance_and_std_from_activations(real_activations=kact1, generated_activations=kact2) 
    return fid, kid



def calculate_is(model, images, n_split=10, eps=1E-16):
  # enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std





def fid(model, reals, fakes):
  #load stored images
  #fakes = load_images_from_folder(dir)

  # load cifar10 test images
  #_, (images, _) = cifar10.load_data()
  #reals = images[:num]
  #reals.shape

  # prepare the inception v3 model
  #model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))


  # define two  collections of images
  images1 = reals
  #images1 = images1.reshape((10,32,32,3))
  images2 = fakes
  #images2 = images2.reshape((10,32,32,3))
  #print('Prepared', images1.shape, images2.shape)


  # convert integer to floating point values
  images1 = images1.astype('float32')
  images2 = images2.astype('float32')
  # resize images
  images1 = scale_images(images1, (299,299,3))
  images2 = scale_images(images2, (299,299,3))
  #print('Scaled', images1.shape, images2.shape)
  # pre-process images
  images1 = preprocess_input(images1)
  images2 = preprocess_input(images2)


  # fid between images1 and images1
  #fid = calculate_fid(model, images1, images1)
  #print('FID (same): %.3f' % fid)
  # fid between images1 and images2
  fid, kid = calculate_fid_kid(model, images1, images2)
  is_mean, is_std = calculate_is(model,images2)
  #print('fid is %.3f' %fid)
  #print('inception score : %.3f +- %.3f' %(is_mean, is_std))
  return fid, kid

## MAIN FUNCTION

def metrics(dir = 'out', num= 128):  
  #load stored images
  fakes = load_images_from_folder(dir)

  # load cifar10 test images
  _, (images, _) = cifar10.load_data()
  reals = images[:num]
  #reals.shape

  # prepare the inception v3 model
  model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
  model2 = InceptionV3()
  f,k = fid(model, reals, fakes)
  i_mean, i_std = calculate_is(model2, fakes)
  
  print("Results for %d generated images:" %num)
  print('fid is %.3f' %f)
  print('kid is %.3f' %k)
  print('inception score : %.3f +- %.3f' %(i_mean, i_std)) 
