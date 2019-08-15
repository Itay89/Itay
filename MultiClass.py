#!/usr/bin/env python
# coding: utf-8

# This notebook contains code for model training on CelebA dataset. Please refer to Data_prepration.ipynb notebook to see the preprocessing steps for the dataset.

# ## Boiler Plate

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np
from fastai.vision import *
import matplotlib.pyplot as plt
from pathlib import Path
pd.set_option('display.max_columns', 500)


# ## Creating a databunch

# In[3]:


path = 'faces/'


# In[4]:


## Function to filter validation samples
def validation_func(x):
    return 'validation' in x


# In[5]:


tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate=30, max_lighting=0.3)


# In[6]:


ImageList.from_csv(path, csv_name='labels1.csv', cols='image_id')


# In[7]:


src = (ImageList.from_csv(path, csv_name='labels1.csv', cols='image_id')
       .split_by_valid_func(validation_func)
       .label_from_df(cols='tags',label_delim=' '))

data = (src.transform(tfms, size=128)
       .databunch(bs=256).normalize(imagenet_stats))


# In[8]:


print(data.c,'\n',data.classes)


# In[9]:


data


# In[10]:


data.show_batch(rows=2, figsize=(20,12))


# ## Model

# We are going to use a resnet50 pretrained model and do transfer learning on CelebA dataset.

# In[ ]:


arch = models.resnet50


# In[12]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
acc_03 = partial(accuracy_thresh, thresh=0.3)
acc_04 = partial(accuracy_thresh, thresh=0.4)
acc_05 = partial(accuracy_thresh, thresh=0.5)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, acc_03, acc_04, acc_05, f_score])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-2


# In[ ]:


learn.fit_one_cycle(1, slice(lr))


# In[ ]:


learn.fit_one_cycle(4, slice(lr))


# In[ ]:


learn.save('ff_stage-1-rn50')


# Tuning the whole model

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.save('ff_stage-2-rn50')


# ## Further Training

# In[ ]:


data = (src.transform(tfms, size=256)
       .databunch(bs=64).normalize(imagenet_stats))


# In[ ]:


acc_05 = partial(accuracy_thresh, thresh=0.5)
f_score = partial(fbeta, thresh=0.5)
learn = create_cnn(data, models.resnet50, pretrained=False,metrics=[acc_05, f_score])
learn.load("ff_stage-2-rn50")


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(1, slice(lr))


# In[ ]:


learn.save('ff_stage-1-256-rn50')


# ## Visualize

# In[ ]:


learn = create_cnn(data, models.resnet50, pretrained=False)
learn.load("ff_stage-1-256-rn50")


# In[ ]:


m = learn.model.eval();


# In[ ]:


idx=5
x,y = data.valid_ds[idx]
x.show()
data.valid_ds.y[idx]


# In[ ]:


xb, _ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()


# In[ ]:


from fastai.callbacks.hooks import *


# In[ ]:


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            #preds[0,str(data.valid_ds.y[idx])].backward()
    return hook_a,hook_g


# In[ ]:


hook_a,hook_g = hooked_backward()


# In[ ]:


acts  = hook_a.stored[0].cpu()
acts.shape


# In[ ]:


avg_acts = acts.mean(0)
avg_acts.shape


# In[ ]:


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,256,256,0),
              interpolation='bilinear', cmap='magma');


# In[ ]:


avg_acts


# In[ ]:


show_heatmap(avg_acts)


# In[ ]:


idx=700
x,y = data.valid_ds[idx]
xb, _ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()

hook_a,hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu()
avg_acts = acts.mean(0)
show_heatmap(avg_acts)


# In[ ]:


avg_acts


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




