#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy, scipy, IPython.display as ipy, matplotlib.pyplot as ply
import seaborn
import librosa, librosa.display
ply.rcParams['figure.figsize'] = (14, 5)


# In[73]:


audio3sec = 'sss3orig.wav'
x, sr = librosa.load (audio3sec)


# In[74]:


ipy.Audio(x, rate=sr)


# In[94]:


ply.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[76]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
ply.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')


# In[77]:


hop_length = 100
onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length)


# In[78]:


ply.plot(onset_env)
ply.xlim(0, len(onset_env))


# In[79]:


onset_death = librosa.onset.onset_detect(x,
                                           sr=sr, units='samples', 
                                           hop_length=hop_length, 
                                           backtrack=False,
                                           pre_max=20,
                                           post_max=20,
                                           pre_avg=100,
                                           post_avg=100,
                                           delta=0.2,
                                           wait=0)


# In[80]:


onset_death


# In[81]:


onset_boundary = numpy.concatenate([[0], onset_death, [len(x)]])


# In[82]:


print (onset_boundary)


# In[83]:


onset_seconds = librosa.samples_to_time(onset_boundary, sr=sr)


# In[84]:


onset_seconds


# In[85]:


librosa.display.waveplot(x, sr=sr)
ply.vlines(onset_seconds, -1, 1, color='r')


# In[86]:


def find_pitch(segment, sr, fmin=50.0, fmax=2000.0):

    r = librosa.autocorrelate(segment)
 
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    

    i = r.argmax()
    f0 = float(sr)/i
    return f0


# In[87]:


def make_sine(f0, sr, n_duration):
    n = numpy.arange(n_duration)
    return 0.2*numpy.sin(2*numpy.pi*f0*n/float(sr))


# In[88]:


def find_pitch_and_make_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    f0 = find_pitch(x[n0:n1], sr)
    return make_sine(f0, sr, n1-n0)


# In[89]:


y = numpy.concatenate([
    find_pitch_and_make_sine(x, onset_boundary, i, sr=sr)
    for i in range(len(onset_boundary)-1)
])


# In[90]:


cqt = librosa.cqt(y, sr=sr)


# In[92]:


librosa.display.specshow(abs(cqt), sr=sr, x_axis='time', y_axis='cqt_hz')


# In[97]:


desiredY=interp1('time','cqt_hz',0.5);


# In[ ]:





# In[ ]:





# In[ ]:




